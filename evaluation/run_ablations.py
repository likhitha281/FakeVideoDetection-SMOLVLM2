#!/usr/bin/env python3
"""
run_ablations.py — runs 3 short-budget ablations against your baseline config
and evaluates each on the same held-out val subsample, so results are directly
comparable. Reuses all classes and CONFIG from finetune_lora_latest.py — no
logic is duplicated or re-typed, only the training loop is assembled fresh
(the full loop in your script is guarded under `if __name__ == "__main__":`
and can't be called directly).

Ablations (see report Section 4.4 for justification of each):
  1. baseline        — your current config, unchanged
  2. single_frame    — num_frames=1 instead of 4 (temporal info ablation)
  3. no_contrastive  — contrastive_loss_weight=0.0 instead of 0.1 (fusion ablation)
  4. lower_lr        — learning_rate=1e-5 instead of 5e-5 (optimization ablation)

Each variant trains for a fixed short step budget (ABLATION_MAX_STEPS) rather
than full epochs, since a full 3-epoch run per variant isn't feasible on a
single Colab GPU. This is called out explicitly in the report as a
compute-driven methodological choice.

Usage:
    python run_ablations.py
"""
import copy, json, math, random, gc, inspect
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import TrainingArguments

import finetune_lora_latest as ft

BASE_CONFIG = ft.CONFIG
ABLATION_MAX_STEPS = 350          # short, fixed budget — same for every variant
EVAL_SUBSAMPLE_SIZE = 200         # smaller than the Table-2 eval (400) to keep this fast
NUM_WORKERS = 4

ABLATIONS = [
    ("baseline",       {}),
    ("single_frame",   {"num_frames": 1}),
    ("no_contrastive", {"contrastive_loss_weight": 0.0}),
    ("lower_lr",       {"learning_rate": 1e-5}),
]

# ---- Load raw data + reproduce the exact seed=42 split once, shared across all variants ----
data_root = Path(BASE_CONFIG["data_root"])
jp = data_root / BASE_CONFIG["train_json"]
with open(jp) as f:
    raw_data_full = json.load(f)

random.seed(42)
random.shuffle(raw_data_full)
val_ratio = BASE_CONFIG.get("val_ratio", 0.10)
n_val = max(1, int(len(raw_data_full) * val_ratio))
val_data_full = raw_data_full[:n_val]
train_data_full = raw_data_full[n_val:]
print(f"✅ Reproduced split: {len(train_data_full)} train / {len(val_data_full)} val (seed=42)")


def stratified_subsample(data, max_samples, seed=42):
    if max_samples is None or max_samples >= len(data):
        return data
    reals = [x for x in data if str(x.get("dataset_label") or "").lower() == "real"]
    fakes = [x for x in data if str(x.get("dataset_label") or "").lower() != "real"]
    rng = random.Random(seed)
    rng.shuffle(reals); rng.shuffle(fakes)
    frac = max_samples / len(data)
    n_real = max(1, round(len(reals) * frac))
    n_fake = max(1, max_samples - n_real)
    return reals[:n_real] + fakes[:n_fake]


eval_subset = stratified_subsample(val_data_full, EVAL_SUBSAMPLE_SIZE)
print(f"   → shared eval subsample: {len(eval_subset)} videos, used identically for every variant")

# FIXED_CLS_WEIGHTS is a module-level global in finetune_lora_latest.py that's only computed
# inside its `if __name__ == "__main__":` block (which we never execute, since we just import
# the module for its classes/CONFIG). Left as its default None, the classification loss would
# silently run UNweighted for every ablation variant here, which would not match your baseline
# training behavior. Replicate the exact same computation against train_data_full so class
# weighting is identical to your real training run.
_reals = sum(1 for x in train_data_full if str(x.get("dataset_label") or "").lower() == "real")
_fakes = len(train_data_full) - _reals
_total = _reals + _fakes
ft.FIXED_CLS_WEIGHTS = torch.tensor(
    [_total / (2.0 * _reals), _total / (2.0 * _fakes)], dtype=torch.float32
)
print(f"⚖️  FIXED_CLS_WEIGHTS set to match baseline training: "
      f"real={ft.FIXED_CLS_WEIGHTS[0]:.3f}, fake={ft.FIXED_CLS_WEIGHTS[1]:.3f}")


@torch.no_grad()
def evaluate(model, val_dataset, collator, batch_size=4, num_workers=NUM_WORKERS):
    model.eval()
    loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collator,
        pin_memory=True, persistent_workers=(num_workers > 0),
    )
    tp = fp = fn = tn = correct = total = 0
    dev = next(model.parameters()).device
    for batch in loader:
        batch = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in batch.items()}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(**{k: v for k, v in batch.items() if k != "cls_label"})
        preds = out["cls_logits"].argmax(dim=-1).cpu()
        labels = batch["cls_label"].cpu()
        correct += (preds == labels).sum().item()
        total += len(preds)
        tp += ((preds == 0) & (labels == 0)).sum().item()
        fp += ((preds == 0) & (labels == 1)).sum().item()
        fn += ((preds == 1) & (labels == 0)).sum().item()
        tn += ((preds == 1) & (labels == 1)).sum().item()
    acc = correct / max(1, total)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    return {"accuracy": acc, "real_precision": prec, "real_recall": rec, "n": total}


def build_training_args(cfg, output_dir):
    desired = dict(
        output_dir=output_dir,
        max_steps=ABLATION_MAX_STEPS,          # short fixed budget, overrides num_train_epochs
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        warmup_ratio=cfg["warmup_ratio"],
        max_grad_norm=cfg["max_grad_norm"],
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        bf16=cfg["bf16"],
        gradient_checkpointing=cfg["gradient_checkpointing"],
        use_cache=False,
        dataloader_num_workers=cfg["dataloader_num_workers"],
        dataloader_persistent_workers=(cfg["dataloader_num_workers"] > 0),
        dataloader_pin_memory=False,
        dataloader_prefetch_factor=1,
        logging_steps=25,
        save_strategy="no",     # we evaluate in-process right after training; no need to persist
        eval_strategy="no",
        remove_unused_columns=False,
        report_to="none",
        disable_tqdm=False,
    )
    valid_params = set(inspect.signature(TrainingArguments.__init__).parameters.keys())
    ta_kwargs = {k: v for k, v in desired.items() if k in valid_params}
    dropped = sorted(set(desired) - set(ta_kwargs))
    if dropped:
        if "warmup_ratio" in dropped and "warmup_steps" in valid_params:
            ta_kwargs["warmup_steps"] = cfg["warmup_ratio"]
    return TrainingArguments(**ta_kwargs)


def run_variant(name, overrides):
    print(f"\n{'='*70}\n=== Ablation: {name}  (overrides={overrides}) ===\n{'='*70}")
    cfg = copy.deepcopy(BASE_CONFIG)
    cfg.update(overrides)
    cfg["output_dir"] = f"{BASE_CONFIG['output_dir']}_ablation_{name}"

    # Your baseline CONFIG runs with gradient_checkpointing=False, which is why the
    # original full run was already close to the GPU's ceiling. Halving the per-device
    # batch size (and doubling grad-accum to keep the *effective* batch size at 16,
    # identical to baseline) gives real headroom without changing what's being ablated.
    cfg["per_device_train_batch_size"] = max(1, cfg["per_device_train_batch_size"] // 2)
    cfg["gradient_accumulation_steps"] = cfg["gradient_accumulation_steps"] * 2

    model = ft.MultiTaskSmolVLM(cfg["model_id"], cfg["head_lr_multiplier"], cfg["vision_lr_multiplier"])
    train_dataset = ft.VideoFakeClueDataset(
        train_data_full, model.processor, ml=cfg["max_seq_length"], num_frames=cfg["num_frames"]
    )

    # Schedule knobs, recomputed against the SHORT ablation budget (not full-epoch step count)
    # so the contrastive ramp/boost/decay schedule stays internally consistent for this run.
    ft.CONFIG = cfg  # MultiTaskTrainer's compute_loss reads module-level CONFIG — point it at this variant
    cfg["contrastive_ramp_steps"] = max(1, int(0.10 * ABLATION_MAX_STEPS))
    cfg["boost_after_step"] = max(1, int(0.35 * ABLATION_MAX_STEPS))
    cfg["late_aux_decay_start"] = max(1, int(0.65 * ABLATION_MAX_STEPS))
    cfg["generation_sample_interval"] = max(1, int(0.5 * ABLATION_MAX_STEPS))  # infrequent; not the focus here

    args = build_training_args(cfg, cfg["output_dir"])

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.empty_cache(); gc.collect()
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    trainer = ft.MultiTaskTrainer(
        model=model, args=args, train_dataset=train_dataset, eval_dataset=None,
        data_collator=ft.CustomDataCollator(model.processor.tokenizer),
    )
    model._trainer_ref = trainer
    trainer.add_callback(ft.UnfreezeCallback())

    torch.cuda.empty_cache(); gc.collect()  # match the original script's pre-train() safety call
    trainer.train()

    torch.cuda.empty_cache(); gc.collect()  # release optimizer/activation memory before eval
    eval_dataset = ft.VideoFakeClueDataset(eval_subset, model.processor, num_frames=cfg["num_frames"])
    collator = ft.CustomDataCollator(model.processor.tokenizer)
    metrics = evaluate(model, eval_dataset, collator, batch_size=max(1, cfg["per_device_train_batch_size"]))
    print(f"   RESULT [{name}]: accuracy={metrics['accuracy']:.4f}  "
          f"real_precision={metrics['real_precision']:.4f}  "
          f"real_recall={metrics['real_recall']:.4f}  (n={metrics['n']})")

    del model, trainer
    torch.cuda.empty_cache(); gc.collect()
    ft.CONFIG = BASE_CONFIG  # restore before the next variant
    return metrics


def main():
    results = {}
    for name, overrides in ABLATIONS:
        results[name] = run_variant(name, overrides)

    print(f"\n{'='*70}\nABLATION SUMMARY (max_steps={ABLATION_MAX_STEPS}, eval n={len(eval_subset)})\n{'='*70}")
    print(f"{'Variant':<16} {'Accuracy':>10} {'Prec(real)':>12} {'Rec(real)':>12}")
    for name, m in results.items():
        print(f"{name:<16} {m['accuracy']*100:>9.1f}% {m['real_precision']*100:>11.1f}% {m['real_recall']*100:>11.1f}%")

    out_path = Path(BASE_CONFIG["output_dir"]) / "ablation_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n✅ Saved to {out_path}")


if __name__ == "__main__":
    main()

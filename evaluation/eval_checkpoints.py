#!/usr/bin/env python3
"""
eval_checkpoints.py — reload your saved per-epoch checkpoints and compute
val-set accuracy / precision / recall (real class), without retraining.

Run this from the SAME directory / environment as finetune_lora_latest.py
(it imports classes and CONFIG directly from that file, so no logic is
duplicated or re-typed). Uses the same seed=42 val split, so you get the
exact same val set that was used during training.

Usage:
    python eval_checkpoints.py
"""
import json, glob, math
from pathlib import Path
import torch
from torch.utils.data import DataLoader

import finetune_lora_latest as ft   # your training script — must be on the same path

CONFIG = ft.CONFIG

# Cap the number of val videos evaluated per checkpoint. 1,616 videos x 3 checkpoints
# x full video-frame decoding is what took 22+ minutes. A stratified subsample keeps
# the real/fake ratio intact (so precision/recall on the real class stay meaningful)
# while cutting runtime roughly proportionally. Set to None to evaluate the full val set.
EVAL_MAX_SAMPLES = 400
NUM_WORKERS = 4   # parallel video-decoding workers; tune down if Colab CPU is limited

# ---- Reproduce the exact val split used during training (seed=42) ----
data_root = Path(CONFIG["data_root"])
jp = data_root / CONFIG["train_json"]
with open(jp) as f:
    raw_data = json.load(f)

import random
random.seed(42)
random.shuffle(raw_data)
val_ratio = CONFIG.get("val_ratio", 0.10)
n_val = max(1, int(len(raw_data) * val_ratio))
val_data = raw_data[:n_val]
print(f"✅ Reproduced val split: {len(val_data)} videos (seed=42, same as training)")


def stratified_subsample(data, max_samples, seed=42):
    """Keep the real/fake ratio of `data` intact while capping total size at
    max_samples. Uses a separate RNG so it doesn't disturb the seed=42 split above."""
    if max_samples is None or max_samples >= len(data):
        return data
    reals = [x for x in data if str(x.get("dataset_label") or "").lower() == "real"]
    fakes = [x for x in data if str(x.get("dataset_label") or "").lower() != "real"]
    rng = random.Random(seed)
    rng.shuffle(reals)
    rng.shuffle(fakes)
    frac = max_samples / len(data)
    n_real = max(1, round(len(reals) * frac))
    n_fake = max(1, max_samples - n_real)
    sub = reals[:n_real] + fakes[:n_fake]
    rng.shuffle(sub)
    print(f"   → stratified subsample: {len(sub)} videos "
          f"({n_real} real, {len(sub) - n_real} fake, same ratio as full val split)")
    return sub


val_data = stratified_subsample(val_data, EVAL_MAX_SAMPLES)

# ---- Checkpoints to evaluate: (label, path) ----
# Adjust this list to whatever epoch checkpoints you actually have on Drive.
CHECKPOINTS = [
    ("epoch_1", f"{CONFIG['output_dir']}/checkpoint-910"),
    ("epoch_2", f"{CONFIG['output_dir']}/checkpoint-1820"),
    ("epoch_3", f"{CONFIG['output_dir']}/checkpoint-2730"),
]

def load_state_dict_flexible(ckpt_dir):
    """Trainer saves either model.safetensors or pytorch_model.bin depending on
    installed transformers version / save_safetensors setting — try both."""
    ckpt_dir = Path(ckpt_dir)
    safet = ckpt_dir / "model.safetensors"
    binf = ckpt_dir / "pytorch_model.bin"
    if safet.exists():
        from safetensors.torch import load_file
        return load_file(str(safet))
    if binf.exists():
        return torch.load(str(binf), map_location="cpu")
    # Some setups shard large checkpoints — fall back to any matching file
    candidates = list(ckpt_dir.glob("*.safetensors")) + list(ckpt_dir.glob("*.bin"))
    if not candidates:
        raise FileNotFoundError(f"No weights file found in {ckpt_dir}")
    if candidates[0].suffix == ".safetensors":
        from safetensors.torch import load_file
        return load_file(str(candidates[0]))
    return torch.load(str(candidates[0]), map_location="cpu")


@torch.no_grad()
def evaluate(model, val_dataset, collator, batch_size=4, num_workers=NUM_WORKERS):
    model.eval()
    # A plain Python for-loop over val_dataset[j] decodes each video's frames
    # synchronously in the main process, blocking the GPU between batches — that's
    # almost certainly what made the first run slow. A DataLoader with num_workers>0
    # decodes upcoming videos in background worker processes while the GPU is busy
    # with the current batch, which should meaningfully cut wall-clock time.
    loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collator,
        pin_memory=True, persistent_workers=(num_workers > 0),
    )

    tp = fp = fn = tn = 0
    correct = total = 0
    dev = next(model.parameters()).device

    for batch in loader:
        batch = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in batch.items()}
        # Trainer used TrainingArguments(bf16=True), which wraps every forward pass in
        # torch.autocast(dtype=bfloat16) — that's what reconciled the bf16 base model
        # output against the float32 cls_head/contrastive_* weights during training.
        # Without this context here, the same mixed dtypes raise a RuntimeError.
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
    precision_real = tp / max(1, tp + fp)
    recall_real = tp / max(1, tp + fn)
    return {"accuracy": acc, "real_precision": precision_real,
            "real_recall": recall_real, "n_eval_samples": total}


def main():
    results = {}
    for label, ckpt_path in CHECKPOINTS:
        if not Path(ckpt_path).exists():
            print(f"⚠️  {label}: {ckpt_path} not found — skipping")
            continue

        print(f"\n=== Evaluating {label}: {ckpt_path} ===")
        # Fresh model with the SAME architecture (LoRA config, heads) as training,
        # then overwrite its weights with the checkpoint's saved state_dict.
        model = ft.MultiTaskSmolVLM(
            CONFIG["model_id"], CONFIG["head_lr_multiplier"], CONFIG["vision_lr_multiplier"],
        )
        state_dict = load_state_dict_flexible(ckpt_path)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"   (info) {len(missing)} missing keys on load (expected if base/vision frozen weights weren't saved)")
        if unexpected:
            print(f"   (info) {len(unexpected)} unexpected keys on load")
        # NOTE: do NOT force dtype=MODEL_DTYPE here — cls_head / contrastive_*_proj
        # are plain nn.Linear layers left in float32 by design (forward() casts inputs
        # to .float() before them). Only move device, leave each submodule's dtype as-is,
        # matching how the Trainer ran during actual training.
        model.to(ft.DEVICE)

        val_dataset = ft.VideoFakeClueDataset(val_data, model.processor, num_frames=CONFIG["num_frames"])
        collator = ft.CustomDataCollator(model.processor.tokenizer)

        metrics = evaluate(model, val_dataset, collator, batch_size=4)
        print(f"   accuracy={metrics['accuracy']:.4f}  "
              f"real_precision={metrics['real_precision']:.4f}  "
              f"real_recall={metrics['real_recall']:.4f}  "
              f"(n={metrics['n_eval_samples']})")
        results[label] = metrics

        del model
        torch.cuda.empty_cache()

    out_path = Path(CONFIG["output_dir"]) / "val_cls_metrics_from_checkpoints.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n✅ Saved all results to {out_path}")


if __name__ == "__main__":
    main()

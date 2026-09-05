#!/usr/bin/env python3
"""
finetune_smolvlm2_frames.py

SmolVLM2-256M Video Instruct Fine-Tuning — FRAMES VERSION.
Real/fake video classification + natural language artifact explanations.

DATA SOURCE
-----------
This version expects data produced by precompute_frames_streaming.py:

    ./exddv_frames/train.json
    ./exddv_frames/val.json
    ./exddv_frames/test.json
    ./exddv_frames/frames/train/{real,fake}/<video_stem>/f0.jpg ... f7.jpg
    ./exddv_frames/frames/val/{real,fake}/<video_stem>/f0.jpg ... f7.jpg
    ./exddv_frames/frames/test/{real,fake}/<video_stem>/f0.jpg ... f7.jpg

WHAT CHANGED VS. THE ORIGINAL VIDEO-BASED SCRIPT
--------------------------------------------------
1. No decord / video decoding anywhere. VideoFakeClueDatasetFrames reads the
   pre-extracted JPEGs directly — this is what makes the <10GB disk budget
   workflow possible (see precompute_frames_streaming.py).
2. Each train.json entry has "frames": [list of jpg paths] instead of
   "image": <video path>. There is no _find_video_path/_build_video_index
   step anymore — frame paths are already absolute/relative-correct as
   written by the precompute script.
3. VideoFakeClueDatasetFrames still picks a jittered num_frames-of-8 subset
   per __getitem__ call and applies the same flip/brightness augmentation as
   the original — so per-epoch variety is preserved even though the frame
   pool itself was extracted once.
4. GenerationMonitorCallback's sanity-check generation now reads frames
   directly from the raw entry's "frames" list instead of calling into
   decord — everything else about it (prints GT vs. prediction every N
   steps) is unchanged.
5. The old _PathOnlyDataset video-resolution sanity check (before training)
   is replaced by resolution_check(), which checks that each entry's frame
   files actually exist on disk instead of checking for a matching .mp4.
6. use_precomputed_vision / vision_features_dir are UNUSED in this version —
   this script keeps the full vision-encoder + connector gradient path
   (unlike that mode, which freezes vision and only trains heads on top of
   cached embeddings). Frames-on-disk is a storage-format change only, not a
   training-quality tradeoff.
Everything else (MultiTaskSmolVLM, loss weighting, LoRA, callbacks, trainer
setup) is unchanged from your original script.

IMPORTANT FOR SHARED GPUs:
The environment variables below are set at the very top of this file (before
torch is imported) to reduce memory fragmentation.
"""
import os, json, warnings, re, random

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoModelForImageTextToText, AutoProcessor, TrainingArguments, Trainer, TrainerCallback,
    EarlyStoppingCallback,
)
from transformers.utils import logging
import gc

try:
    from peft import LoraConfig, get_peft_model
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False

warnings.filterwarnings("ignore")
logging.set_verbosity_error()

# ================== SINGLE-GPU OPTIMIZATIONS (RTX PRO 6000 / T4) ==================
DEVICE = "cuda"
MODEL_DTYPE = torch.bfloat16
print(f"🚀 PyTorch: {torch.__version__} | Device: {DEVICE} | Dtype: {MODEL_DTYPE}")

# ================== CONFIG ==================
CONFIG = {
    "model_id": "HuggingFaceTB/SmolVLM2-256M-Instruct",
    "output_dir": "./checkpoints/smolvlm2-video-exddv-v5-lora-frames",

    # Point this at the --out_dir you passed to precompute_frames_streaming.py
    "data_root": "./exddv_frames",
    "train_json": "train.json",
    "val_json": "val.json",
    "test_json": "test.json",

    # --- Core training ---
    "bf16": True,
    "num_train_epochs": 3,
    "learning_rate": 5e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.05,
    "max_grad_norm": 1.0,
    "lr_scheduler_type": "cosine",
    "gradient_checkpointing": True,

    "per_device_train_batch_size": 4,
    "per_device_eval_batch_size": 4,
    "gradient_accumulation_steps": 2,
    "dataloader_num_workers": 2,

    "max_seq_length": 4096,

    "subset_ratio": 1.0,

    # --- Multi-task loss weights ---
    "cls_loss_weight": 1.2,
    "lm_loss_weight": 0.8,
    "contrastive_loss_weight": 0.0,
    "temperature": 0.07,
    "contrastive_dim": 256,

    "head_lr_multiplier": 2.0,
    "vision_lr_multiplier": 0.3,

    # --- LoRA ---
    "use_lora": True,
    "lora_r": 32,
    "lora_alpha": 64,
    "lora_target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],

    # --- Progressive unfreeze (no-op when use_lora=True) ---
    "vision_unfreeze_schedule": {
        0: [],
        40: ["layers.12.", "post_layernorm"],
        100: ["layers.10.", "layers.11.", "layers.12.", "post_layernorm"],
        180: ["layers.8.", "layers.9.", "layers.10.", "layers.11.", "layers.12.", "post_layernorm"],
        260: ["layers.6.", "layers.7.", "layers.8.", "layers.9.", "layers.10.", "layers.11.", "layers.12.", "post_layernorm"],
    },

    # --- Frames input ---
    # Must be <= the --num_frames_extract used in precompute_frames_streaming.py
    # (default there is 8). Dataset picks a jittered num_frames-of-N subset each call.
    "num_frames": 4,
    "frame_augment": True,

    # --- Monitoring / logging / val ---
    "log_every": 20,
    "global_step": 20,
    "val_ratio": 0.10,  # fallback only, used if val.json is missing

    # --- Loss schedule controls ---
    "contrastive_ramp_steps": None,
    "boost_after_step": None,
    "boost_contr_factor": 1.0,
    "boost_cls_factor": 1.1,
    "late_aux_decay_start": None,
    "late_cls_weight": None,

    # --- Early stopping / best-checkpoint tracking ---
    "save_total_limit": 3,
    "early_stopping_patience": 2,
    "val_cls_acc_max_samples": 600,

    # --- Lightweight generation sampling ---
    "generation_sample_interval": None,   # -> ~2% of total steps
    "generation_max_new_tokens": 48,
    "generation_sample_temperature": 0.7,
}

FIXED_CLS_WEIGHTS = None


# ================== MULTI-TASK MODEL (unchanged from original) ==================
class MultiTaskSmolVLM(nn.Module):
    def __init__(self, model_id, head_lr_multiplier=5.0, vision_lr_multiplier=0.2):
        super().__init__()
        self.base = AutoModelForImageTextToText.from_pretrained(
            model_id, torch_dtype=MODEL_DTYPE, attn_implementation="sdpa"
        )
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.head_lr_multiplier = head_lr_multiplier
        self.vision_lr_multiplier = vision_lr_multiplier

        self.vision_unfreeze_schedule = CONFIG.get(
            "vision_unfreeze_schedule",
            {0: [], 80: ["layers.12.", "post_layernorm"]}
        )

        self.use_lora = bool(CONFIG.get("use_lora", False))
        if self.use_lora:
            if not HAS_PEFT:
                print("⚠️  peft not installed (pip install peft). Falling back to full-param path (use_lora=False).")
                self.use_lora = False
            else:
                lora_cfg = LoraConfig(
                    r=CONFIG.get("lora_r", 32),
                    lora_alpha=CONFIG.get("lora_alpha", 64),
                    target_modules=CONFIG.get("lora_target_modules", ["q_proj", "v_proj"]),
                    lora_dropout=0.05,
                    bias="none",
                    task_type="CAUSAL_LM",
                )
                self.base = get_peft_model(self.base, lora_cfg)
                print("✅ LoRA adapters injected into text model. "
                      "cls_head / contrastive projs / connector stay full-rank trainable.")
                vision_model = self._get_vision_model()
                for p in vision_model.parameters():
                    p.requires_grad = False
        else:
            print("Using full fine-tune path with progressive vision unfreeze schedule (no LoRA).")
            self._apply_vision_unfreeze(0)

        connector = self._get_connector()
        for p in connector.parameters():
            p.requires_grad = True

        self._trainer_ref = None

        self.cls_head = nn.Sequential(
            nn.Linear(576, 512), nn.ReLU(), nn.Dropout(0.1), nn.Linear(512, 2)
        )

        hidden_dim = self.base.config.text_config.hidden_size
        self.contrastive_vision_proj = nn.Linear(576, CONFIG["contrastive_dim"], bias=False)
        self.contrastive_text_proj = nn.Linear(hidden_dim, CONFIG["contrastive_dim"], bias=False)

        self._vision_features = None
        self._connector_hook_handle = None

    def _get_base_model(self):
        m = self.base
        if hasattr(m, "base_model"):
            m = m.base_model
        if hasattr(m, "model"):
            m = m.model
        return m

    def _get_vision_model(self):
        base = self._get_base_model()
        if hasattr(base, "vision_model"):
            return base.vision_model
        if hasattr(base, "model") and hasattr(base.model, "vision_model"):
            return base.model.vision_model
        raise AttributeError(
            "Could not locate vision_model. Try setting 'use_lora': false in CONFIG, "
            "or inspect model structure with print(dir(model))."
        )

    def _get_connector(self):
        base = self._get_base_model()
        if hasattr(base, "connector"):
            return base.connector
        if hasattr(base, "model") and hasattr(base.model, "connector"):
            return base.model.connector
        if hasattr(base, "multi_modal_projector"):
            return base.multi_modal_projector
        raise AttributeError(
            "Could not locate connector (or multi_modal_projector). "
            "Update the _get_connector() helper if the transformers version changed the attribute name."
        )

    def _attach_vision_feature_hook(self):
        if self._connector_hook_handle is not None:
            self._connector_hook_handle.remove()
        connector = self._get_connector()

        def _hook(module, input, output):
            self._vision_features = output

        self._connector_hook_handle = connector.register_forward_hook(_hook)

    def _apply_vision_unfreeze(self, step):
        if getattr(self, "use_lora", False):
            return
        layers_to_unfreeze = []
        for sched_step, layers in self.vision_unfreeze_schedule.items():
            if step >= sched_step:
                layers_to_unfreeze.extend(layers)
        vision = self._get_vision_model()
        for name, param in vision.named_parameters():
            param.requires_grad = any(layer in name for layer in layers_to_unfreeze)
        self._reset_vision_optimizer_states(layers_to_unfreeze)

    def _reset_vision_optimizer_states(self, layers_just_considered):
        trainer = getattr(self, "_trainer_ref", None)
        if trainer is None or not hasattr(trainer, "optimizer") or trainer.optimizer is None:
            return
        opt = trainer.optimizer
        vision = self._get_vision_model()
        cleared = 0
        for name, param in vision.named_parameters():
            if not param.requires_grad:
                continue
            if param in opt.state:
                opt.state.pop(param, None)
                cleared += 1
        if cleared > 0:
            try:
                torch.cuda.empty_cache()
                gc.collect()
            except Exception:
                pass

    def on_step_end(self, step):
        if not getattr(self, "use_lora", False) and step in self.vision_unfreeze_schedule:
            self._apply_vision_unfreeze(step)
            print(f"🔓 Unfroze vision layers at step {step} (fresh Adam states for new layers)")

    def gradient_checkpointing_enable(self, **kwargs):
        self.base.gradient_checkpointing_enable(**kwargs)

    def forward(self, pixel_values, input_ids=None, attention_mask=None, labels=None, **kw):
        self._attach_vision_feature_hook()

        pv_for_base = pixel_values
        pam_for_base = kw.get("pixel_attention_mask")
        if pv_for_base is not None:
            if pv_for_base.dim() == 4:
                pv_for_base = pv_for_base.unsqueeze(1)
            elif pv_for_base.dim() == 5:
                B, P, C, H, W = pv_for_base.shape
                pv_for_base = pv_for_base.reshape(B * P, 1, C, H, W)
        if pam_for_base is not None:
            if pam_for_base.dim() == 3:
                pam_for_base = pam_for_base.unsqueeze(1)
            elif pam_for_base.dim() == 4:
                Bm, Pm, *sp = pam_for_base.shape
                pam_for_base = pam_for_base.reshape(Bm * Pm, 1, *sp)

        lm = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            pixel_values=pv_for_base,
            pixel_attention_mask=pam_for_base,
            output_hidden_states=True
        )

        if self._connector_hook_handle is not None:
            self._connector_hook_handle.remove()
            self._connector_hook_handle = None

        connector_out = self._vision_features
        self._vision_features = None

        if connector_out is None:
            with torch.no_grad():
                vision_model = self._get_vision_model()
                connector = self._get_connector()
                if pixel_values.dim() == 5:
                    B, P, C, H, W = pixel_values.shape
                    flat_pv = pixel_values.reshape(B * P, C, H, W)
                else:
                    flat_pv = pixel_values
                vf = vision_model(flat_pv)
                connector_out = connector(vf.last_hidden_state)

        batch_size = input_ids.shape[0] if input_ids is not None else (pixel_values.shape[0] if pixel_values is not None else 1)
        num_images = kw.get("num_images")

        if connector_out.dim() == 3:
            per_unit = connector_out.mean(dim=1)
        else:
            per_unit = connector_out
        total_units = per_unit.shape[0] if per_unit.dim() > 0 else 1

        if num_images is not None:
            stride = pixel_values.shape[1] if pixel_values.dim() == 5 else None
            image_embeds_list = []
            pos = 0
            for n in num_images.tolist():
                n = int(n)
                if n > 0 and pos < total_units:
                    if stride is not None:
                        sample_units = per_unit[pos: pos + n]
                        pos += stride
                    else:
                        sample_units = per_unit[pos: pos + n]
                        pos += n
                    emb = sample_units.mean(dim=0) if (sample_units.dim() > 1 and sample_units.shape[0] > 0) else sample_units
                    image_embeds_list.append(emb)
                else:
                    dim = per_unit.shape[-1] if per_unit.dim() > 0 else 576
                    image_embeds_list.append(torch.zeros(dim, device=per_unit.device, dtype=per_unit.dtype))
                    pos += stride if stride is not None else n
            image_embeds = torch.stack(image_embeds_list, dim=0)
        else:
            B = batch_size
            if total_units % B == 0 and total_units > 0:
                units_per_sample = total_units // B
                if per_unit.dim() == 1:
                    image_embeds = per_unit.reshape(B, units_per_sample).mean(dim=1)
                else:
                    image_embeds = per_unit.reshape(B, units_per_sample, -1).mean(dim=1)
            else:
                if per_unit.shape[0] == B and per_unit.dim() > 1:
                    image_embeds = per_unit.mean(dim=1) if per_unit.dim() == 2 else per_unit
                else:
                    if per_unit.dim() > 1:
                        image_embeds = per_unit.mean(dim=0, keepdim=True).expand(B, -1)
                    else:
                        image_embeds = per_unit.unsqueeze(0).expand(B, -1)

        torch.cuda.empty_cache()

        cls_logits = self.cls_head(image_embeds)

        if attention_mask is not None and lm.hidden_states is not None:
            seq_lens = attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(lm.hidden_states[-1].shape[0], device=lm.hidden_states[-1].device)
            text_embeds = lm.hidden_states[-1][batch_indices, seq_lens, :]
        else:
            text_embeds = lm.hidden_states[-1][:, -1, :]

        contr_vision = self.contrastive_vision_proj(image_embeds)
        contr_text = self.contrastive_text_proj(text_embeds.float())

        return {
            "loss": lm.loss,
            "cls_logits": cls_logits,
            "lm_loss": lm.loss,
            "lm_logits": getattr(lm, "logits", None),
            "image_embeds": image_embeds,
            "text_embeds": text_embeds,
            "contr_vision": contr_vision,
            "contr_text": contr_text,
        }


# ================== FRAMES DATASET (replaces video decoding) ==================
class VideoFakeClueDatasetFrames(Dataset):
    """
    Reads pre-extracted JPEG frames (written by precompute_frames_streaming.py)
    instead of decoding video. Picks a jittered num_frames-of-N subset from the
    saved pool each __getitem__ call, applies the same flip/brightness
    augmentation the original decord-based dataset used, and builds the exact
    same model inputs (pixel_values, labels, cls_label, etc.).
    """
    def __init__(self, data, proc, ml=2048, num_frames=None):
        self.data = data
        self.proc = proc
        self.ml = ml
        if num_frames is None:
            num_frames = CONFIG.get("num_frames", 4)
        self.num_frames = int(num_frames)
        self.augment = bool(CONFIG.get("frame_augment", False))

    def __len__(self):
        return len(self.data)

    def _pick_frame_subset(self, frame_paths):
        n_avail = len(frame_paths)
        if n_avail == 0:
            return []
        if n_avail <= self.num_frames:
            return list(frame_paths) + [frame_paths[-1]] * (self.num_frames - n_avail)
        base = [i * n_avail / self.num_frames for i in range(self.num_frames)]
        indices = []
        for b in base:
            jitter = random.choice([-1, 0, 1])
            idx = max(0, min(n_avail - 1, int(b) + jitter))
            indices.append(idx)
        return [frame_paths[i] for i in sorted(indices)]

    def _load_frames(self, frame_paths):
        chosen = self._pick_frame_subset(frame_paths)
        frames = []
        for p in chosen:
            try:
                frames.append(Image.open(p).convert("RGB"))
            except Exception:
                frames.append(Image.new("RGB", (384, 384), "gray"))

        if self.augment:
            if random.random() < 0.5:
                frames = [f.transpose(Image.FLIP_LEFT_RIGHT) for f in frames]
            if random.random() < 0.35:
                factor = random.uniform(0.82, 1.18)
                frames = [Image.eval(f, lambda x: max(0, min(255, int(x * factor)))) for f in frames]
        return frames

    def _find_assistant_start(self, input_ids):
        tok = self.proc.tokenizer
        markers = [
            "<|im_start|>assistant",
            "assistant\n",
            "<|assistant|>",
            "Assistant:",
            "\nassistant",
        ]
        for m in markers:
            try:
                m_ids = tok.encode(m, add_special_tokens=True)
                if not m_ids:
                    continue
                L = len(m_ids)
                for i in range(len(input_ids) - L + 1):
                    if input_ids[i: i + L].tolist() == m_ids:
                        return i + L
            except Exception:
                pass
        print("⚠️  _find_assistant_start: could not locate assistant header via tokens; using 70% heuristic")
        return int(len(input_ids) * 0.70)

    def __getitem__(self, idx):
        item = self.data[idx]
        frame_paths = item.get("frames") or []
        frames = self._load_frames(frame_paths) if frame_paths else [Image.new("RGB", (384, 384), "gray")] * self.num_frames

        msgs = []
        for c in item["conversations"]:
            role = "user" if c.get("from") == "human" else "assistant"
            text_val = c.get("value", "")
            if role == "user":
                clean_text = re.sub(r'\s*Clicks:\s*.*', '', text_val, flags=re.DOTALL).replace("<image>", "").strip()
                content = [{"type": "image"} for _ in frames]
                content.append({"type": "text", "text": clean_text})
                msgs.append({"role": role, "content": content})
            else:
                clean_val = re.sub(r'\s*Clicks:\s*.*', '', text_val, flags=re.DOTALL).strip()
                clean_val = re.sub(r',?\s*"\d+"\s*:\s*\{[^}]*\}', '', clean_val).strip()
                clean_val = re.sub(r'\s+', ' ', clean_val).strip()
                msgs.append({"role": role, "content": [{"type": "text", "text": clean_val}]})

        prompt = self.proc.apply_chat_template(msgs, add_generation_prompt=False, tokenize=False)
        inp = self.proc(text=prompt, images=frames, return_tensors="pt", truncation=False)
        pv = inp["pixel_values"]
        if pv.dim() == 5:
            pv = pv.squeeze(0)
        pam = inp.get("pixel_attention_mask")

        input_ids_0 = inp["input_ids"][0]
        as_st = self._find_assistant_start(input_ids_0)
        tok = self.proc.tokenizer
        lbls = input_ids_0.clone()
        lbls[:as_st] = -100
        lbls[lbls == tok.pad_token_id] = -100

        return {
            "pixel_values": pv,
            "pixel_attention_mask": pam,
            "input_ids": inp["input_ids"].squeeze(0),
            "attention_mask": inp["attention_mask"].squeeze(0),
            "labels": lbls,
            "cls_label": 1 if item["label"] == "fake" else 0,
        }


def resolution_check(data, label=""):
    """Replaces the old _PathOnlyDataset video-resolution check: verifies each
    entry's frame files actually exist on disk."""
    good = 0
    for item in data:
        frame_paths = item.get("frames") or []
        if frame_paths and all(Path(p).exists() for p in frame_paths):
            good += 1
    total = len(data)
    pct = 100 * good / total if total else 0.0
    print(f"   {label} frame resolution check: {good}/{total} ({pct:.1f}%) entries have all frame files present.")
    if good < total * 0.5:
        print("⚠️  Many entries are missing frame files — check that CONFIG['data_root'] points at the "
              "--out_dir you passed to precompute_frames_streaming.py.")
    return good, total


# ================== COLLATOR, TRAINER, CALLBACKS (unchanged) ==================
class CustomDataCollator:
    def __init__(self, tok):
        self.tok = tok

    def __call__(self, feats):
        pad = self.tok.pad(
            {"input_ids": [f["input_ids"] for f in feats],
             "attention_mask": [f["attention_mask"] for f in feats]},
            padding=True, return_tensors="pt"
        )
        batch = {"input_ids": pad["input_ids"], "attention_mask": pad["attention_mask"]}
        max_len = max(f["labels"].shape[0] for f in feats)
        labels = torch.full((len(feats), max_len), -100, dtype=torch.long)
        for i, f in enumerate(feats):
            labels[i, :f["labels"].shape[0]] = f["labels"]
        batch["labels"] = labels

        pvs = [f.get("pixel_values") for f in feats]
        p_shapes = [pv.shape for pv in pvs]
        if len(set(s[0] for s in p_shapes)) == 1:
            pv_batch = torch.stack(pvs, dim=0)
        else:
            max_p = max(pv.shape[0] for pv in pvs)
            C, H, W = pvs[0].shape[1], pvs[0].shape[2], pvs[0].shape[3]
            pv_batch = torch.zeros((len(feats), max_p, C, H, W), dtype=pvs[0].dtype)
            for i, pv in enumerate(pvs):
                pv_batch[i, :pv.shape[0]] = pv
        batch["pixel_values"] = pv_batch
        num_images_list = [pv.shape[0] if pv.dim() == 4 else 1 for pv in pvs]
        batch["num_images"] = torch.tensor(num_images_list, dtype=torch.long)

        batch["cls_label"] = torch.tensor([f["cls_label"] for f in feats])
        return batch


class MultiTaskTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.contr_warmup_steps = 500
        self._last_logged_step = -1

    def log(self, logs, start_time=None, **kwargs):
        if "loss" in logs:
            acc = max(1, self.args.gradient_accumulation_steps)
            logs = {**logs, "loss": round(float(logs["loss"]) / acc, 4)}
        super().log(logs, start_time, **kwargs)

    def create_optimizer(self):
        all_params = list(self.model.named_parameters())
        cls_head_params, vision_params, contr_head_params, base_params = [], [], [], []
        for name, param in all_params:
            if "cls_head" in name:
                cls_head_params.append(param)
            elif "contrastive" in name:
                contr_head_params.append(param)
            elif "vision_model" in name:
                vision_params.append(param)
            else:
                base_params.append(param)

        n_frozen = sum(1 for p in vision_params if not p.requires_grad)
        n_active = len(vision_params) - n_frozen
        lora_note = " (LoRA)" if getattr(self.model, "use_lora", False) else ""
        print(f"💡 Vision: {n_active} active, {n_frozen} frozen{lora_note}")

        lr = self.args.learning_rate
        optimizer_grouped_parameters = [
            {"params": [p for p in base_params if p.requires_grad], "lr": lr},
            {"params": [p for p in cls_head_params if p.requires_grad], "lr": lr * CONFIG["head_lr_multiplier"]},
            {"params": contr_head_params, "lr": lr * CONFIG["head_lr_multiplier"]},
            {"params": vision_params, "lr": lr * CONFIG["vision_lr_multiplier"]},
        ]
        self.optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=self.args.weight_decay,
        )
        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        out = model(**{k: v for k, v in inputs.items() if k != "cls_label"})

        cls_labels = inputs["cls_label"]
        if FIXED_CLS_WEIGHTS is not None:
            cls_weight = FIXED_CLS_WEIGHTS.to(cls_labels.device)
        else:
            n_real = (cls_labels == 0).sum().float()
            n_fake = (cls_labels == 1).sum().float()
            cls_weight = torch.ones(2, device=cls_labels.device)
            if n_real > 0 and n_fake > 0:
                cls_weight[0] = n_fake / (n_real + n_fake) * 2
                cls_weight[1] = n_real / (n_real + n_fake) * 2
            else:
                cls_weight = cls_weight * 0.5
        cls_loss = F.cross_entropy(out["cls_logits"], cls_labels, weight=cls_weight, label_smoothing=0.05)

        lm_loss = out["lm_loss"]

        if inputs.get("labels") is not None:
            labels_flat = inputs["labels"].view(-1)
            valid_mask = labels_flat != -100
            if valid_mask.any():
                valid_labels = labels_flat[valid_mask]
                uniq_count = len(valid_labels.unique())
                total_count = valid_labels.shape[0]
                if total_count > 3 and uniq_count / total_count < 0.6:
                    rep_penalty = 0.5 * (1.0 - uniq_count / total_count)
                    lm_loss = lm_loss * (1.0 + rep_penalty)

        ie = F.normalize(out["contr_vision"], dim=-1)
        te = F.normalize(out["contr_text"], dim=-1)
        logits = (ie @ te.T) / CONFIG["temperature"]
        labs = torch.arange(logits.shape[0], device=logits.device)
        contr_loss = (F.cross_entropy(logits + 1e-8, labs) + F.cross_entropy(logits.T + 1e-8, labs)) / 2

        ramp = CONFIG.get("contrastive_ramp_steps", 300)
        boost_step = CONFIG.get("boost_after_step", 500)
        boost_contr = CONFIG.get("boost_contr_factor", 1.8)
        boost_cls = CONFIG.get("boost_cls_factor", 1.3)

        contr_weight = CONFIG["contrastive_loss_weight"] * min(1.0, self.state.global_step / max(ramp, 1))
        if self.state.global_step > boost_step:
            contr_weight *= boost_contr
            cls_loss = cls_loss * boost_cls

        decay_start = CONFIG.get("late_aux_decay_start")
        if decay_start is not None and self.state.global_step > decay_start:
            contr_weight = 0.0
            target_cls_w = CONFIG.get("late_cls_weight")
            if target_cls_w is not None:
                if not getattr(self, "_late_phase_logged", False):
                    print(f"[late phase @ {self.state.global_step}] aux down-weighted for LM focus.")
                    self._late_phase_logged = True
                base_cls_w = CONFIG.get("cls_loss_weight", 1.0)
                if base_cls_w > 1e-6:
                    cls_loss = cls_loss * (target_cls_w / base_cls_w)
            elif not getattr(self, "_late_phase_logged", False):
                print(f"[late phase @ {self.state.global_step}] contrastive disabled; CLS weight kept unchanged.")
                self._late_phase_logged = True

        total_loss = (
            CONFIG["cls_loss_weight"] * cls_loss +
            CONFIG["lm_loss_weight"] * lm_loss +
            contr_weight * contr_loss
        )

        with torch.no_grad():
            cls_preds = out["cls_logits"].argmax(dim=-1)
            cls_acc = (cls_preds == inputs["cls_label"]).float().mean().item()

        log_every = CONFIG.get("log_every", CONFIG.get("global_step", 20))
        if (self.state.global_step > self._last_logged_step and
                self.state.global_step % max(log_every, 1) == 0):
            print(f"[step {self.state.global_step:4d}] "
                  f"Total:{total_loss.item():.3f} | CLS:{cls_loss.item():.3f} | "
                  f"LM:{lm_loss.item():.3f} | Contr:{contr_loss.item():.3f} | "
                  f"CLSAcc:{cls_acc:.3f} | cw:{contr_weight:.2f}")
            self._last_logged_step = self.state.global_step

        return (total_loss, out) if return_outputs else total_loss


class UnfreezeCallback(TrainerCallback):
    def on_step_end(self, args, state, control, model=None, **kwargs):
        if hasattr(model, 'on_step_end'):
            model.on_step_end(state.global_step)
        return control


class GenerationMonitorCallback(TrainerCallback):
    """Sanity-check generation every N steps. Reads frames directly from the raw
    entry's 'frames' list — no video decoding needed anymore."""
    def __init__(self, val_dataset=None, train_dataset=None):
        self.val_dataset = val_dataset
        self.train_dataset = train_dataset
        self._cached_examples = None

    def _get_examples(self):
        if self._cached_examples is not None:
            return self._cached_examples
        exs = []
        ds = self.val_dataset if self.val_dataset is not None else self.train_dataset
        if ds is not None and len(ds) > 0:
            for idx in [0, max(1, len(ds) // 3)]:
                if idx < len(ds):
                    try:
                        raw = ds.data[idx] if hasattr(ds, "data") else None
                        if raw is not None:
                            exs.append(raw)
                    except Exception:
                        pass
        self._cached_examples = exs or []
        return self._cached_examples

    def on_step_end(self, args, state, control, model=None, **kwargs):
        try:
            interval = int(CONFIG.get("generation_sample_interval", 0))
            if interval <= 0 or state.global_step % interval != 0 or model is None:
                return control

            exs = self._get_examples()
            if not exs:
                return control

            base = getattr(model, "base", model)
            proc = getattr(model, "processor", None)
            if proc is None:
                return control

            max_new = int(CONFIG.get("generation_max_new_tokens", 48))
            printed = 0
            was_training = base.training
            base.eval()

            for raw in exs[:2]:
                try:
                    frame_paths = (raw.get("frames") or [])[: CONFIG.get("num_frames", 4)]
                    frames = [Image.open(p).convert("RGB") for p in frame_paths] if frame_paths else None
                    if not frames:
                        frames = [Image.new("RGB", (384, 384), "gray")]

                    human_text = ""
                    for c in raw.get("conversations", []):
                        if c.get("from") == "human":
                            human_text = re.sub(r'\s*Clicks:\s*.*', '', c.get("value", ""), flags=re.DOTALL).replace("<image>", "").strip()
                            break
                    if not human_text:
                        human_text = "Is this video real or fake? Describe artifacts if any."

                    content = [{"type": "image"} for _ in frames]
                    content.append({"type": "text", "text": human_text})
                    msgs = [{"role": "user", "content": content}]

                    prompt = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
                    inputs = proc(text=prompt, images=frames, return_tensors="pt", truncation=False)
                    dev = next(base.parameters()).device
                    inputs = {k: v.to(dev) if hasattr(v, "to") else v for k, v in inputs.items()}

                    temperature = float(CONFIG.get("generation_sample_temperature", 0.7))
                    gen_kwargs = {"max_new_tokens": max_new, "use_cache": True}
                    if temperature > 0.0:
                        gen_kwargs["do_sample"] = True
                        gen_kwargs["temperature"] = temperature
                    else:
                        gen_kwargs["do_sample"] = False

                    with torch.no_grad():
                        gen_ids = base.generate(**inputs, **gen_kwargs)

                    prompt_len = inputs["input_ids"].shape[1]
                    new_tokens = gen_ids[0, prompt_len:]
                    pred = proc.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

                    gt = ""
                    for c in raw.get("conversations", []):
                        if c.get("from") in ("gpt", "assistant"):
                            gt = c.get("value", "").strip()
                            break

                    print(f"[gen @{state.global_step}] GT: {gt[:110]}{'...' if len(gt) > 110 else ''} | Pred: {pred[:110]}{'...' if len(pred) > 110 else ''}")
                    printed += 1
                    if printed >= 2:
                        break
                except Exception:
                    continue

            if was_training:
                base.train()
            else:
                base.eval()
            torch.cuda.empty_cache()
        except Exception:
            pass
        return control


class ValidationClsAccuracyCallback(TrainerCallback):
    def __init__(self, val_dataset, collator, output_dir, max_eval_samples=600, batch_size=4):
        self.val_dataset = val_dataset
        self.collator = collator
        self.output_dir = Path(output_dir)
        self.max_eval_samples = max_eval_samples
        self.batch_size = batch_size
        self.best_acc = -1.0

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if self.val_dataset is None or model is None:
            return control
        n_total = len(self.val_dataset)
        n = min(n_total, self.max_eval_samples)
        stride = max(1, n_total // n)
        idxs = list(range(0, n_total, stride))[:n]

        was_training = model.training
        model.eval()
        correct, total = 0, 0
        try:
            with torch.no_grad():
                for i in range(0, len(idxs), self.batch_size):
                    chunk = [self.val_dataset[j] for j in idxs[i:i + self.batch_size]]
                    batch = self.collator(chunk)
                    dev = next(model.parameters()).device
                    batch = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in batch.items()}
                    out = model(**{k: v for k, v in batch.items() if k != "cls_label"})
                    preds = out["cls_logits"].argmax(dim=-1).cpu()
                    correct += (preds == batch["cls_label"].cpu()).sum().item()
                    total += len(preds)
        except Exception as e:
            print(f"⚠️  ValidationClsAccuracyCallback failed this epoch ({e}); skipping.")
            if was_training:
                model.train()
            return control

        acc = correct / max(1, total)
        print(f"[val-cls-acc @ epoch {state.epoch:.0f}] {acc:.4f} on {total} samples")
        if acc > self.best_acc:
            self.best_acc = acc
            save_path = self.output_dir / "best_cls_acc"
            save_path.mkdir(parents=True, exist_ok=True)
            print(f"   ✅ New best val CLS accuracy ({acc:.4f}) — saving to {save_path}")
            base = getattr(model, "base", model)
            base.save_pretrained(str(save_path))
            torch.save({"cls_head": model.cls_head.state_dict()}, save_path / "cls_head.pt")

        if was_training:
            model.train()
        return control


# ================== MAIN TRAINING ==================
if __name__ == "__main__":
    data_root = Path(CONFIG["data_root"])

    jp = data_root / CONFIG["train_json"]
    if not jp.exists():
        raise FileNotFoundError(
            f"{jp} not found. Run precompute_frames_streaming.py first with --out_dir {CONFIG['data_root']}."
        )
    with open(jp) as f:
        raw_data = json.load(f)
    print(f"📄 Using train annotations: {jp.name} ({len(raw_data):,} entries)")

    if CONFIG.get("subset_ratio", 1.0) < 1.0:
        raw_data = raw_data[: int(len(raw_data) * CONFIG["subset_ratio"])]

    val_data = None
    vp = data_root / CONFIG.get("val_json", "val.json")
    if vp.exists():
        with open(vp) as f:
            val_data = json.load(f)
        print(f"✅ Loaded real val split: {vp.name} ({len(val_data):,} entries)")
    else:
        val_ratio = CONFIG.get("val_ratio", 0.0)
        if val_ratio > 0 and len(raw_data) > 20:
            random.seed(42)
            random.shuffle(raw_data)
            n_val = max(1, int(len(raw_data) * val_ratio))
            val_data = raw_data[:n_val]
            raw_data = raw_data[n_val:]
            print(f"⚠️  {vp.name} not found — falling back to carved split → "
                  f"train: {len(raw_data):,} | val: {len(val_data):,} (seed=42)")

    reals = sum(1 for x in raw_data if str(x.get("label", "")).lower() == "real")
    fakes = len(raw_data) - reals
    print(f"✅ Loaded {len(raw_data):,} videos — Real: {reals} | Fake: {fakes}")
    if reals == 0 or fakes == 0:
        print("⚠️  WARNING: One class is completely missing (or extremely rare). Classification head cannot learn a boundary.")
    elif reals > 0 and fakes > 0:
        total_n = reals + fakes
        w_real = total_n / (2.0 * reals)
        w_fake = total_n / (2.0 * fakes)
        FIXED_CLS_WEIGHTS = torch.tensor([w_real, w_fake], dtype=torch.float32)
        print(f"⚖️  Fixed balanced CLS weights from train counts — real: {w_real:.3f}, fake: {w_fake:.3f}")

    resolution_check(raw_data, label="train")
    if val_data:
        resolution_check(val_data, label="val")

    model = MultiTaskSmolVLM(
        CONFIG["model_id"],
        CONFIG["head_lr_multiplier"],
        CONFIG["vision_lr_multiplier"],
    )

    train_dataset = VideoFakeClueDatasetFrames(
        raw_data, model.processor, ml=CONFIG["max_seq_length"], num_frames=CONFIG["num_frames"]
    )
    val_dataset = (
        VideoFakeClueDatasetFrames(val_data, model.processor, num_frames=CONFIG["num_frames"])
        if val_data else None
    )

    args = TrainingArguments(
        output_dir=CONFIG["output_dir"],
        num_train_epochs=CONFIG["num_train_epochs"],
        per_device_train_batch_size=CONFIG["per_device_train_batch_size"],
        gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
        learning_rate=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
        warmup_ratio=CONFIG["warmup_ratio"],
        max_grad_norm=CONFIG["max_grad_norm"],
        lr_scheduler_type=CONFIG.get("lr_scheduler_type", "cosine"),
        bf16=CONFIG["bf16"],
        gradient_checkpointing=CONFIG["gradient_checkpointing"],
        use_cache=False,
        dataloader_num_workers=CONFIG["dataloader_num_workers"],
        dataloader_persistent_workers=(CONFIG["dataloader_num_workers"] > 0),
        dataloader_pin_memory=False,
        dataloader_prefetch_factor=1,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="no",
        remove_unused_columns=False,
        report_to="none",
        disable_tqdm=False,
    )

    torch.cuda.empty_cache()
    gc.collect()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print("✅ TF32 + empty_cache + persistent dataloader enabled")

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("✅ Gradient checkpointing enabled on model")

    torch.cuda.empty_cache()
    gc.collect()
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f"💾 GPU memory at trainer start: {free/1024**3:.2f} GiB free / {total/1024**3:.2f} GiB total")

    trainer = MultiTaskTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=CustomDataCollator(model.processor.tokenizer),
    )
    model._trainer_ref = trainer

    trainer.add_callback(UnfreezeCallback())
    trainer.add_callback(GenerationMonitorCallback(val_dataset=val_dataset, train_dataset=train_dataset))
    if val_dataset is not None:
        trainer.add_callback(ValidationClsAccuracyCallback(
            val_dataset=val_dataset,
            collator=CustomDataCollator(model.processor.tokenizer),
            output_dir=CONFIG["output_dir"],
            max_eval_samples=CONFIG.get("val_cls_acc_max_samples", 600),
        ))

    mode = "full fine-tune (progressive unfreeze, no LoRA)" if not getattr(model, "use_lora", False) else "LoRA"
    print(f"\n✅ Starting multi-task video fine-tuning ({mode}) — real/fake + explanations [FRAMES MODE]")

    torch.cuda.empty_cache()
    gc.collect()

    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        is_full_finetune = not CONFIG.get("use_lora", True)
        min_required_gb = 3.0 if is_full_finetune else 1.5
        if free < min_required_gb * 1024**3:
            mode_str = "full fine-tune (no LoRA) + effective batch ~8" if is_full_finetune else "current settings"
            print(f"\n❌ FATAL: Only {free/1024**3:.2f} GiB free on GPU. You need at least ~{min_required_gb} GiB free for this process ({mode_str}).")
            raise RuntimeError("Insufficient free GPU memory due to other processes.")

    trainer.train()

    trainer.save_model(CONFIG["output_dir"])
    model.processor.save_pretrained(CONFIG["output_dir"])

    extra = {
        "cls_head": model.cls_head.state_dict(),
        "contrastive_vision_proj": model.contrastive_vision_proj.state_dict(),
        "contrastive_text_proj": model.contrastive_text_proj.state_dict(),
    }
    torch.save(extra, Path(CONFIG["output_dir"]) / "multitask_heads.pt")

    print(f"\n🎉 Training finished! Model + processor + multitask_heads.pt saved to {CONFIG['output_dir']}")

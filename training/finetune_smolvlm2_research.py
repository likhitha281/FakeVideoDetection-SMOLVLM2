#!/usr/bin/env python3
"""
finetune_lora.py — SmolVLM2-256M Video Instruct Fine-Tuning
Real/fake video classification + natural-language artifact explanations.
Data: LikhithaGeddam/fakevideodetection-private (Hub dataset), driven by a local
captions file shaped like your sample:
    {"filename": "real/train_real_001.mp4", "dataset_label": "real", "caption": "..."}

================================================================================
WHAT THIS VERSION DOES DIFFERENTLY (fresh pass over your uploaded script)
================================================================================
1. NEW DATA PATH (the actual point of this rewrite)
   Your captions file has no "conversations"/"image" fields — VideoFakeClueDataset
   now detects that schema and:
     - builds a human/assistant turn pair on the fly from dataset_label + caption
       (REAL_QUESTION/FAKE_QUESTION templates, near the top of this file)
     - resolves each video by streaming it from the Hub repo on demand via
       huggingface_hub.hf_hub_download(repo_id=..., filename=f"{split}/{filename}"),
       which caches under ~/.cache/huggingface/hub — each video downloads at most
       once across the whole run (and future runs, until that cache is cleared).
   The old local-disk path finder (basename index + a dozen hardcoded FaceForensics++
   folder-layout guesses, from when videos were expected pre-downloaded to disk) has
   been removed entirely — this pipeline only ever uses the captions schema now.

2. BUG FIX: the schedule constants were dead weight
   contrastive_ramp_steps / boost_after_step / late_aux_decay_start were hardcoded
   to numbers derived from one specific old dataset (7004 samples, 2 epochs) — with
   a different dataset size these numbers no longer land where the comments said
   they would (e.g. "decay starts at 60%" was only true for that one run). main()
   now computes them as fractions of the ACTUAL total step count for whatever data
   you're training on, so the schedule scales correctly regardless of dataset size.

3. BUG FIX: the val split was built but never evaluated
   val_dataset was carved off via val_ratio, passed into eval_dataset=..., but
   eval_strategy="no" meant it did nothing — no accuracy number was ever produced.
   Added ValidationClsAccuracyCallback: runs a real classification-accuracy pass
   over val_dataset at the end of every epoch, prints it, and separately checkpoints
   the best-scoring model to best_cls_acc/ — this is the actual signal you use to
   pick a checkpoint, instead of only the qualitative GT-vs-Pred generation samples.

4. CONFIG resolved to ONE coherent recommendation instead of three conflicting ones
   The original file had a "RECOMMENDED SETTINGS" block in the docstring (full
   fine-tune, no LoRA, batch 8, lr 5e-6), a different actual CONFIG dict at the top
   (LoRA, batch 8, lr 5e-5), and a THIRD "Recommended CONFIG" block near the bottom
   — all mutually inconsistent, left over from iterative experimentation. This
   version keeps a single LoRA-based CONFIG (fast, memory-light, appropriate for a
   256M model) with values chosen for a stable, efficient run — see inline comments
   on each changed value below.

Everything else — the gradient-flow fix (hook-based connector feature capture
instead of a second no-grad vision pass), the LoRA/progressive-unfreeze paths,
token-id-based assistant-header label masking, class-weighted classification loss,
and the qualitative generation monitor — is unchanged in spirit from your script.
================================================================================
IMPORTANT FOR SHARED GPUs:
The environment variables below are set at the very top of this file (before torch
is imported) to reduce memory fragmentation.

Run with: python finetune_lora.py
"""
import os, json, warnings, re

# These MUST be set before torch does any CUDA allocation.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
# Video cache on LOCAL disk, not Drive. Testing showed Drive-mounted (FUSE) I/O for the
# video cache roughly doubled per-step time (~36s/it vs ~19-23s/it on local disk) — likely
# because Drive's FUSE mount is slow for lots of small/medium file reads/writes. With 70GB
# free, Drive space isn't the constraint anymore, so speed wins: local cache is faster,
# and checkpoints (on Drive, below) are what actually needs to survive a disconnect anyway.
# Trade-off: a disconnect now costs a video re-download, not a training restart — worth it
# compared to a ~2x slowdown across the entire remaining run.
os.environ.setdefault("HF_HOME", "/content/hf_cache")
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
import random
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor, TrainingArguments, Trainer, TrainerCallback
from transformers.utils import logging
import gc
import decord  # pip install decord

# Optional peft for the recommended LoRA path (stable updates, avoids unfreeze cliff)
try:
    from peft import LoraConfig, get_peft_model
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False

warnings.filterwarnings("ignore")
logging.set_verbosity_error()

# ================== SINGLE-GPU OPTIMIZATIONS ==================
DEVICE = "cuda"
MODEL_DTYPE = torch.bfloat16
print(f"🚀 PyTorch: {torch.__version__} | Device: {DEVICE} | Dtype: {MODEL_DTYPE}")

# ================== CONFIG (single coherent recommendation — see docstring item 4) ==================
CONFIG = {
    "model_id": "HuggingFaceTB/SmolVLM2-256M-Instruct",
    "output_dir": "/scratch/lgeddam1/gemma_captioning/checkpoints/smolvlm2_lora",
    # CHANGED: pointed at Drive instead of ./checkpoints (which lives on Colab's ephemeral
    # disk) — a runtime disconnect/eviction no longer loses your checkpoints.

    # Resume support: set this to a checkpoint-N folder to continue training from there
    # instead of step 0. None = start fresh. Must live under output_dir above (or wherever
    # your checkpoints actually are) — this is exactly what you asked to resume from.
    "resume_from_checkpoint": None,

    "data_root": "/scratch/lgeddam1/gemma_captioning/checkpoints",
    "train_json": "caption_progress_gemma4_12b_video.jsonl",         # your filename/dataset_label/caption file

    # --- NEW: HF-hosted video source ---
    "hf_repo_id": "LikhithaGeddam/fakevideodetection-private",
    "hf_split": "train",
    "hf_token": None,   # None = auto-detect from HF_TOKEN env var or `huggingface_hub.login()` cache.
    # ⚠️ SECURITY: this is a live token in plaintext. Recommend rotating it at
    # https://huggingface.co/settings/tokens and using `login()` in a Colab cell instead.

    # --- NEW: local dataset on Drive (avoids re-downloading from the Hub every run) ---
    # Flat structure, confirmed: <local_dataset_root>/real/*.mp4, <local_dataset_root>/fake/*.mp4
    # — no train/ split subfolder. filename already includes the real/fake prefix.
    "use_local_dataset": True,
    "local_dataset_root": "/scratch/lgeddam1/gemma_captioning/dataset/train",

    # --- Core training ---
    "bf16": True,
    "num_train_epochs": 3,                 # LoRA converges fast; 3 epochs avoids the cosine-tail waste noted in the original docstring
    "learning_rate": 5e-5,                 # LoRA standard: 10-50x higher than full fine-tune
    "weight_decay": 0.02,
    "warmup_ratio": 0.08,
    "max_grad_norm": 1.0,
    "lr_scheduler_type": "cosine",
    "gradient_checkpointing": False,
    # CHANGED: was True. Your last run showed 39.08/39.49 GiB free — a 256M model with
    # LoRA doesn't need the memory savings, and gradient checkpointing trades real speed
    # (recomputing forward activations during backward) for memory you're not using.
    # This is very likely your single biggest lever against the ~18-20s/it you were seeing.

    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,      # effective batch = 16 — smooths per-step class-imbalance noise
    "dataloader_num_workers": 2,

    "max_seq_length": 4096,

    "subset_ratio": 1.0,

    # --- Multi-task loss weights ---
    "cls_loss_weight": 1.3,                # classification is the primary deliverable; prioritized over LM
    "lm_loss_weight": 0.9,
    "contrastive_loss_weight": 0.1,        # small but nonzero — effective batch 16 gives enough in-batch negatives to be useful
    "temperature": 0.07,
    "contrastive_dim": 256,

    "head_lr_multiplier": 2.5,             # cls_head is randomly initialized, needs to catch up faster
    "vision_lr_multiplier": 0.2,

    # --- LoRA ---
    "use_lora": True,
    "lora_r": 32,
    "lora_alpha": 64,
    "lora_target_modules": [               # attention + MLP: adapts both what's attended to and how it's transformed
        "q_proj", "v_proj", "k_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],

    # --- Progressive unfreeze (no-op when use_lora=True; kept for the legacy no-LoRA path) ---
    "vision_unfreeze_schedule": {
        0: [],
        40: ["layers.12.", "post_layernorm"],
        100: ["layers.10.", "layers.11.", "layers.12.", "post_layernorm"],
        180: ["layers.8.", "layers.9.", "layers.10.", "layers.11.", "layers.12.", "post_layernorm"],
        260: ["layers.6.", "layers.7.", "layers.8.", "layers.9.", "layers.10.", "layers.11.", "layers.12.", "post_layernorm"],
    },

    # --- Video input ---
    "num_frames": 4,
    "frame_augment": True,

    # --- Pre-computed vision: not used by this data pipeline (kept as unused model-level
    # plumbing in forward()/collator only — safe to ignore; not wired to any CONFIG flag here) ---

    # --- Monitoring / logging / val ---
    "log_every": 20,
    "global_step": 20,
    "val_ratio": 0.0,
    "val_dataset_root": "/scratch/lgeddam1/gemma_captioning/dataset/val",

    # --- Loss schedule controls ---
    # NOTE: these four are placeholders — main() OVERWRITES them from the actual total
    # step count once the dataset size is known (see docstring item 2 / "BUG FIX").
    "contrastive_ramp_steps": None,
    "boost_after_step": None,
    "boost_contr_factor": 1.0,
    "boost_cls_factor": 1.1,
    "late_aux_decay_start": None,
    "late_cls_weight": 0.15,

    # --- Lightweight generation sampling (also overwritten dynamically in main()) ---
    "generation_sample_interval": None,
    "generation_max_new_tokens": 48,
    "generation_sample_temperature": 0.7,
}


# ============================================================
# SLURM RESEARCH OVERRIDES
# ============================================================
# One canonical training implementation is used for all
# frame-count experiments. Slurm controls only the experimental
# variables, preventing code drift between 4/6/8/12-frame runs.

EXPERIMENT_NUM_FRAMES = int(
    os.environ.get(
        "SMOL_NUM_FRAMES",
        str(CONFIG.get("num_frames", 4))
    )
)

if EXPERIMENT_NUM_FRAMES not in {4, 6, 8, 12}:
    raise ValueError(
        f"SMOL_NUM_FRAMES must be one of 4, 6, 8, 12; "
        f"got {EXPERIMENT_NUM_FRAMES}"
    )

CONFIG["num_frames"] = EXPERIMENT_NUM_FRAMES

EXPERIMENT_BATCH_SIZE = int(
    os.environ.get(
        "SMOL_BATCH_SIZE",
        str(CONFIG["per_device_train_batch_size"])
    )
)

EXPERIMENT_GRAD_ACCUM = int(
    os.environ.get(
        "SMOL_GRAD_ACCUM",
        str(CONFIG["gradient_accumulation_steps"])
    )
)

CONFIG["per_device_train_batch_size"] = EXPERIMENT_BATCH_SIZE
CONFIG["gradient_accumulation_steps"] = EXPERIMENT_GRAD_ACCUM

effective_batch = (
    CONFIG["per_device_train_batch_size"]
    * CONFIG["gradient_accumulation_steps"]
)

if effective_batch != 16:
    raise ValueError(
        f"Effective batch must remain 16 for controlled experiments; "
        f"got {CONFIG['per_device_train_batch_size']} x "
        f"{CONFIG['gradient_accumulation_steps']} = {effective_batch}"
    )

CONFIG["output_dir"] = os.environ.get(
    "SMOL_OUTPUT_DIR",
    f"/scratch/lgeddam1/gemma_captioning/checkpoints/"
    f"smolvlm2_multitask_{EXPERIMENT_NUM_FRAMES}frames"
)

print("=" * 70)
print("EXPERIMENT CONFIGURATION")
print("=" * 70)
print("Frames/video :", CONFIG["num_frames"])
print("Micro-batch  :", CONFIG["per_device_train_batch_size"])
print("Grad accum   :", CONFIG["gradient_accumulation_steps"])
print("Effective BS :", CONFIG["per_device_train_batch_size"] * CONFIG["gradient_accumulation_steps"])
print("Output dir   :", CONFIG["output_dir"])
print("=" * 70)

# ================== QUESTION TEMPLATES FOR THE NEW CAPTIONS-BASED DATA FORMAT ==================
# Used by VideoFakeClueDataset._build_conversation_from_caption() to build a human/assistant
# turn pair on the fly from your captions.json (filename/dataset_label/caption), since that
# file doesn't itself contain a "conversations" field.
NEUTRAL_QUESTION = (
    "Watch these video frames carefully. Is this video real or a deepfake? "
    "Determine whether the video is REAL or FAKE and describe the specific visual "
    "evidence that supports your decision."
)

REAL_QUESTION = NEUTRAL_QUESTION
FAKE_QUESTION = NEUTRAL_QUESTION


# Computed once in main() from the FULL train set's real/fake counts, then used by
# MultiTaskTrainer.compute_loss() instead of the old per-batch reweighting. With this
# dataset's severe imbalance (~13% real), a per-batch computation is unreliable: most
# batches (size 4) contain zero real examples, which silently fell back to uniform
# weighting instead of correcting for the imbalance. A dataset-level fixed weight
# applies real correction on every single step, regardless of what's in that batch.
FIXED_CLS_WEIGHTS = None

# ================== MULTI-TASK MODEL (gradient flow fixed + optional LoRA) ==================
class MultiTaskSmolVLM(nn.Module):
    """
    Multi-task wrapper:
      - base SmolVLM2 for LM explanations (with pixel_values)
      - cls_head on (now *gradient-enabled*) pooled connector features for real/fake
      - lightweight contrastive (vision pooled vs. last assistant token hidden)
    """
    def __init__(self, model_id, head_lr_multiplier=5.0, vision_lr_multiplier=0.2):
        super().__init__()
        self.base = AutoModelForImageTextToText.from_pretrained(
            model_id, torch_dtype=MODEL_DTYPE, attn_implementation="sdpa"
        )
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.head_lr_multiplier = head_lr_multiplier
        self.vision_lr_multiplier = vision_lr_multiplier

        # Pull schedule from CONFIG (supports the moved definition)
        self.vision_unfreeze_schedule = CONFIG.get(
            "vision_unfreeze_schedule",
            {0: [], 80: ["layers.12.", "post_layernorm"]}  # minimal fallback
        )

        # === LoRA path (recommended) or classic unfreeze ===
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
                # Vision stays mostly frozen when using LoRA
                vision_model = self._get_vision_model()
                for p in vision_model.parameters():
                    p.requires_grad = False
        else:
            # Classic progressive unfreeze path (ACTIVE for full fine-tuning without LoRA)
            print("Using full fine-tune path with progressive vision unfreeze schedule (no LoRA).")
            self._apply_vision_unfreeze(0)

        # Always train the connector + our small heads (critical for multi-task)
        connector = self._get_connector()
        for p in connector.parameters():
            p.requires_grad = True

        # Will be set from the outside (main) after the Trainer is constructed so that
        # _reset_vision_optimizer_states can reach trainer.optimizer.state for fresh-Adam resets on unfreeze.
        self._trainer_ref = None

        # 576 is correct for SmolVLM2-256M connector output (text hidden_size)
        self.cls_head = nn.Sequential(
            nn.Linear(576, 512), nn.ReLU(), nn.Dropout(0.1), nn.Linear(512, 2)
        )

        hidden_dim = self.base.config.text_config.hidden_size
        self.contrastive_vision_proj = nn.Linear(576, CONFIG["contrastive_dim"], bias=False)
        self.contrastive_text_proj = nn.Linear(hidden_dim, CONFIG["contrastive_dim"], bias=False)

        # For efficient feature capture (we attach a hook during forward to grab connector output
        # from the *main* lm call instead of doing an expensive second vision forward)
        self._vision_features = None
        self._connector_hook_handle = None

    # ---------- Robust access helpers for SmolVLM2 (handles raw model + PeftModel/LoRA) ----------
    def _get_base_model(self):
        """Unwrap PeftModel (and nested .model) until we reach the SmolVLMModel that owns vision_model/connector."""
        m = self.base
        # Unwrap PEFT wrapper
        if hasattr(m, "base_model"):
            m = m.base_model
        # SmolVLMForConditionalGeneration has the real SmolVLMModel at .model
        if hasattr(m, "model"):
            m = m.model
        return m

    def _get_vision_model(self):
        base = self._get_base_model()
        if hasattr(base, "vision_model"):
            return base.vision_model
        # Extra safety for some wrapper depths
        if hasattr(base, "model") and hasattr(base.model, "vision_model"):
            return base.model.vision_model
        raise AttributeError(
            "Could not locate vision_model. "
            "The model structure may have changed in this version of transformers. "
            "Try setting 'use_lora': false in CONFIG, or inspect model structure with print(dir(model))."
        )

    def _get_connector(self):
        base = self._get_base_model()
        if hasattr(base, "connector"):
            return base.connector
        if hasattr(base, "model") and hasattr(base.model, "connector"):
            return base.model.connector
        # Some SmolVLM variants use multi_modal_projector
        if hasattr(base, "multi_modal_projector"):
            return base.multi_modal_projector
        raise AttributeError(
            "Could not locate connector (or multi_modal_projector). "
            "Update the _get_connector() helper if the transformers version changed the attribute name."
        )

    def _attach_vision_feature_hook(self):
        """Attach a forward hook to the connector so we capture its output during the main lm forward.
        This lets us reuse the (expensive) vision+connector computation that already happened,
        avoiding a second full vision pass that was causing OOM with 4-8 frames.
        """
        if self._connector_hook_handle is not None:
            self._connector_hook_handle.remove()

        connector = self._get_connector()

        def _hook(module, input, output):
            # Capture the raw connector output. We will pool + release quickly in forward().
            self._vision_features = output

        self._connector_hook_handle = connector.register_forward_hook(_hook)

    def _apply_vision_unfreeze(self, step):
        if getattr(self, "use_lora", False):
            return  # LoRA path does not use manual unfreeze
        layers_to_unfreeze = []
        for sched_step, layers in self.vision_unfreeze_schedule.items():
            if step >= sched_step:
                layers_to_unfreeze.extend(layers)
        vision = self._get_vision_model()
        for name, param in vision.named_parameters():
            param.requires_grad = any(layer in name for layer in layers_to_unfreeze)

        # --- Critical stability fix for progressive unfreeze under full fine-tune ---
        # Newly-activated vision params have no (or stale) Adam moments because they produced ~0 grad while frozen.
        # Clearing their optimizer state here lets the *current* LR scale create fresh moments on the next .step().
        # This eliminates the huge shock updates / loss spikes observed right after the 80/180/350/500 events.
        self._reset_vision_optimizer_states(layers_to_unfreeze)

    def _reset_vision_optimizer_states(self, layers_just_considered):
        """Remove Adam (or other) optimizer state for vision params that are now active.
        Safe to call even if some params were already active; missing state entries are ignored.
        """
        trainer = getattr(self, "_trainer_ref", None)
        if trainer is None or not hasattr(trainer, "optimizer") or trainer.optimizer is None:
            return
        opt = trainer.optimizer
        vision = self._get_vision_model()
        cleared = 0
        for name, param in vision.named_parameters():
            if not param.requires_grad:
                continue
            # Only bother with params that match the layers touched at this schedule point (or all active ones on step 0)
            if layers_just_considered and not any(layer in name for layer in layers_just_considered):
                # On the cumulative schedule we still want to ensure fresh state for anything that just became active.
                # Being slightly over-eager (clearing a few extra) is harmless.
                pass
            if param in opt.state:
                opt.state.pop(param, None)
                cleared += 1
        if cleared > 0:
            # Light hygiene after a real unfreeze
            try:
                torch.cuda.empty_cache()
                import gc
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
        # === CRITICAL FIX for plateau + major memory optimization ===
        # Attach a forward hook to the connector *before* the main call.
        # During the internal vision+connector processing (which the base model must do anyway),
        # the hook captures the connector output with the full autograd graph.
        # This gives us the exact features we need for the cls head and contrastive loss
        # *without* a second full vision_model forward pass.
        # The old duplicate vision pass was extremely expensive in both memory and time
        # when using 4-8 video frames.
        # If precomputed image_hidden_states are provided, bypass vision entirely.
        image_hidden_states = kw.get("image_hidden_states")
        has_precomputed = image_hidden_states is not None

        if not has_precomputed:
            self._attach_vision_feature_hook()

        # Main generative forward
        # To satisfy SmolVLM's get_image_features (which unconditionally does 5D unpack
        # batch, num_images, c, h, w = pixel_values.shape and then unfold on the mask
        # assuming the num_images dim corresponds to a "sub-image/tile count" that is often 1),
        # we force a 5D view with num_images=1 for each individual image/frame.
        # This puts the (potentially large) "num frames" or padded count into the batch dim of vision,
        # and keeps the "tile count" =1 , so the unfold dim1 has size 1 (avoiding "max size 1 but size is 16").
        # Each frame is treated as its own "image" with its own patch grid; the text sequence's <image> tokens
        # (placed per original sample's frames) will consume the features in the provided order.
        pv_for_base = pixel_values if not has_precomputed else None
        pam_for_base = kw.get("pixel_attention_mask") if not has_precomputed else None
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
            image_hidden_states=image_hidden_states,
            output_hidden_states=True
        )

        # Clean up hook
        if not has_precomputed and self._connector_hook_handle is not None:
            self._connector_hook_handle.remove()
            self._connector_hook_handle = None

        connector_out = self._vision_features
        self._vision_features = None  # drop reference immediately

        # When using precomputed features, connector_out is not captured (no hook).
        # The cls head will use precomputed_image_embeds below, so skip vision pooling.
        if not has_precomputed:
            if connector_out is None:
                # Very rare fallback – do a minimal no-grad reconstruction
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

            # Generic per-sample image embed extraction.
            # Supports both the padded 5D form (for model compatibility with SmolVLM's get_image_features
            # which does batch_size, num_images, C, H, W = pixel_values.shape ) and provides num_images
            # for exact per-sample grouping (skipping dummy padded image slots).
            # pixel_values may be 5D [B, max_p, C, H, W] (padded for variable frames) or 4D.
            # We use the provided "num_images" (actual count per text sample) + stride from 5D to correctly
            # average only real images per sample.
            batch_size = input_ids.shape[0] if input_ids is not None else (pixel_values.shape[0] if pixel_values is not None else 1)
            num_images = kw.get("num_images")
            if connector_out is not None:
                if connector_out.dim() == 3:
                    per_unit = connector_out.mean(dim=1)
                else:
                    per_unit = connector_out

                total_units = per_unit.shape[0] if per_unit.dim() > 0 else 1

                if num_images is not None:
                    # Use stride if 5D padded (to skip dummy slots for samples with fewer real frames)
                    if pixel_values.dim() == 5:
                        stride = pixel_values.shape[1]  # max_p
                    else:
                        stride = None

                    image_embeds_list = []
                    pos = 0  # position in the (possibly strided) units
                    for n in num_images.tolist():
                        n = int(n)
                        if n > 0 and pos < total_units:
                            if stride is not None:
                                # for this sample, real units are at the start of its stride group
                                sample_units = per_unit[pos : pos + n]
                                pos += stride  # skip over this sample's full (real + dummy) slots
                            else:
                                sample_units = per_unit[pos : pos + n]
                                pos += n
                            if sample_units.dim() > 1 and sample_units.shape[0] > 0:
                                emb = sample_units.mean(dim=0)
                            else:
                                emb = sample_units
                            image_embeds_list.append(emb)
                        else:
                            dim = per_unit.shape[-1] if per_unit.dim() > 0 else 576
                            image_embeds_list.append( torch.zeros(dim, device=per_unit.device if per_unit is not None else "cpu", dtype=per_unit.dtype if per_unit is not None else torch.float32) )
                            if stride is not None:
                                pos += stride
                            else:
                                pos += n
                    image_embeds = torch.stack(image_embeds_list, dim=0)
                else:
                    # Legacy uniform case
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
            else:
                dim = 576
                device = pixel_values.device if pixel_values is not None else "cpu"
                image_embeds = torch.zeros(batch_size, dim, device=device, dtype=torch.float32)
        else:
            # Precomputed path: image_embeds set below from precomputed_image_embeds
            image_embeds = None

        # Aggressively release the (potentially huge) full connector sequence tensor
        if 'connector_out' in locals() and connector_out is not None:
            del connector_out
        if 'pooled' in locals():
            del pooled
        torch.cuda.empty_cache()

        # If the batch came with precomputed (pooled) image embeds (from offline extraction
        # when vision is frozen under LoRA), use them directly. This avoids running vision
        # again for the custom heads and is the key to running on a heavily shared GPU.
        if "precomputed_image_embeds" in kw and kw["precomputed_image_embeds"] is not None:
            pre = kw["precomputed_image_embeds"]
            if pre.dim() == 1:
                pre = pre.unsqueeze(0)
            image_embeds = pre.float()

        # Keep classification-head dtype aligned with the pooled vision features.
        cls_param = next(self.cls_head.parameters())
        cls_input = image_embeds.to(dtype=cls_param.dtype)
        cls_logits = self.cls_head(cls_input)

        # Test-time fast path. Classification evaluation does not need the
        # contrastive heads, so return before executing those auxiliary layers.
        if kw.get("classification_only", False):
            return {
                "cls_logits": cls_logits,
                "image_embeds": image_embeds,
            }

        # Text side for contrastive: last real token (under teacher-forcing this lands in the explanation)
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
# ================== VIDEO DATASET (FIXED path resolution + processor handling) ==================
class VideoFakeClueDataset(Dataset):
    """
    Video dataset with:
    - robust path resolution (basename index + many layout candidates for the actual
      extracted FaceForensics++_C23 structure + the mangled names in ExDDV JSONs)
    - multi-frame loading (configurable num_frames)
    - cheap on-the-fly augmentations (flip + brightness) when frame_augment=True
    - jittered temporal sampling (helps catch motion artifacts in deepfakes)
    The return contract (pixel_values, input_ids, ..., cls_label) is unchanged.
    """
    def __init__(self, data, proc, ml=2048, num_frames=None, local_dataset_root=None, hf_split=None, augment=None):
        self.data = data
        self.proc = proc
        self.ml = ml
        self.local_dataset_root = Path(local_dataset_root or CONFIG["local_dataset_root"])
        self.hf_split = hf_split or CONFIG["hf_split"]
        # Honor CONFIG (new default 8) while allowing explicit override
        if num_frames is None:
            num_frames = CONFIG.get("num_frames", 6)
        self.num_frames = int(num_frames)
        self.augment = bool(CONFIG.get("frame_augment", False)) if augment is None else bool(augment)
        decord.bridge.set_bridge("torch")

    def __len__(self):
        return len(self.data)

    def _load_video_frames(self, video_path):
        try:
            vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
            total = len(vr)
            if total == 0:
                raise ValueError

            # Stratified sampling + small random jitter (better coverage of temporal artifacts)
            base = [i * total / self.num_frames for i in range(self.num_frames)]
            jitter = [random.uniform(-total * 0.025, total * 0.025) for _ in range(self.num_frames)]
            indices = [max(0, min(total - 1, int(b + j))) for b, j in zip(base, jitter)]

            frames = vr.get_batch(indices)
            frames = [Image.fromarray(f.numpy()) for f in frames]

            # Lightweight on-the-fly augmentation (zero new heavy deps).
            if getattr(self, "augment", False):
                if random.random() < 0.5:
                    frames = [f.transpose(Image.FLIP_LEFT_RIGHT) for f in frames]
                if random.random() < 0.35:
                    factor = random.uniform(0.82, 1.18)
                    frames = [
                        Image.eval(f, lambda x: max(0, min(255, int(x * factor))))
                        for f in frames
                    ]

            return frames
        except Exception:
            print(f"⚠️ Could not load {video_path} — using gray fallback")
            n = self.num_frames
            return [Image.new("RGB", (512, 512), "gray")] * n

    def _find_assistant_start(self, input_ids):
        """Locate the first token of the assistant RESPONSE (after the header)."""
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
                    if input_ids[i : i + L].tolist() == m_ids:
                        return i + L
            except Exception:
                pass
        print("⚠️  _find_assistant_start: could not locate assistant header via tokens; using 70% heuristic")
        return int(len(input_ids) * 0.70)

    # ---------------------------------------------------------------------
    # HF-HOSTED VIDEO RESOLUTION (new data format: filename/dataset_label/caption)
    # ---------------------------------------------------------------------
    # Your captions file (see sample you pasted) has entries shaped like:
    #   {"filename": "real/train_real_001.mp4", "dataset_label": "real", "caption": "..."}
    # "filename" is relative to the split folder (e.g. "train/") in the Hub repo
    # LikhithaGeddam/fakevideodetection-private. We resolve + download it on demand via
    # huggingface_hub, which caches under ~/.cache/huggingface/hub by default — so each
    # video is downloaded at most once across the whole run (and across future runs,
    # until you clear that cache). No manual pre-downloading step is required.
    #
    # SAFETY NET: this repo is private/gated, so an auth problem makes every single
    # fetch fail — silently falling back to gray placeholder frames for the entire
    # dataset would waste a full GPU run without anyone noticing until the loss curve
    # looks meaningless. We track consecutive failures and hard-abort past a threshold
    # instead of degrading silently. See preflight_check_hf_access() in main() for the
    # check that runs before trainer.train() even starts.
    _consecutive_fetch_failures = 0
    _FETCH_FAILURE_ABORT_THRESHOLD = 8

    def _resolve_video(self, filename):
        """Local-first video resolution: since the dataset now lives on Drive
        (CONFIG['local_dataset_root']), check there first — no network call, no
        per-video download, so this is the main source of the runtime improvement.
        Falls back to streaming from the Hub only if the local file is missing
        (e.g. Drive copy incomplete), so a partial local mirror still works.
        """
        if CONFIG.get("use_local_dataset"):
            # Flat structure: <local_dataset_root>/real/*.mp4, <local_dataset_root>/fake/*.mp4
            # — no train/ split subfolder locally, unlike the Hub repo layout. filename already
            # includes the real/fake prefix (e.g. "real/train_real_001.mp4").
            local_path = self.local_dataset_root / filename
            if local_path.exists():
                self.__class__._consecutive_fetch_failures = 0
                return str(local_path)
            # Not found locally — fall through to Hub streaming below.
        return self._resolve_hf_video(filename)

    def _resolve_hf_video(self, filename):
        repo_path = f"{self.hf_split}/{filename}"
        try:
            from huggingface_hub import hf_hub_download
            local_path = hf_hub_download(
                repo_id=CONFIG["hf_repo_id"], repo_type="dataset", filename=repo_path,
                token=CONFIG.get("hf_token"),  # None -> huggingface_hub auto-detects HF_TOKEN env / cached login
            )
            self.__class__._consecutive_fetch_failures = 0
            return local_path
        except Exception as e:
            self.__class__._consecutive_fetch_failures += 1
            print(f"⚠️  Could not fetch {repo_path} from {CONFIG['hf_repo_id']}: {e}")
            if self.__class__._consecutive_fetch_failures >= self._FETCH_FAILURE_ABORT_THRESHOLD:
                raise RuntimeError(
                    f"{self._FETCH_FAILURE_ABORT_THRESHOLD} consecutive video fetches failed from "
                    f"'{CONFIG['hf_repo_id']}'. This almost always means authentication is missing/wrong "
                    "for this private dataset. Run `from huggingface_hub import login; login()` in a cell "
                    "before this script, or set os.environ['HF_TOKEN'] = '...', then restart and retry. "
                    "Aborting now instead of silently training on gray placeholder frames."
                )
            return None

    def _build_conversation_from_caption(self, label, caption):
        """Build a human/assistant turn pair on the fly from a real/fake label + a
        real caption, matching the same prompt style the model was designed around."""
        is_fake = str(label).lower() == "fake"
        question = FAKE_QUESTION if is_fake else REAL_QUESTION
        answer = f"This video is {'FAKE' if is_fake else 'REAL'}. {caption}"
        return [
            {"from": "human", "value": f"<image>\n{question}"},
            {"from": "gpt", "value": answer},
        ]

    def __getitem__(self, idx):
        item = self.data[idx]

        # All entries use the captions schema: filename / dataset_label / caption.
        label = item.get("dataset_label", "real")
        caption = item.get("caption", "")
        conversations = self._build_conversation_from_caption(label, caption)
        video_path = self._resolve_video(item["filename"])
        cls_label_val = 1 if str(label).lower() == "fake" else 0

        frames = self._load_video_frames(video_path) if video_path else [Image.new("RGB", (512, 512), "gray")] * self.num_frames

        # Build conversation — ensure exact number of <image> tokens
        msgs = []
        for c in conversations:
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
            "cls_label": cls_label_val,
        }

# ================== COLLATOR, TRAINER, CALLBACK ==================
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
        has_real_pv = any(pv is not None for pv in pvs)
        if has_real_pv:
            real_pvs = [pv for pv in pvs if pv is not None]
            p_shapes = [pv.shape for pv in real_pvs]
            if len(set(s[0] for s in p_shapes)) == 1 and len(real_pvs) == len(pvs):
                pv_batch = torch.stack(pvs, dim=0)
            else:
                max_p = max(pv.shape[0] for pv in real_pvs)
                C, H, W = real_pvs[0].shape[1], real_pvs[0].shape[2], real_pvs[0].shape[3]
                pv_batch = torch.zeros((len(feats), max_p, C, H, W), dtype=real_pvs[0].dtype)
                for i, pv in enumerate(pvs):
                    if pv is not None:
                        pv_batch[i, :pv.shape[0]] = pv
            batch["pixel_values"] = pv_batch
            num_images_list = [pv.shape[0] if pv.dim() == 4 else 1 for pv in pvs]
            batch["num_images"] = torch.tensor(num_images_list, dtype=torch.long)
        else:
            batch["pixel_values"] = None
            batch["num_images"] = torch.tensor([1] * len(feats), dtype=torch.long)

        batch["cls_label"] = torch.tensor([f["cls_label"] for f in feats])

        if any("precomputed_image_embeds" in f for f in feats):
            pre_embeds = [f.get("precomputed_image_embeds", torch.zeros(576)) for f in feats]
            batch["precomputed_image_embeds"] = torch.stack([p.float() for p in pre_embeds])
        if any("image_hidden_states" in f for f in feats):
            ihs = [f.get("image_hidden_states", torch.zeros(1, 576)) for f in feats]
            batch["image_hidden_states"] = torch.stack([p.float() for p in ihs])

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
        cls_head_params = []
        vision_params = []
        contr_head_params = []
        base_params = []
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
            # Dataset-level weights (see FIXED_CLS_WEIGHTS definition) — applies real
            # imbalance correction on every step, unlike the old per-batch version below,
            # which degenerated to uniform weighting whenever a batch had zero real examples
            # (the common case at ~13% real with batch size 4).
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

        decay_start = CONFIG.get("late_aux_decay_start", 2000)
        if self.state.global_step > decay_start:
            if not getattr(self, "_late_phase_logged", False):
                print(f"[late phase @ {self.state.global_step}] aux heavily down-weighted for LM focus "
                      "(contr=0, cls very low). LM term now dominates generation learning.")
                self._late_phase_logged = True

            contr_weight = 0.0
            target_cls_w = CONFIG.get("late_cls_weight", 0.1)
            base_cls_w = CONFIG.get("cls_loss_weight", 1.0)
            if base_cls_w > 1e-6:
                cls_loss = cls_loss * (target_cls_w / base_cls_w)

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
    """
    Lightweight qualitative monitor for the LM/explanation side.
    Every generation_sample_interval real optimizer steps (when >0), pick 1-2 examples,
    run a short greedy/sampled generation, and print a compact "GT vs. Pred" line.
    Schema-agnostic: works for both the new captions format (filename/dataset_label/caption)
    and the old (image/conversations/label) format.
    """
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

    def _resolve_frames_and_texts(self, raw):
        """Returns (frames, human_text, gt_text), reusing 1 frame for cheap monitoring
        regardless of the training num_frames setting."""
        # Generation examples come from val_dataset when available,
        # so resolve their videos using the same dataset/split.
        ds = self.val_dataset if self.val_dataset is not None else self.train_dataset
        label = raw.get("dataset_label", "real")
        caption = raw.get("caption", "")
        is_fake = str(label).lower() == "fake"
        human_text = FAKE_QUESTION if is_fake else REAL_QUESTION
        gt_text = f"This video is {'FAKE' if is_fake else 'REAL'}. {caption}"
        frames = None
        if ds is not None and hasattr(ds, "_resolve_video"):
            try:
                old_nf = getattr(ds, "num_frames", 1)
                ds.num_frames = 1
                video_path = ds._resolve_video(raw["filename"])
                if video_path:
                    frames = ds._load_video_frames(video_path)
                ds.num_frames = old_nf
            except Exception:
                frames = None
        return frames, human_text, gt_text

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
                    frames, human_text, gt = self._resolve_frames_and_texts(raw)
                    if not frames:
                        frames = [Image.new("RGB", (512, 512), "gray")]

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
    """
    NEW: The original script builds a val_dataset (via val_ratio) but eval_strategy="no"
    means it was never actually evaluated on — the val split existed but did nothing.
    This callback runs a real classification-accuracy pass over val_dataset at the end
    of every epoch and checkpoints the best-scoring model separately, so you have an
    objective signal (and a best checkpoint) instead of only the qualitative gen samples.
    """
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
    import math
    import os

    data_root = Path(CONFIG["data_root"])
    jp = data_root / CONFIG["train_json"]
    if not jp.exists():
        for candidate in ("caption_progress_gemma4_12b_video.jsonl", "trainv2.json", "train.json", "captions.json"):
            p = data_root / candidate
            if p.exists():
                jp = p
                break
    if not jp.exists():
        raise FileNotFoundError(
            f"Could not find {CONFIG['train_json']} (or fallback names) under {data_root}. "
            "Point CONFIG['data_root']/CONFIG['train_json'] at your captions file."
        )
    raw_data = []
    with open(jp, encoding="utf-8") as f:
        if jp.suffix.lower() == ".jsonl":
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                filename = item.get("filename", "")
                if filename.startswith("real/"):
                    item["dataset_label"] = "real"
                elif filename.startswith("fake/"):
                    item["dataset_label"] = "fake"
                else:
                    raise ValueError(f"Cannot infer label at line {line_no}: {filename}")
                raw_data.append(item)
        else:
            raw_data = json.load(f)

    print(f"📄 Using annotations: {jp.name} ({len(raw_data):,} entries) "
          f"— schema: filename/dataset_label/caption")

    if CONFIG.get("subset_ratio", 1.0) < 1.0:
        raw_data = raw_data[: int(len(raw_data) * CONFIG["subset_ratio"])]

    # === Use the dataset's dedicated validation split; do NOT carve train ===
    val_root = Path(CONFIG["val_dataset_root"])
    val_data = []
    for label in ("real", "fake"):
        d = val_root / label
        if not d.exists():
            raise FileNotFoundError(f"Missing validation directory: {d}")
        for vp in sorted(d.iterdir()):
            if vp.is_file() and vp.suffix.lower() in {".mp4", ".flv"}:
                val_data.append({
                    "filename": f"{label}/{vp.name}",
                    "dataset_label": label,
                    "caption": "",
                })
    if len(val_data) != 3304:
        raise RuntimeError(f"Expected 3304 validation videos, found {len(val_data)}")
    val_reals = sum(1 for x in val_data if x["dataset_label"] == "real")
    val_fakes = len(val_data) - val_reals
    print(f"✅ Dedicated validation split → {len(val_data):,} videos | Real: {val_reals:,} | Fake: {val_fakes:,}")

    reals = sum(1 for x in raw_data if str(x.get("dataset_label") or "").lower() == "real")
    fakes = len(raw_data) - reals
    print(f"✅ Loaded {len(raw_data):,} videos — Real: {reals} | Fake: {fakes}")
    if reals == 0 or fakes == 0:
        print("⚠️  WARNING: One class is completely missing (or extremely rare).")
        print("    The classification head cannot learn a meaningful real/fake decision boundary.")
    elif reals > 0 and fakes > 0:
        # Fixed, dataset-level class weights (see FIXED_CLS_WEIGHTS definition near the top).
        # Standard inverse-frequency weighting: rarer class gets a proportionally larger weight.
        total_n = reals + fakes
        w_real = total_n / (2.0 * reals)
        w_fake = total_n / (2.0 * fakes)
        FIXED_CLS_WEIGHTS = torch.tensor([w_real, w_fake], dtype=torch.float32)
        print(f"⚖️  Fixed dataset-level CLS weights (real={reals}, fake={fakes}) — "
              f"real: {w_real:.3f}, fake: {w_fake:.3f}")
        if reals / total_n < 0.2:
            print(f"   ⚠️  Only {100*reals/total_n:.1f}% real in this train split. A trivial "
                  f"'always predict fake' classifier would score ~{100*fakes/total_n:.1f}% accuracy — "
                  "report precision/recall per class, not just overall accuracy, for this to be meaningful.")

    if CONFIG.get("use_local_dataset"):
        print(f"ℹ️  Videos will be read directly from local Drive dataset at "
              f"'{CONFIG['local_dataset_root']}' — Hub streaming is only a per-video fallback "
              f"for anything missing there.")
    else:
        print(f"ℹ️  Videos will be streamed on demand from Hub repo '{CONFIG['hf_repo_id']}' "
              f"(split='{CONFIG['hf_split']}') and cached locally by huggingface_hub.")

    # --- Preflight local-dataset / auth check --------------------------------------------
    # A wrong local_dataset_root (or an incomplete Drive copy) would otherwise only surface
    # as "the loss curve never moves" after a full wasted run — every miss would silently
    # fall through to Hub streaming (slow) or a gray placeholder frame (meaningless). Check
    # a handful of real samples now and report exactly what's going on before committing to
    # a multi-hour run.
    def preflight_check_access(data, n_probe=8):
        probe_items = data[:n_probe] if len(data) >= n_probe else data
        local_ok, hub_ok, total_fail = 0, 0, 0
        last_err = None

        if CONFIG.get("use_local_dataset"):
            root = Path(CONFIG["local_dataset_root"])
            if not root.exists():
                raise RuntimeError(
                    f"CONFIG['local_dataset_root'] = '{root}' does not exist. Double-check the exact "
                    "path to your uploaded dataset folder on Drive and fix that line before rerunning."
                )
            for item in probe_items:
                p = root / item["filename"]
                if p.exists():
                    local_ok += 1
            print(f"📂 Local check: {local_ok}/{len(probe_items)} sample videos found under {root} "
                  f"(flat real/fake structure, no split subfolder).")
            if local_ok == 0:
                print("⚠️  None of the probed videos were found locally — check that the folder structure "
                      f"under '{root}' really matches <local_dataset_root>/real|fake/*.mp4. "
                      "Falling through to verify Hub access as a backup, since every miss will otherwise "
                      "stream from the Hub per-video (slow).")

        if not CONFIG.get("use_local_dataset") or local_ok < len(probe_items):
            from huggingface_hub import hf_hub_download
            for item in probe_items:
                repo_path = f"{CONFIG['hf_split']}/{item['filename']}"
                local_path = Path(CONFIG.get("local_dataset_root", "")) / item["filename"] if CONFIG.get("use_local_dataset") else None
                if local_path is not None and local_path.exists():
                    continue  # already confirmed above
                try:
                    hf_hub_download(
                        repo_id=CONFIG["hf_repo_id"], repo_type="dataset", filename=repo_path,
                        token=CONFIG.get("hf_token"),
                    )
                    hub_ok += 1
                except Exception as e:
                    total_fail += 1
                    last_err = e

        if CONFIG.get("use_local_dataset") and local_ok == len(probe_items):
            print(f"✅ Preflight check passed: all {len(probe_items)} probed videos found locally on Drive.")
            return
        if local_ok + hub_ok == 0:
            raise RuntimeError(
                f"Preflight check failed: none of {len(probe_items)} sample videos were found locally "
                f"OR fetchable from the Hub. Last Hub error: {last_err}\n"
                "Check CONFIG['local_dataset_root'] first; if that's correct, this is likely a Hub auth "
                "issue for whatever's missing locally — run `from huggingface_hub import login; login()`. "
                "Aborting now instead of silently training on gray placeholder frames."
            )
        print(f"✅ Preflight check: {local_ok} local + {hub_ok} via Hub fallback out of "
              f"{len(probe_items)} probed — good enough to proceed.")

    preflight_check_access(raw_data)

    # === Load model ===
    model = MultiTaskSmolVLM(
        CONFIG["model_id"],
        CONFIG["head_lr_multiplier"],
        CONFIG["vision_lr_multiplier"],
    )

    train_dataset = VideoFakeClueDataset(
        raw_data, model.processor, ml=CONFIG["max_seq_length"], num_frames=CONFIG["num_frames"],
        local_dataset_root=CONFIG["local_dataset_root"], hf_split="train", augment=True
    )
    val_dataset = VideoFakeClueDataset(
        val_data, model.processor, num_frames=CONFIG["num_frames"],
        local_dataset_root=CONFIG["val_dataset_root"], hf_split="val", augment=False
    )

    # === BUG FIX carried over from the earlier frames-pipeline work: the schedule keys
    # (contrastive_ramp_steps / boost_after_step / late_aux_decay_start) were hardcoded to
    # numbers derived from a specific old dataset size (7004 samples, 2 epochs). Since the
    # dataset changed, those numbers no longer mean what the comments claim. Compute them
    # from the ACTUAL total step count instead, so the schedule scales correctly regardless
    # of how many videos you end up training on. ===
    steps_per_epoch = math.ceil(
        len(train_dataset) / (CONFIG["per_device_train_batch_size"] * CONFIG["gradient_accumulation_steps"])
    )
    total_train_steps = steps_per_epoch * CONFIG["num_train_epochs"]

    CONFIG["contrastive_ramp_steps"] = max(1, int(0.10 * total_train_steps))
    CONFIG["boost_after_step"] = max(1, int(0.35 * total_train_steps))
    CONFIG["late_aux_decay_start"] = max(1, int(0.65 * total_train_steps))
    CONFIG["generation_sample_interval"] = max(1, int(0.02 * total_train_steps))

    print(f"📐 Total training steps: {total_train_steps} ({steps_per_epoch}/epoch × {CONFIG['num_train_epochs']} epochs) | "
          f"contrastive_ramp_steps={CONFIG['contrastive_ramp_steps']} | "
          f"boost_after_step={CONFIG['boost_after_step']} | "
          f"late_aux_decay_start={CONFIG['late_aux_decay_start']} | "
          f"generation_sample_interval={CONFIG['generation_sample_interval']}")

    # Build the full desired kwargs, then filter to whatever this installed transformers
    # version's TrainingArguments actually accepts. Your Colab environment auto-upgraded to
    # transformers v5, which removed warmup_ratio (merged into warmup_steps, which now also
    # accepts a float <1 to mean "ratio of total steps") — translate instead of crashing or
    # silently losing the warmup schedule.
    import inspect
    desired_ta_kwargs = dict(
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
        save_strategy="steps",
        save_steps=250,
        save_total_limit=3,
        eval_strategy="no",
        remove_unused_columns=False,
        report_to="none",
        disable_tqdm=False,
    )
    valid_params = set(inspect.signature(TrainingArguments.__init__).parameters.keys())
    ta_kwargs = {k: v for k, v in desired_ta_kwargs.items() if k in valid_params}
    dropped = sorted(set(desired_ta_kwargs) - set(ta_kwargs))
    if dropped:
        import transformers as _tf
        print(f"⚠️  transformers=={_tf.__version__} doesn't accept these TrainingArguments params "
              f"(dropped): {dropped}")
        if "warmup_ratio" in dropped and "warmup_steps" in valid_params:
            ta_kwargs["warmup_steps"] = CONFIG["warmup_ratio"]
            print(f"   -> transformers v5: translated warmup_ratio={CONFIG['warmup_ratio']} "
                  f"to warmup_steps={CONFIG['warmup_ratio']} (v5 accepts a float <1 as a ratio here).")
    args = TrainingArguments(**ta_kwargs)

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
            max_eval_samples=600,
        ))

    mode = "full fine-tune (progressive unfreeze, no LoRA)" if not getattr(model, "use_lora", False) else "LoRA"
    print(f"\n✅ Starting multi-task video fine-tuning ({mode}) — real/fake + explanations")

    torch.cuda.empty_cache()
    gc.collect()

    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        is_full_finetune = not CONFIG.get("use_lora", True)
        min_required_gb = 3.0 if is_full_finetune else 1.5
        if free < min_required_gb * 1024**3:
            mode_str = "full fine-tune (no LoRA)" if is_full_finetune else "current settings"
            print(f"\n❌ FATAL: Only {free/1024**3:.2f} GiB free on GPU. You need at least ~{min_required_gb} GiB free for this process ({mode_str}).")
            raise RuntimeError("Insufficient free GPU memory due to other processes.")

    # Automatically resume from the newest Trainer checkpoint.
    output_dir = Path(CONFIG["output_dir"])

    checkpoints = []

    if output_dir.exists():
        for cp in output_dir.glob("checkpoint-*"):
            try:
                step = int(cp.name.split("-")[-1])
                checkpoints.append((step, cp))
            except ValueError:
                pass

    if checkpoints:
        checkpoints.sort(key=lambda x: x[0])
        latest_checkpoint = str(checkpoints[-1][1])

        print(
            f"🔄 AUTO-RESUME: {latest_checkpoint}"
        )

        resume_checkpoint = latest_checkpoint
    else:
        print("🆕 No checkpoint found. Starting from step 0.")
        resume_checkpoint = None

    trainer.train(
        resume_from_checkpoint=resume_checkpoint
    )

    trainer.save_model(CONFIG["output_dir"])
    model.processor.save_pretrained(CONFIG["output_dir"])

    extra = {
        "cls_head": model.cls_head.state_dict(),
        "contrastive_vision_proj": model.contrastive_vision_proj.state_dict(),
        "contrastive_text_proj": model.contrastive_text_proj.state_dict(),
    }
    torch.save(extra, Path(CONFIG["output_dir"]) / "multitask_heads.pt")

    print(f"\n🎉 Training finished! Model + processor + multitask_heads.pt saved to {CONFIG['output_dir']}")
    print(" • Classification head (real/fake) trained with gradient flow into vision features")
    print(" • LM head generates natural-language explanations")
    print(" • Best val-accuracy checkpoint saved separately under best_cls_acc/ (see ValidationClsAccuracyCallback)")
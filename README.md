# FakeVideoDetection-SMOLVLM2

Fine-tuning SmolVLM2-256M-Instruct into a multi-task deepfake video detector
that jointly predicts a real/fake label and generates a natural-language
explanation of the visual artifacts supporting that prediction.

Full write-up: [`docs/report.pdf`](docs/report.pdf).

## Dataset

Training data is hosted as a private dataset repository on the Hugging Face
Hub: [`LikhithaGeddam/fakevideodetection-private`](https://huggingface.co/datasets/LikhithaGeddam/fakevideodetection-private).
Videos are labeled `real`/`fake` and organized into `train`/`val`/`test` splits
(16,166 / 3,304 / 3,465 videos respectively). You'll need read access to this
dataset (or your own equivalent, in the same JSON schema) to reproduce
training.

## Repository structure

```
training/
  finetune_lora_video_based.py    # Original pipeline: reads video directly (decord), streams
                                   # from the HF Hub dataset with local Drive mirroring.
  finetune_smolvlm2_frames.py     # Current pipeline: trains from precomputed JPEG frames
                                   # instead of decoding video (see "Data pipeline" below).
evaluation/
  eval_checkpoints.py             # Reload saved per-epoch checkpoints and compute
                                   # val accuracy / precision / recall (real class) without retraining.
  run_ablations.py                # Short-budget ablation runner: frame count, contrastive
                                   # alignment weight, and learning rate, evaluated on a
                                   # shared stratified validation subsample.
docs/
  report.pdf / report.tex         # Full project report (ACM format).
```

## Data pipeline

Training does **not** read video files directly. A two-stage pipeline is used:

1. **One-time frame precomputation** (`precompute_frames_streaming.py` — not
   yet included in this repo; see note below) streams each video once from
   the Hugging Face Hub dataset, extracts a fixed pool of 8 frames per video,
   and writes them to local disk as JPEGs under
   `<data_root>/frames/<split>/{real,fake}/<video_stem>/f0.jpg ... f7.jpg`,
   alongside `train.json` / `val.json` / `test.json` manifests referencing
   frame paths. This keeps the persistent disk footprint under 10 GB.
2. **Frames-only training** (`training/finetune_smolvlm2_frames.py`) reads
   only these JPEGs — no video decoding library is used at training time.

> **TODO**: `precompute_frames_streaming.py` is referenced by
> `finetune_smolvlm2_frames.py`'s docstring but has not been added to this
> repo yet — add it alongside the training scripts so the pipeline is fully
> reproducible from the dataset alone.

## Setup

```bash
pip install -r requirements.txt
```

Set `data_root` in whichever training script's `CONFIG` dict you're using to
point at your local copy of the dataset (frames-based pipeline) or your Drive
mount (video-based pipeline).

## Training

```bash
# Current (frames-based) pipeline — recommended
python training/finetune_smolvlm2_frames.py

# Original (video-based) pipeline
python training/finetune_lora_video_based.py
```

Both use LoRA (rank 32, α=64) on the language backbone's attention and
feed-forward projections, with the vision encoder frozen and the
classification head, contrastive alignment heads, and multimodal connector
fully trainable. See `docs/report.pdf` Section 4–5 for the full architecture
and training-objective description.

## Evaluation

```bash
# Per-epoch val accuracy / precision / recall from saved checkpoints, no retraining
python evaluation/eval_checkpoints.py

# Ablation studies (frame count, contrastive weight, learning rate)
python evaluation/run_ablations.py
```

Both scripts import directly from the training script's `CONFIG` and model
classes (rather than duplicating them), so keep them in the same directory as
the training script they're evaluating, or adjust the `import` at the top of
each.

## Reproducibility notes

- Fixed seed (`42`) is used everywhere a train/val split is carved from raw
  data, so re-running any script against the same `train.json` reproduces the
  identical split.
- `FIXED_CLS_WEIGHTS` (classification loss class weights) is computed once
  from the true training-set real/fake counts — see `docs/report.pdf`
  Section 3.2 for why this matters and how it's computed.
- Checkpoints are saved every epoch; a separate `best_cls_acc/` checkpoint
  tracks the best validation classification accuracy seen so far,
  independent of the most recent epoch checkpoint.

## Known limitations

See `docs/report.pdf` Sections 8–9 (Limitations, Failure Analysis) for a full
discussion, including an observed disagreement between the classification
head's predicted label and the label stated in the generated explanation
text on a subset of held-out examples.

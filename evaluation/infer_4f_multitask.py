#!/usr/bin/env python3

import os
import re
import csv
import json
import time
from pathlib import Path

import torch
import decord
from PIL import Image
from safetensors.torch import load_file
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

# ============================================================
# PATHS / CONFIG
# ============================================================

ROOT = Path("/scratch/lgeddam1/gemma_captioning")

TRAIN_SCRIPT = ROOT / "scripts/finetune_smolvlm2_research.py"

MODEL_DIR = (
    ROOT / "checkpoints/smolvlm2_multitask_4frames"
)

FINAL_CHECKPOINT = (
    MODEL_DIR / "checkpoint-3033/model.safetensors"
)

TEST_ROOT = ROOT / "dataset/test"

OUT_DIR = ROOT / "evaluation/4frames"
OUT_DIR.mkdir(parents=True, exist_ok=True)

JSONL_PATH = OUT_DIR / "multitask_predictions.jsonl"
CSV_PATH = OUT_DIR / "multitask_test_results.csv"
METRICS_PATH = OUT_DIR / "multitask_metrics.json"

NUM_FRAMES = 4

QUESTION = (
    "Watch these video frames carefully. Is this video real or a deepfake? "
    "Determine whether the video is REAL or FAKE and describe the specific visual "
    "evidence that supports your decision."
)

MAX_NEW_TOKENS = 160

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

decord.bridge.set_bridge("torch")


# ============================================================
# IMPORT TRAINING MODEL DEFINITION
# ============================================================

# Make sure importing the training module constructs the same
# 4-frame architecture.
os.environ["SMOL_NUM_FRAMES"] = "4"
os.environ["SMOL_BATCH_SIZE"] = "4"
os.environ["SMOL_GRAD_ACCUM"] = "4"
os.environ["SMOL_OUTPUT_DIR"] = str(MODEL_DIR)

import importlib.util

spec = importlib.util.spec_from_file_location(
    "smol_training",
    str(TRAIN_SCRIPT),
)

training = importlib.util.module_from_spec(spec)
spec.loader.exec_module(training)

MultiTaskSmolVLM = training.MultiTaskSmolVLM
CONFIG = training.CONFIG


# ============================================================
# VIDEO LOADING
# ============================================================

def load_frames(video_path, num_frames=4):
    vr = decord.VideoReader(
        str(video_path),
        ctx=decord.cpu(0),
    )

    total = len(vr)

    if total <= 0:
        raise RuntimeError(
            f"Video has no readable frames: {video_path}"
        )

    # Deterministic evenly-spaced inference sampling.
    # NO training-time random jitter or augmentation.
    if num_frames == 1:
        indices = [total // 2]
    else:
        indices = [
            round(i * (total - 1) / (num_frames - 1))
            for i in range(num_frames)
        ]

    batch = vr.get_batch(indices)

    frames = [
        Image.fromarray(frame.numpy()).convert("RGB")
        for frame in batch
    ]

    return frames


# ============================================================
# TEST SET
# ============================================================

def build_test_records():
    records = []

    for label in ("real", "fake"):
        folder = TEST_ROOT / label

        if not folder.exists():
            raise RuntimeError(
                f"Missing test directory: {folder}"
            )

        files = sorted(
            [
                p
                for p in folder.iterdir()
                if p.suffix.lower()
                in {".mp4", ".flv", ".avi", ".mov", ".mkv"}
            ]
        )

        for p in files:
            records.append(
                {
                    "filename": f"{label}/{p.name}",
                    "path": p,
                    "ground_truth": label,
                }
            )

    return records


# ============================================================
# RESUME SUPPORT
# ============================================================

def load_completed():
    completed = {}

    if not JSONL_PATH.exists():
        return completed

    with open(JSONL_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(line)
                completed[row["filename"]] = row
            except Exception:
                # Ignore an incomplete final line if Slurm killed
                # the process while it was being written.
                continue

    return completed


def append_result(row):
    with open(JSONL_PATH, "a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                row,
                ensure_ascii=False,
            )
            + "\n"
        )

        # Make every video durable immediately.
        f.flush()
        os.fsync(f.fileno())


# ============================================================
# LM PARSING
# ============================================================

def parse_lm_prediction(text):
    t = text.strip()

    patterns = [
        r"^\s*this video is\s+(real|fake)\b",
        r"^\s*the video is\s+(real|fake)\b",
        r"^\s*(real|fake)\b",
    ]

    for pattern in patterns:
        m = re.search(
            pattern,
            t,
            flags=re.IGNORECASE,
        )

        if m:
            return m.group(1).lower()

    # Conservative fallback:
    # look only near the beginning rather than interpreting
    # words buried in the explanation.
    prefix = t[:120].lower()

    real_pos = prefix.find("real")
    fake_pos = prefix.find("fake")

    if real_pos >= 0 and fake_pos < 0:
        return "real"

    if fake_pos >= 0 and real_pos < 0:
        return "fake"

    if real_pos >= 0 and fake_pos >= 0:
        return (
            "real"
            if real_pos < fake_pos
            else "fake"
        )

    return "unknown"


def extract_caption(text, lm_prediction):
    text = text.strip()

    if lm_prediction in {"real", "fake"}:
        patterns = [
            rf"^\s*this video is\s+{lm_prediction}\s*[\.\:\-]?\s*",
            rf"^\s*the video is\s+{lm_prediction}\s*[\.\:\-]?\s*",
            rf"^\s*{lm_prediction}\s*[\.\:\-]?\s*",
        ]

        for pattern in patterns:
            cleaned = re.sub(
                pattern,
                "",
                text,
                count=1,
                flags=re.IGNORECASE,
            )

            if cleaned != text:
                return cleaned.strip()

    return text


# ============================================================
# LOAD FINAL MULTI-TASK MODEL
# ============================================================

def load_model():
    print("=" * 70)
    print("LOADING 4-FRAME FINAL MULTI-TASK MODEL")
    print("=" * 70)

    if not FINAL_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Missing checkpoint: {FINAL_CHECKPOINT}"
        )

    model = MultiTaskSmolVLM(
        CONFIG["model_id"],
        CONFIG["head_lr_multiplier"],
        CONFIG["vision_lr_multiplier"],
    )

    print(
        "Loading final Trainer checkpoint:",
        FINAL_CHECKPOINT,
    )

    state = load_file(
        str(FINAL_CHECKPOINT),
        device="cpu",
    )

    missing, unexpected = model.load_state_dict(
        state,
        strict=False,
    )

    print("Missing keys:", len(missing))
    print("Unexpected keys:", len(unexpected))

    if missing:
        print(
            "First missing keys:",
            missing[:20],
        )

    if unexpected:
        print(
            "First unexpected keys:",
            unexpected[:20],
        )

    model.to(DEVICE)
    model.eval()

    # Keep classification head dtype compatible with
    # pooled BF16 vision representations.
    cls_dtype = next(
        model.cls_head.parameters()
    ).dtype

    print("Device:", DEVICE)
    print(
        "Classification head dtype:",
        cls_dtype,
    )

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    print("=" * 70)

    return model


# ============================================================
# CLASSIFICATION FROM THE SAME FORWARD FEATURES
# ============================================================

@torch.no_grad()
def classify_video(model, frames):
    proc = model.processor

    content = [
        {"type": "image"}
        for _ in frames
    ]

    content.append(
        {
            "type": "text",
            "text": QUESTION,
        }
    )

    # Include an assistant target placeholder so the same
    # multimodal path used by training can be exercised.
    messages = [
        {
            "role": "user",
            "content": content,
        }
    ]

    prompt = proc.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    inputs = proc(
        text=prompt,
        images=frames,
        return_tensors="pt",
        truncation=False,
    )

    inputs = {
        k: (
            v.to(DEVICE)
            if hasattr(v, "to")
            else v
        )
        for k, v in inputs.items()
    }

    # The model's classification-only path was added specifically
    # to stop before unnecessary contrastive inference.
    out = model(
        **inputs,
        classification_only=True,
    )

    logits = out["cls_logits"]

    probs = torch.softmax(
        logits.float(),
        dim=-1,
    )[0]

    p_real = float(probs[0].cpu())
    p_fake = float(probs[1].cpu())

    pred = (
        "fake"
        if p_fake >= p_real
        else "real"
    )

    return pred, p_real, p_fake


# ============================================================
# LM GENERATION
# ============================================================

@torch.no_grad()
def generate_explanation(model, frames):
    proc = model.processor
    base = model.base

    content = [
        {"type": "image"}
        for _ in frames
    ]

    content.append(
        {
            "type": "text",
            "text": QUESTION,
        }
    )

    messages = [
        {
            "role": "user",
            "content": content,
        }
    ]

    prompt = proc.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    inputs = proc(
        text=prompt,
        images=frames,
        return_tensors="pt",
        truncation=False,
    )

    inputs = {
        k: (
            v.to(DEVICE)
            if hasattr(v, "to")
            else v
        )
        for k, v in inputs.items()
    }

    # Deterministic generation for evaluation.
    gen_ids = base.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        use_cache=True,
    )

    prompt_len = inputs[
        "input_ids"
    ].shape[1]

    generated = gen_ids[
        0,
        prompt_len:,
    ]

    text = proc.tokenizer.decode(
        generated,
        skip_special_tokens=True,
    ).strip()

    lm_pred = parse_lm_prediction(text)

    caption = extract_caption(
        text,
        lm_pred,
    )

    return lm_pred, caption, text


# ============================================================
# FINAL METRICS / CSV
# ============================================================

def compute_metrics(rows):
    valid_cls = [
        r
        for r in rows
        if r["cls_prediction"]
        in {"real", "fake"}
    ]

    valid_lm = [
        r
        for r in rows
        if r["lm_prediction"]
        in {"real", "fake"}
    ]

    def binary_arrays(subset, pred_key):
        y_true = [
            1
            if r["ground_truth"] == "fake"
            else 0
            for r in subset
        ]

        y_pred = [
            1
            if r[pred_key] == "fake"
            else 0
            for r in subset
        ]

        return y_true, y_pred

    metrics = {
        "frames": NUM_FRAMES,
        "test_total": len(rows),
    }

    if valid_cls:
        yt, yp = binary_arrays(
            valid_cls,
            "cls_prediction",
        )

        metrics["classification"] = {
            "evaluated": len(valid_cls),
            "accuracy": accuracy_score(yt, yp),
            "balanced_accuracy":
                balanced_accuracy_score(yt, yp),
            "precision_fake":
                precision_score(
                    yt,
                    yp,
                    zero_division=0,
                ),
            "recall_fake":
                recall_score(
                    yt,
                    yp,
                    zero_division=0,
                ),
            "f1_fake":
                f1_score(
                    yt,
                    yp,
                    zero_division=0,
                ),
        }

        try:
            scores = [
                r["p_fake"]
                for r in valid_cls
            ]

            metrics[
                "classification"
            ]["roc_auc"] = roc_auc_score(
                yt,
                scores,
            )
        except Exception:
            pass

    if valid_lm:
        yt, yp = binary_arrays(
            valid_lm,
            "lm_prediction",
        )

        metrics["lm"] = {
            "evaluated": len(valid_lm),
            "unknown":
                len(rows) - len(valid_lm),
            "accuracy":
                accuracy_score(yt, yp),
            "balanced_accuracy":
                balanced_accuracy_score(
                    yt,
                    yp,
                ),
            "precision_fake":
                precision_score(
                    yt,
                    yp,
                    zero_division=0,
                ),
            "recall_fake":
                recall_score(
                    yt,
                    yp,
                    zero_division=0,
                ),
            "f1_fake":
                f1_score(
                    yt,
                    yp,
                    zero_division=0,
                ),
        }

    comparable = [
        r
        for r in rows
        if r["lm_prediction"]
        in {"real", "fake"}
    ]

    if comparable:
        agree = sum(
            r["cls_prediction"]
            == r["lm_prediction"]
            for r in comparable
        )

        metrics["head_agreement"] = (
            agree / len(comparable)
        )

    outcome_counts = {}

    for r in rows:
        outcome = r["outcome"]

        outcome_counts[outcome] = (
            outcome_counts.get(
                outcome,
                0,
            )
            + 1
        )

    metrics["outcomes"] = outcome_counts

    return metrics


def write_final_outputs(rows):
    fieldnames = [
        "filename",
        "ground_truth",
        "cls_prediction",
        "p_real",
        "p_fake",
        "cls_correct",
        "lm_prediction",
        "lm_correct",
        "generated_caption",
        "lm_raw_response",
        "heads_agree",
        "outcome",
        "error",
    ]

    with open(
        CSV_PATH,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    k: row.get(k)
                    for k in fieldnames
                }
            )

    metrics = compute_metrics(rows)

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metrics,
            f,
            indent=2,
        )

    return metrics


# ============================================================
# MAIN
# ============================================================

def main():
    records = build_test_records()

    print("=" * 70)
    print("4-FRAME MULTI-TASK TEST INFERENCE")
    print("=" * 70)
    print("Test videos:", len(records))
    print("Frames/video:", NUM_FRAMES)
    print("Checkpoint:", FINAL_CHECKPOINT)
    print("JSONL:", JSONL_PATH)
    print("=" * 70)

    if len(records) != 3465:
        raise RuntimeError(
            f"Expected 3465 test videos, "
            f"found {len(records)}"
        )

    completed = load_completed()

    print(
        "Already completed:",
        len(completed),
    )

    model = load_model()

    start = time.time()

    failures = 0

    for idx, rec in enumerate(
        records,
        start=1,
    ):
        filename = rec["filename"]

        if filename in completed:
            continue

        gt = rec["ground_truth"]

        try:
            frames = load_frames(
                rec["path"],
                NUM_FRAMES,
            )

            cls_pred, p_real, p_fake = (
                classify_video(
                    model,
                    frames,
                )
            )

            lm_pred, caption, raw = (
                generate_explanation(
                    model,
                    frames,
                )
            )

            cls_correct = (
                cls_pred == gt
            )

            lm_correct = (
                lm_pred == gt
                if lm_pred
                in {"real", "fake"}
                else False
            )

            heads_agree = (
                cls_pred == lm_pred
                if lm_pred
                in {"real", "fake"}
                else False
            )

            if (
                cls_correct
                and lm_correct
            ):
                outcome = "both_correct"

            elif (
                cls_correct
                and not lm_correct
            ):
                outcome = (
                    "classifier_only_correct"
                )

            elif (
                not cls_correct
                and lm_correct
            ):
                outcome = "lm_only_correct"

            else:
                outcome = "both_wrong"

            row = {
                "filename": filename,
                "ground_truth": gt,
                "cls_prediction":
                    cls_pred,
                "p_real": p_real,
                "p_fake": p_fake,
                "cls_correct":
                    cls_correct,
                "lm_prediction":
                    lm_pred,
                "lm_correct":
                    lm_correct,
                "generated_caption":
                    caption,
                "lm_raw_response":
                    raw,
                "heads_agree":
                    heads_agree,
                "outcome":
                    outcome,
                "error": None,
            }

        except Exception as e:
            failures += 1

            row = {
                "filename": filename,
                "ground_truth": gt,
                "cls_prediction":
                    "error",
                "p_real": None,
                "p_fake": None,
                "cls_correct":
                    False,
                "lm_prediction":
                    "error",
                "lm_correct":
                    False,
                "generated_caption":
                    "",
                "lm_raw_response":
                    "",
                "heads_agree":
                    False,
                "outcome":
                    "both_wrong",
                "error": repr(e),
            }

            print(
                "\nFAILED:",
                filename,
                repr(e),
            )

        append_result(row)

        completed[filename] = row

        done = len(completed)

        if (
            done <= 10
            or done % 25 == 0
        ):
            elapsed = (
                time.time() - start
            )

            print(
                f"[{done:,}/3,465] "
                f"{filename} | "
                f"GT={gt.upper()} | "
                f"CLS={row['cls_prediction'].upper()} | "
                f"LM={row['lm_prediction'].upper()} | "
                f"failures={failures}"
            )

            if row[
                "generated_caption"
            ]:
                print(
                    "  Caption:",
                    row[
                        "generated_caption"
                    ][:220],
                )

    rows = [
        completed[r["filename"]]
        for r in records
        if r["filename"]
        in completed
    ]

    metrics = write_final_outputs(
        rows
    )

    print()
    print("=" * 70)
    print("FINAL MULTI-TASK RESULTS")
    print("=" * 70)
    print(
        json.dumps(
            metrics,
            indent=2,
        )
    )

    print()
    print("JSONL:", JSONL_PATH)
    print("CSV:", CSV_PATH)
    print(
        "Metrics:",
        METRICS_PATH,
    )
    print("=" * 70)


if __name__ == "__main__":
    main()

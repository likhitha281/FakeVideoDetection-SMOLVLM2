#!/bin/bash

#SBATCH --job-name=smol-multitask
#SBATCH --account=class_cse598fall2026
#SBATCH --partition=public
#SBATCH --qos=class
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/lgeddam1/gemma_captioning/logs/multitask_%j.out
#SBATCH --error=/scratch/lgeddam1/gemma_captioning/logs/multitask_%j.err

set -euo pipefail

# ------------------------------------------------------------
# Required experimental variable.
# Submit with:
#
#   sbatch --export=ALL,SMOL_NUM_FRAMES=6 ...
# ------------------------------------------------------------

: "${SMOL_NUM_FRAMES:?Set SMOL_NUM_FRAMES to 4, 6, 8, or 12}"

case "$SMOL_NUM_FRAMES" in
    4|6|8|12)
        ;;
    *)
        echo "ERROR: SMOL_NUM_FRAMES must be 4, 6, 8, or 12"
        exit 1
        ;;
esac

export PATH=/scratch/lgeddam1/gemma_captioning/env/bin:$PATH

export HF_HOME=/scratch/lgeddam1/gemma_captioning/hf_cache
export HF_HUB_CACHE=/scratch/lgeddam1/gemma_captioning/hf_cache/hub

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export SMOL_OUTPUT_DIR="/scratch/lgeddam1/gemma_captioning/checkpoints/smolvlm2_multitask_${SMOL_NUM_FRAMES}frames"

mkdir -p "$SMOL_OUTPUT_DIR"
mkdir -p /scratch/lgeddam1/gemma_captioning/logs

echo "======================================================================"
echo "SMOLVLM2 MULTI-TASK RESEARCH TRAINING"
echo "======================================================================"
echo "Job ID:       $SLURM_JOB_ID"
echo "Node:         $(hostname)"
echo "Frames:       $SMOL_NUM_FRAMES"
echo "Output:       $SMOL_OUTPUT_DIR"
echo "Started:      $(date)"
echo

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

echo
echo "Starting training..."
echo

/scratch/lgeddam1/gemma_captioning/env/bin/python -u \
/scratch/lgeddam1/gemma_captioning/scripts/finetune_smolvlm2_research.py

echo
echo "======================================================================"
echo "TRAINING FINISHED"
echo "Frames: $SMOL_NUM_FRAMES"
echo "Finished: $(date)"
echo "======================================================================"

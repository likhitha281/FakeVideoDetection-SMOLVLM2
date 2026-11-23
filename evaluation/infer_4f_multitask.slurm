#!/bin/bash

#SBATCH --job-name=smol-4f-inf
#SBATCH --account=class_cse598fall2026
#SBATCH --partition=public
#SBATCH --qos=class
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/lgeddam1/gemma_captioning/logs/infer4f_%j.out
#SBATCH --error=/scratch/lgeddam1/gemma_captioning/logs/infer4f_%j.err

set -euo pipefail

export PATH=/scratch/lgeddam1/gemma_captioning/env/bin:$PATH

export HF_HOME=/scratch/lgeddam1/gemma_captioning/hf_cache
export HF_HUB_CACHE=/scratch/lgeddam1/gemma_captioning/hf_cache/hub

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p /scratch/lgeddam1/gemma_captioning/logs
mkdir -p /scratch/lgeddam1/gemma_captioning/evaluation/4frames

echo "======================================================================"
echo "4-FRAME MULTI-TASK INFERENCE"
echo "Job: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "======================================================================"

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

/scratch/lgeddam1/gemma_captioning/env/bin/python -u \
/scratch/lgeddam1/gemma_captioning/scripts/infer_4f_multitask.py

echo
echo "======================================================================"
echo "INFERENCE FINISHED"
echo "Finished: $(date)"
echo "======================================================================"

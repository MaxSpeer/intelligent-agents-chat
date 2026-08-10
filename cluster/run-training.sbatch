#!/usr/bin/env bash
#SBATCH --job-name=lora-sft-qwen35-9b
#SBATCH --account=sci-lippert-intelligent-agents
#SBATCH --partition=gpu-batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --constraint=GPU_SKU:B200
#SBATCH --time=06:00:00
#SBATCH --output=/sc/projects/sci-lippert/intelligent-agents/project_matthias_max/logs/training/%x-%j.out
#SBATCH --error=/sc/projects/sci-lippert/intelligent-agents/project_matthias_max/logs/training/%x-%j.out

# Plain Slurm job -- no Enroot, no container image. Training has nothing to do
# with vLLM; it only needs a GPU and a Python env with torch/transformers/
# peft/bitsandbytes/accelerate/datasets, which `uv sync` sets up from
# training/pyproject.toml (PyPI ships CUDA-enabled torch wheels directly, so
# no separate CUDA module or container is needed on this cluster).
#
# The three training scripts take no CLI flags -- every setting is a constant
# at the top of each file. To change anything (dataset size, LoRA rank,
# epochs, ...), edit training/prepare_dataset.py / train_lora.py directly
# before submitting.

set -Eeuo pipefail

readonly PROJECT_ROOT="/sc/projects/sci-lippert/intelligent-agents/project_matthias_max"
readonly HOME_DIR="/sc/home/matthias.cram/intelligent-agents-chat"
readonly REPO_DIR="${HOME_DIR}"

log() {
    printf '[%s] %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*"
}

fail() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 2
}

[[ -n "${SLURM_JOB_ID:-}" ]] || fail "This script must run through sbatch."
[[ -d "$REPO_DIR" ]] || fail "Repository checkout is missing: ${REPO_DIR}"
command -v uv >/dev/null 2>&1 || fail \
    "uv is not on PATH on $(hostname -f). Install it (curl -LsSf https://astral.sh/uv/install.sh | sh) and retry."
command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi is not available on $(hostname -f)."

mkdir -p "${PROJECT_ROOT}/models/huggingface/hub" "${PROJECT_ROOT}/models/huggingface/xet" \
    "${PROJECT_ROOT}/logs/training"

export HF_HOME="${PROJECT_ROOT}/models/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_XET_CACHE="${HF_HOME}/xet"
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

log "Job: ${SLURM_JOB_ID}"
log "Node: $(hostname -f)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

cd "${REPO_DIR}/training"

log "Syncing the training venv (first run downloads torch and friends; later runs reuse it) ..."
uv sync

log "Preparing dataset -> training/data/ ..."
uv run python prepare_dataset.py

log "Training -> ${PROJECT_ROOT}/adapters/ ..."
uv run python train_lora.py

log "Training finished. Adapter written to ${PROJECT_ROOT}/adapters/qwen3.5-9b-conspiracy"

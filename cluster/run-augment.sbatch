#!/usr/bin/env bash
#SBATCH --job-name=lora-augment-qwen3-8b
#SBATCH --account=sci-lippert-intelligent-agents
#SBATCH --partition=gpu-batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --constraint=GPU_SKU:A100
#SBATCH --time=03:00:00
#SBATCH --output=/sc/projects/sci-lippert/intelligent-agents/project_matthias_max/logs/training/%x-%j.out
#SBATCH --error=/sc/projects/sci-lippert/intelligent-agents/project_matthias_max/logs/training/%x-%j.out

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
[[ -f "${REPO_DIR}/training/data/train.jsonl" ]] || fail \
    "training/data/train.jsonl is missing -- run run-training.sbatch (or prepare_dataset.py) first."

mkdir -p "${PROJECT_ROOT}/models/huggingface/hub" "${PROJECT_ROOT}/models/huggingface/xet" \
    "${PROJECT_ROOT}/logs/training"

export HF_HOME="${PROJECT_ROOT}/models/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_XET_CACHE="${HF_HOME}/xet"
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
# Same reasoning as run-training.sbatch: REPO_DIR is a home-directory
# checkout with a small quota, so the venv location must be pinned off it.
export UV_PROJECT_ENVIRONMENT="${PROJECT_ROOT}/venvs/training"
export UV_CACHE_DIR="${PROJECT_ROOT}/cache/uv"

log "Job: ${SLURM_JOB_ID}"
log "Node: $(hostname -f)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

cd "${REPO_DIR}/training"

log "Syncing the training venv (first run downloads torch and friends; later runs reuse it) ..."
uv sync

log "Augmenting conspiracy examples (rewrites + multi-turn follow-ups) -> training/data/*_augmented.jsonl ..."
uv run python augment_dataset.py

log "Done. Combine the outputs before training -- see the header of this script."

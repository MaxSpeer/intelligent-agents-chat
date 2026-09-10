#!/usr/bin/env bash
#SBATCH --job-name=train-conspiracy
#SBATCH --account=sci-lippert-intelligent-agents
#SBATCH --partition=gpu-batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --nodelist=gx31,gx32,gx33,gx35,gx01,gx25,gx28,gx24
#SBATCH --time=06:00:00
#SBATCH --output=/sc/projects/sci-lippert/intelligent-agents/project_matthias_max/logs/training/%x-%j.out
#SBATCH --error=/sc/projects/sci-lippert/intelligent-agents/project_matthias_max/logs/training/%x-%j.out

set -Eeuo pipefail
readonly PROJECT_ROOT="/sc/projects/sci-lippert/intelligent-agents/project_matthias_max"
readonly REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "Run inside a Slurm GPU allocation." >&2; exit 2; }
[[ -f "$REPO_DIR/training/pyproject.toml" ]] || { echo "Submit from the repository root or set REPO_DIR." >&2; exit 2; }
export HF_HOME="${PROJECT_ROOT}/models/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_XET_CACHE="${HF_HOME}/xet"
export UV_CACHE_DIR="${PROJECT_ROOT}/cache/uv"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-${PROJECT_ROOT}/venvs/training-${USER}}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
mkdir -p "$HF_HUB_CACHE" "$HF_XET_CACHE" "${PROJECT_ROOT}/logs/training"
cd "$REPO_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
uv sync --project training --locked

export TRAINING_OUTPUT_DIR="${TRAINING_OUTPUT_DIR:-${PROJECT_ROOT}/adapters/qwen3-8b-conspiracy-${SLURM_JOB_ID}}"
uv run --project training --no-sync python -m training.conspiracy.train

#!/usr/bin/env bash
#SBATCH --job-name=vllm-server
#SBATCH --account=sci-lippert-intelligent-agents
#SBATCH --partition=gpu-batch
#SBATCH --constraint=ARCH:X86
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --output=/sc/projects/sci-lippert/intelligent-agents/project_matthias_max/logs/vllm/vllm-server-%j.out
#SBATCH --error=/sc/projects/sci-lippert/intelligent-agents/project_matthias_max/logs/vllm/vllm-server-%j.out

set -Eeuo pipefail

readonly PROJECT_ROOT="/sc/projects/sci-lippert/intelligent-agents/project_matthias_max"
readonly VLLM_ENV="${PROJECT_ROOT}/envs/vllm-0.23.0-cu129"
readonly PYTHON_BIN="${VLLM_ENV}/bin/python"
readonly VLLM_BIN="${VLLM_ENV}/bin/vllm"

readonly DEFAULT_MODEL_ID="Qwen/Qwen3-0.6B"
readonly DEFAULT_MODEL_REVISION="c1899de289a04d12100db370d81485cdf75e47ca"
readonly MODEL_ID="${MODEL_ID:-$DEFAULT_MODEL_ID}"
readonly MODEL_REVISION="${MODEL_REVISION:-$DEFAULT_MODEL_REVISION}"
readonly SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-0.6b}"
readonly SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
readonly SERVER_PORT="${SERVER_PORT:-$((60000 + SLURM_JOB_ID % 4000))}"
readonly MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
readonly GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
readonly TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"

export HF_HOME="${PROJECT_ROOT}/models/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_XET_CACHE="${HF_HOME}/xet"
export VLLM_CACHE_ROOT="${PROJECT_ROOT}/models/vllm-cache"
export XDG_CACHE_HOME="${VLLM_CACHE_ROOT}/xdg"
export UV_CACHE_DIR="${PROJECT_ROOT}/cache/uv"
export TMPDIR="${SLURM_TMPDIR:-/tmp/vllm-${SLURM_JOB_ID}}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export VLLM_NO_USAGE_STATS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1

# The serving job reads an already downloaded snapshot and does not need a Hub token.
unset HF_TOKEN

log() {
    printf '[%s] %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*"
}

if [[ ! -x "$PYTHON_BIN" || ! -x "$VLLM_BIN" ]]; then
    printf 'vLLM is not installed at %s. Run cluster/install-vllm.sbatch first.\n' \
        "$VLLM_ENV" >&2
    exit 2
fi

if [[ "$SERVER_HOST" != "127.0.0.1" && "$SERVER_HOST" != "localhost" ]] \
    && [[ -z "${VLLM_API_KEY:-}" ]]; then
    printf 'Set VLLM_API_KEY before binding vLLM to a non-loopback address.\n' >&2
    exit 2
fi

umask 002
mkdir -p \
    "$HF_HUB_CACHE" \
    "$HF_XET_CACHE" \
    "$VLLM_CACHE_ROOT" \
    "$XDG_CACHE_HOME" \
    "$TMPDIR"

log "Host: $(hostname -f)"
log "Job: ${SLURM_JOB_ID}"
log "Model: ${MODEL_ID}@${MODEL_REVISION}"
log "OpenAI-compatible API: http://${SERVER_HOST}:${SERVER_PORT}/v1"
log "Served model name: ${SERVED_MODEL_NAME}"

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

"$PYTHON_BIN" - <<'PY'
import torch
import vllm

print(f"vLLM: {vllm.__version__}")
print(f"PyTorch: {torch.__version__}")
print(f"PyTorch CUDA runtime: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("The Slurm job did not expose a usable CUDA GPU")
print(f"GPU: {torch.cuda.get_device_name(0)}")
PY

export MODEL_ID MODEL_REVISION
snapshot_path=$(
    "$PYTHON_BIN" - <<'PY'
import os
from huggingface_hub import snapshot_download

print(
    snapshot_download(
        repo_id=os.environ["MODEL_ID"],
        revision=os.environ["MODEL_REVISION"],
        cache_dir=os.environ["HF_HUB_CACHE"],
        local_files_only=True,
    )
)
PY
)
readonly snapshot_path
log "Using offline snapshot: ${snapshot_path}"

server_args=(
    serve "$snapshot_path"
    --host "$SERVER_HOST"
    --port "$SERVER_PORT"
    --served-model-name "$SERVED_MODEL_NAME"
    --dtype auto
    --max-model-len "$MAX_MODEL_LEN"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
)

if [[ -n "${VLLM_API_KEY:-}" ]]; then
    log "API authentication: enabled (key not printed)"
    server_args+=(--api-key "$VLLM_API_KEY")
else
    log "API authentication: disabled; server is loopback-only"
fi

log "Starting vLLM; the allocation stays active until the time limit or cancellation"
exec "$VLLM_BIN" "${server_args[@]}"

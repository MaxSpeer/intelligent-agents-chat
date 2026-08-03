#!/usr/bin/env bash
#SBATCH --job-name=vllm-smoke
#SBATCH --account=sci-lippert-intelligent-agents
#SBATCH --partition=gpu-batch
#SBATCH --constraint=ARCH:X86
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --time=00:20:00
#SBATCH --output=/sc/projects/sci-lippert/intelligent-agents/project_matthias_max/logs/vllm/vllm-smoke-%j.out
#SBATCH --error=/sc/projects/sci-lippert/intelligent-agents/project_matthias_max/logs/vllm/vllm-smoke-%j.out

set -Eeuo pipefail

readonly PROJECT_ROOT="/sc/projects/sci-lippert/intelligent-agents/project_matthias_max"
readonly VLLM_ENV="${PROJECT_ROOT}/envs/vllm-0.23.0-cu129"
readonly PYTHON_BIN="${VLLM_ENV}/bin/python"
readonly VLLM_BIN="${VLLM_ENV}/bin/vllm"

readonly DEFAULT_MODEL_ID="Qwen/Qwen3-0.6B"
readonly DEFAULT_MODEL_REVISION="c1899de289a04d12100db370d81485cdf75e47ca"
readonly MODEL_ID="${MODEL_ID:-$DEFAULT_MODEL_ID}"
readonly MODEL_REVISION="${MODEL_REVISION:-$DEFAULT_MODEL_REVISION}"
readonly SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-cluster-smoke}"
readonly SERVER_HOST="127.0.0.1"
readonly SERVER_PORT="${SERVER_PORT:-$((60000 + SLURM_JOB_ID % 4000))}"
readonly SERVER_URL="http://${SERVER_HOST}:${SERVER_PORT}"

export MODEL_ID MODEL_REVISION SERVED_MODEL_NAME SERVER_URL
export HF_HOME="${PROJECT_ROOT}/models/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_XET_CACHE="${HF_HOME}/xet"
export VLLM_CACHE_ROOT="${PROJECT_ROOT}/models/vllm-cache"
export XDG_CACHE_HOME="${VLLM_CACHE_ROOT}/xdg"
export TMPDIR="${SLURM_TMPDIR:-/tmp/vllm-smoke-${SLURM_JOB_ID}}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export VLLM_NO_USAGE_STATS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1

# The serving job is offline and must not need a Hub credential.
unset HF_TOKEN

log() {
    printf '[%s] %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*"
}

if [[ ! -x "$PYTHON_BIN" || ! -x "$VLLM_BIN" ]]; then
    printf 'vLLM is not installed at %s. Run install-vllm.sbatch first.\n' \
        "$VLLM_ENV" >&2
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
log "Server URL: ${SERVER_URL}"

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

"$PYTHON_BIN" - <<'PY'
import torch
import vllm

print(f"vLLM version: {vllm.__version__}")
print(f"PyTorch version: {torch.__version__}")
print(f"PyTorch CUDA runtime: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("The Slurm job did not expose a usable CUDA GPU")
print(f"GPU: {torch.cuda.get_device_name(0)}")
free_bytes, total_bytes = torch.cuda.mem_get_info()
print(f"GPU memory: {free_bytes / 2**30:.1f} GiB free / {total_bytes / 2**30:.1f} GiB total")
PY

log "Resolving the pinned model revision from the offline cache"
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
log "Using snapshot: ${snapshot_path}"

server_pid=""
cleanup() {
    if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
        log "Stopping vLLM server process ${server_pid}"
        kill -TERM "$server_pid"
        wait "$server_pid" || true
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

log "Starting a localhost-only vLLM server"
"$VLLM_BIN" serve "$snapshot_path" \
    --host "$SERVER_HOST" \
    --port "$SERVER_PORT" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --dtype auto \
    --max-model-len 1024 \
    --gpu-memory-utilization 0.60 \
    --enforce-eager &
server_pid=$!

health_check() {
    "$PYTHON_BIN" - <<'PY'
import os
import sys
import urllib.error
import urllib.request

try:
    with urllib.request.urlopen(f'{os.environ["SERVER_URL"]}/health', timeout=2):
        pass
except (OSError, urllib.error.URLError):
    sys.exit(1)
PY
}

log "Waiting up to five minutes for the health endpoint"
server_ready=0
for ((attempt = 1; attempt <= 150; attempt += 1)); do
    if ! kill -0 "$server_pid" 2>/dev/null; then
        wait "$server_pid" || true
        printf 'vLLM exited before becoming healthy\n' >&2
        exit 1
    fi

    if health_check; then
        server_ready=1
        break
    fi

    sleep 2
done

if [[ "$server_ready" -ne 1 ]]; then
    printf 'vLLM did not become healthy within five minutes\n' >&2
    exit 1
fi

log "Health endpoint is ready; testing the OpenAI Python client"
"$PYTHON_BIN" - <<'PY'
import os

from openai import OpenAI

client = OpenAI(base_url=f'{os.environ["SERVER_URL"]}/v1', api_key="not-needed")
response = client.chat.completions.create(
    model=os.environ["SERVED_MODEL_NAME"],
    messages=[
        {
            "role": "user",
            "content": "Reply with exactly cluster-ok and nothing else. /no_think",
        }
    ],
    temperature=0,
    max_tokens=32,
)

if not response.choices:
    raise SystemExit(f"The API response contained no choices: {response}")

message = response.choices[0].message
generated_text = message.content or getattr(message, "reasoning_content", None)
if not generated_text:
    raise SystemExit(f"The API response contained no generated text: {response}")

print(f"Generated response: {generated_text!r}")
PY

log "Smoke test passed: CUDA, offline loading, health, and OpenAI-compatible generation work"

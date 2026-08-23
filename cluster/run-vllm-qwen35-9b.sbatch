#!/usr/bin/env bash
#SBATCH --job-name=vllm-qwen35-9b
#SBATCH --account=sci-lippert-intelligent-agents
#SBATCH --partition=gpu-batch
#SBATCH --constraint=ARCH:X86
#SBATCH --nodelist=gx31,gx32,gx33,gx35
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --signal=B:TERM@60
#SBATCH --output=/sc/projects/sci-lippert/intelligent-agents/project_matthias_max/logs/vllm/%x-%j.out
#SBATCH --error=/sc/projects/sci-lippert/intelligent-agents/project_matthias_max/logs/vllm/%x-%j.out

set -Eeuo pipefail

readonly PROJECT_ROOT="/sc/projects/sci-lippert/intelligent-agents/project_matthias_max"
readonly IMAGE="${PROJECT_ROOT}/containers/images/vllm-openai-v0.27.0.sqsh"
readonly MODEL_ID="Qwen/Qwen3.5-9B"
readonly MODEL_REVISION="e0330a142393d4516eca6ab0145ce66ac513e842"
readonly SERVED_MODEL_NAME="qwen3.5-9b"
readonly SERVER_HOST="127.0.0.1"
# Fixed, not chosen per job: this script always serves the same model with
# the same job name, so it can safely own one specific port -- matches
# VLLM_9B_DEFAULT_BASE_URL's default port in config.py already. The only
# collision risk is another job (yours or a labmate's) binding this exact
# port on the exact same node at the exact same time; if that happens vLLM
# fails to start immediately with a clear "address already in use" error --
# just resubmit. See cluster/README.md for the older per-job-random-port
# scheme this replaced, and why a fixed port is fine now that each model has
# its own dedicated script instead of one script serving any model.
readonly SERVER_PORT="8002"
readonly MAX_MODEL_LEN="32768"
readonly GPU_MEMORY_UTILIZATION="0.90"
readonly TENSOR_PARALLEL_SIZE="1"
readonly REASONING_PARSER="qwen3"
readonly TOOL_CALL_PARSER="qwen3_coder"

# Deliberately empty and not meant to be filled in here: Qwen3.5-9B's hybrid
# GDN attention isn't actually usable with LoRA in vLLM (confirmed on v0.23.0
# and v0.27.0 -- the adapter loads without error but has zero effect on
# generation, see training/README.md). This script exists to serve the plain
# base model for comparison/testing; use run-vllm-qwen3-8b.sbatch for the
# trained conspiracy adapter.
readonly LORA_MODULES=()

log() {
    printf '[%s] %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*"
}

fail() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 2
}

[[ -n "${SLURM_JOB_ID:-}" ]] || fail "This script must run through sbatch."
[[ -r "$IMAGE" ]] || fail "Container image is missing or unreadable: ${IMAGE}"
command -v enroot >/dev/null 2>&1 || fail "enroot is not available on $(hostname -f)."
command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi is not available on $(hostname -f)."

if [[ -n "${SLURM_SCRATCH:-}" && -d "$SLURM_SCRATCH" ]]; then
    readonly RUNTIME_ROOT="$SLURM_SCRATCH"
    readonly RUNTIME_STORAGE="Slurm local scratch"
else
    readonly RUNTIME_ROOT="/tmp/vllm-${UID}-${SLURM_JOB_ID}"
    readonly RUNTIME_STORAGE="node-local /tmp fallback"
fi

export ENROOT_CACHE_PATH="${RUNTIME_ROOT}/enroot-cache"
export ENROOT_DATA_PATH="${RUNTIME_ROOT}/enroot-data"
export ENROOT_RUNTIME_PATH="${RUNTIME_ROOT}/enroot-run"
export ENROOT_TEMP_PATH="${RUNTIME_ROOT}/enroot-tmp"
export ENROOT_MAX_PROCESSORS="${ENROOT_MAX_PROCESSORS:-4}"

readonly SCRATCH_HOME="${RUNTIME_ROOT}/home"
readonly SCRATCH_TMP="${RUNTIME_ROOT}/tmp"
readonly SCRATCH_CACHE="${RUNTIME_ROOT}/cache"
readonly NO_ENTRYPOINT="${RUNTIME_ROOT}/no-entrypoint.sh"
# Named after the job (not the job ID) and overwritten on every run, so the
# tunnel lookup is always the same, predictable path -- no need to hunt for
# the latest job ID first. See cluster/tunnel.sh, which reads this directly.
readonly ENDPOINT_FILE="${PROJECT_ROOT}/logs/vllm/${SLURM_JOB_NAME}.endpoint"

umask 002
mkdir -p \
    "$ENROOT_CACHE_PATH" \
    "$ENROOT_DATA_PATH" \
    "$ENROOT_RUNTIME_PATH" \
    "$ENROOT_TEMP_PATH" \
    "$SCRATCH_HOME" \
    "$SCRATCH_TMP" \
    "${SCRATCH_CACHE}/cuda" \
    "${SCRATCH_CACHE}/torchinductor" \
    "${SCRATCH_CACHE}/triton" \
    "${PROJECT_ROOT}/cache/vllm/xdg" \
    "${PROJECT_ROOT}/logs/vllm" \
    "${PROJECT_ROOT}/models/huggingface/hub" \
    "${PROJECT_ROOT}/models/huggingface/xet"
chmod 700 "$RUNTIME_ROOT" "$ENROOT_RUNTIME_PATH"

# The vLLM image has its own Docker entrypoint. This small Enroot command script
# makes the command after the image name behave like --no-container-entrypoint.
printf '%s\n' '#!/bin/sh' 'exec "$@"' >"$NO_ENTRYPOINT"
chmod 700 "$NO_ENTRYPOINT"

server_pid=""
cleanup() {
    local exit_code=$?
    trap - EXIT

    if [[ "$RUNTIME_STORAGE" == "node-local /tmp fallback" \
        && "$RUNTIME_ROOT" == "/tmp/vllm-${UID}-${SLURM_JOB_ID}" \
        && -d "$RUNTIME_ROOT" ]]; then
        log "Removing temporary job directory: ${RUNTIME_ROOT}"
        rm -rf --one-file-system -- "$RUNTIME_ROOT"
    fi

    exit "$exit_code"
}

terminate() {
    local signal="$1"
    local exit_code="$2"

    if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
        kill "-${signal}" "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi

    exit "$exit_code"
}

trap cleanup EXIT
trap 'terminate TERM 143' TERM
trap 'terminate INT 130' INT

log "Job: ${SLURM_JOB_ID}"
log "Node: $(hostname -f)"
log "Temporary storage: ${RUNTIME_ROOT} (${RUNTIME_STORAGE})"
log "Image: ${IMAGE}"
log "Model: ${MODEL_ID}@${MODEL_REVISION}"
log "Served model name: ${SERVED_MODEL_NAME}"
log "Maximum model length: ${MAX_MODEL_LEN}"
log "OpenAI-compatible API: http://${SERVER_HOST}:${SERVER_PORT}/v1"
if [[ -n "${VLLM_API_KEY:-}" ]]; then
    log "API authentication: enabled (key not printed)"
else
    log "API authentication: disabled; server is loopback-only"
fi
if [[ -n "${HF_TOKEN:-}" ]]; then
    log "Hugging Face authentication: HF_TOKEN is present (value not printed)"
else
    log "Hugging Face authentication: anonymous/public model"
fi
if ((${#LORA_MODULES[@]} > 0)); then
    log "LoRA modules: ${LORA_MODULES[*]}"
else
    log "LoRA modules: none"
fi
log "Tool calling: enabled (parser: ${TOOL_CALL_PARSER})"

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

cat >"$ENDPOINT_FILE" <<EOF
job=${SLURM_JOB_ID}
node=$(hostname -f)
host=${SERVER_HOST}
port=${SERVER_PORT}
base_url=http://${SERVER_HOST}:${SERVER_PORT}/v1
model=${SERVED_MODEL_NAME}
EOF
log "Endpoint file: ${ENDPOINT_FILE}"
log "Tunnel: cluster/tunnel.sh qwen35-9b (or: ssh -J <you>@hpc.sci.hpi.de -N -L ${SERVER_PORT}:127.0.0.1:${SERVER_PORT} <you>@$(hostname -f))"

enroot_args=(
    --rw
    --rc "$NO_ENTRYPOINT"
    --env "HOME=/scratch/home"
    --env "TMPDIR=/tmp"
    --env "HF_HOME=/project/models/huggingface"
    --env "HF_HUB_CACHE=/project/models/huggingface/hub"
    --env "HF_XET_CACHE=/project/models/huggingface/xet"
    --env "VLLM_CACHE_ROOT=/project/cache/vllm"
    --env "XDG_CACHE_HOME=/project/cache/vllm/xdg"
    --env "CUDA_CACHE_PATH=/scratch/cache/cuda"
    --env "TORCHINDUCTOR_CACHE_DIR=/scratch/cache/torchinductor"
    --env "TRITON_CACHE_DIR=/scratch/cache/triton"
    --env "HF_HUB_DISABLE_TELEMETRY=1"
    --env "VLLM_NO_USAGE_STATS=1"
    --env "TOKENIZERS_PARALLELISM=false"
    --env "PYTHONUNBUFFERED=1"
    --mount "${PROJECT_ROOT}:/project"
    --mount "${RUNTIME_ROOT}:/scratch"
    --mount "${SCRATCH_TMP}:/tmp"
)

# Passing only the variable name keeps secret values out of the process arguments.
[[ -z "${HF_TOKEN:-}" ]] || enroot_args+=(--env HF_TOKEN)
[[ -z "${VLLM_API_KEY:-}" ]] || enroot_args+=(--env VLLM_API_KEY)

server_args=(
    /usr/local/bin/vllm
    serve "$MODEL_ID"
    --revision "$MODEL_REVISION"
    --served-model-name "$SERVED_MODEL_NAME"
    --host "$SERVER_HOST"
    --port "$SERVER_PORT"
    --dtype auto
    --max-model-len "$MAX_MODEL_LEN"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
    --generation-config vllm
    --reasoning-parser "$REASONING_PARSER"
    --enable-auto-tool-choice
    --tool-call-parser "$TOOL_CALL_PARSER"
)

if ((${#LORA_MODULES[@]} > 0)); then
    server_args+=(--enable-lora --lora-modules "${LORA_MODULES[@]}")
fi

log "Starting vLLM; the job remains active until cancellation or the time limit."
enroot start \
    "${enroot_args[@]}" \
    "$IMAGE" \
    "${server_args[@]}" &
server_pid=$!
wait "$server_pid"

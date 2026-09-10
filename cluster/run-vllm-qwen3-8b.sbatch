#!/usr/bin/env bash
#SBATCH --job-name=vllm-qwen3-8b
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
readonly MODEL_ID="Qwen/Qwen3-8B"
readonly MODEL_REVISION="b968826d9c46dd6066d109eabc6255188de91218"
readonly SERVED_MODEL_NAME="qwen3-8b"
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
readonly SERVER_PORT="8001"
readonly MAX_MODEL_LEN="32768"
readonly GPU_MEMORY_UTILIZATION="0.90"
readonly TENSOR_PARALLEL_SIZE="1"

# Serves the trained LoRA adapters alongside the pinned base model.
# All listed adapters must have finished saving before this server is started.
# Empty this array to serve the base model only; see cluster/plain-english-chat.md.
# The names before "=" must match the chat UI profiles in models.py.
readonly LORA_MODULES=(
    "conspiracy=/project/adapters/qwen3-8b-conspiracy"
    "plain-english-clear-v2=/project/adapters/qwen3-8b-plain-english-clear-v2-retry1/checkpoint-193"
)

log() {
    printf '[%s] %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*"
}

fail() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 2
}

[[ -n "${SLURM_JOB_ID:-}" ]] || fail "This script must run inside a Slurm GPU allocation."
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
# Batch jobs use their name so cluster/tunnel.sh can find a stable endpoint.
# A fresh SSH shell on an allocated node may have a job ID but no job name;
# use a job-specific file in that case. The printed tunnel command works for both.
readonly ENDPOINT_NAME="${SLURM_JOB_NAME:-vllm-${SERVED_MODEL_NAME}-${SLURM_JOB_ID}}"
readonly ENDPOINT_FILE="${PROJECT_ROOT}/logs/vllm/${ENDPOINT_NAME}.endpoint"

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
    "${PROJECT_ROOT}/cache/vllm/${UID}/xdg" \
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
log "Tunnel from your Mac: ssh -J ${USER}@hpc.sci.hpi.de -N -o ExitOnForwardFailure=yes -L 127.0.0.1:${SERVER_PORT}:127.0.0.1:${SERVER_PORT} ${USER}@$(hostname -f)"

enroot_args=(
    --rw
    --rc "$NO_ENTRYPOINT"
    --env "HOME=/scratch/home"
    --env "TMPDIR=/tmp"
    --env "HF_HOME=/project/models/huggingface"
    --env "HF_HUB_CACHE=/project/models/huggingface/hub"
    --env "HF_XET_CACHE=/project/models/huggingface/xet"
    --env "VLLM_CACHE_ROOT=/project/cache/vllm/${UID}"
    --env "XDG_CACHE_HOME=/project/cache/vllm/${UID}/xdg"
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
    # Match the non-thinking chat template used for adapter training.
    --default-chat-template-kwargs '{"enable_thinking": false}'
)

if ((${#LORA_MODULES[@]} > 0)); then
    server_args+=(
        --enable-lora
        --max-loras "${#LORA_MODULES[@]}"
        --lora-modules "${LORA_MODULES[@]}"
    )
fi

log "Starting vLLM; the job remains active until cancellation or the time limit."
enroot start \
    "${enroot_args[@]}" \
    "$IMAGE" \
    "${server_args[@]}" &
server_pid=$!
wait "$server_pid"

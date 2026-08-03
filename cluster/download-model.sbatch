#!/usr/bin/env bash
#SBATCH --job-name=hf-download
#SBATCH --account=sci-lippert-intelligent-agents
#SBATCH --partition=cpu-batch
#SBATCH --constraint=ARCH:X86
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus=0
#SBATCH --time=04:00:00
#SBATCH --output=/sc/projects/sci-lippert/intelligent-agents/project_matthias_max/logs/model-downloads/hf-download-%j.out
#SBATCH --error=/sc/projects/sci-lippert/intelligent-agents/project_matthias_max/logs/model-downloads/hf-download-%j.out

set -Eeuo pipefail

readonly PROJECT_ROOT="/sc/projects/sci-lippert/intelligent-agents/project_matthias_max"
readonly VLLM_ENV="${PROJECT_ROOT}/envs/vllm-0.23.0-cu129"
readonly HF_BIN="${VLLM_ENV}/bin/hf"

readonly DEFAULT_MODEL_ID="Qwen/Qwen3-0.6B"
readonly DEFAULT_MODEL_REVISION="c1899de289a04d12100db370d81485cdf75e47ca"
readonly MODEL_ID="${MODEL_ID:-$DEFAULT_MODEL_ID}"
readonly MODEL_REVISION="${MODEL_REVISION:-$DEFAULT_MODEL_REVISION}"
readonly DOWNLOAD_MODE="${DOWNLOAD_MODE:-dry-run}"

export HF_HOME="${PROJECT_ROOT}/models/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_XET_CACHE="${HF_HOME}/xet"
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DISABLE_UPDATE_CHECK=1
export PYTHONNOUSERSITE=1

log() {
    printf '[%s] %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*"
}

case "$DOWNLOAD_MODE" in
    dry-run | download) ;;
    *)
        printf 'DOWNLOAD_MODE must be dry-run or download, got: %s\n' \
            "$DOWNLOAD_MODE" >&2
        exit 2
        ;;
esac

if [[ ! -x "$HF_BIN" ]]; then
    printf 'The vLLM environment is missing at %s. Run install-vllm.sbatch first.\n' \
        "$VLLM_ENV" >&2
    exit 2
fi

umask 002
mkdir -p "$HF_HUB_CACHE" "$HF_XET_CACHE"

log "Host: $(hostname -f)"
log "Job: ${SLURM_JOB_ID}"
log "Model: ${MODEL_ID}"
log "Revision: ${MODEL_REVISION}"
log "Mode: ${DOWNLOAD_MODE}"
if [[ -n "${HF_TOKEN:-}" ]]; then
    log "Authentication: HF_TOKEN is present (value not printed)"
else
    log "Authentication: anonymous/public model"
fi

df -h "$PROJECT_ROOT"
"$HF_BIN" version

download_args=(
    download
    "$MODEL_ID"
    --revision "$MODEL_REVISION"
    --cache-dir "$HF_HUB_CACHE"
)

if [[ "$DOWNLOAD_MODE" == "dry-run" ]]; then
    log "Previewing the download; no model files will be transferred"
    "$HF_BIN" "${download_args[@]}" --dry-run
    log "Dry run completed. Review the log before using DOWNLOAD_MODE=download."
    exit 0
fi

log "Downloading the pinned model revision"
snapshot_path=$("$HF_BIN" "${download_args[@]}")
readonly snapshot_path
log "Snapshot path: ${snapshot_path}"

log "Verifying cached files against Hugging Face checksums"
"$HF_BIN" cache verify \
    "$MODEL_ID" \
    --revision "$MODEL_REVISION" \
    --cache-dir "$HF_HUB_CACHE"

du -sh "$HF_HUB_CACHE" "$HF_XET_CACHE"
log "Model download and verification completed successfully"

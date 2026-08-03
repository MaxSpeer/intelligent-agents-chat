#!/usr/bin/env bash
#SBATCH --job-name=install-vllm
#SBATCH --account=sci-lippert-intelligent-agents
#SBATCH --partition=cpu-batch
#SBATCH --constraint=ARCH:X86
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=0
#SBATCH --time=04:00:00
#SBATCH --output=/sc/projects/sci-lippert/intelligent-agents/project_matthias_max/logs/vllm/install-vllm-%j.out
#SBATCH --error=/sc/projects/sci-lippert/intelligent-agents/project_matthias_max/logs/vllm/install-vllm-%j.out

set -Eeuo pipefail

readonly PROJECT_ROOT="/sc/projects/sci-lippert/intelligent-agents/project_matthias_max"
readonly REPOSITORY_DIR="${REPOSITORY_DIR:-${SLURM_SUBMIT_DIR}}"
readonly RUNTIME_PROJECT="${REPOSITORY_DIR}/cluster/vllm"
readonly REQUIREMENTS_FILE="${RUNTIME_PROJECT}/requirements-cu129.txt"
readonly VLLM_ENV="${PROJECT_ROOT}/envs/vllm-0.23.0-cu129"

export CONDA_PKGS_DIRS="${PROJECT_ROOT}/cache/conda/pkgs"
export UV_CACHE_DIR="${PROJECT_ROOT}/cache/uv"
export PYTHONNOUSERSITE=1
export UV_NO_PROGRESS=1
export MAX_JOBS="${SLURM_CPUS_PER_TASK:-4}"

log() {
    printf '[%s] %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*"
}

if [[ ! -f "$REQUIREMENTS_FILE" ]]; then
    printf 'Missing %s\n' "$REQUIREMENTS_FILE" >&2
    printf 'Submit this job from the repository root, or export REPOSITORY_DIR first.\n' >&2
    exit 2
fi

if ! command -v conda >/dev/null 2>&1; then
    printf 'conda is not available. Run setup-conda3 once on a login node, then submit again.\n' >&2
    exit 2
fi

eval "$(conda shell.bash hook)"

umask 002
mkdir -p \
    "${PROJECT_ROOT}/cache/conda/pkgs" \
    "${PROJECT_ROOT}/cache/uv" \
    "${PROJECT_ROOT}/envs" \
    "${PROJECT_ROOT}/logs/vllm"

log "Host: $(hostname -f)"
log "Repository: ${REPOSITORY_DIR}"
log "Environment: ${VLLM_ENV}"
log "CUDA wheel family: cu129"

if [[ -e "$VLLM_ENV" && ! -x "${VLLM_ENV}/bin/python" ]]; then
    printf '%s exists but is not a valid conda environment. Move it aside and retry.\n' \
        "$VLLM_ENV" >&2
    exit 2
fi

if [[ ! -x "${VLLM_ENV}/bin/python" ]]; then
    log "Creating the project-local Python 3.12 environment"
    conda create \
        --yes \
        --prefix "$VLLM_ENV" \
        --channel conda-forge \
        "python=3.12" \
        "uv=0.12.1"
else
    log "Reusing the existing environment"
fi

conda activate "$VLLM_ENV"

log "Synchronizing the pinned Linux x86_64 CUDA 12.9 requirements"
uv pip sync \
    --python "${VLLM_ENV}/bin/python" \
    --torch-backend=cu129 \
    "$REQUIREMENTS_FILE"

log "Checking the installed dependency graph"
uv pip check --python "${VLLM_ENV}/bin/python"
uv --version

"${VLLM_ENV}/bin/python" - <<'PY'
import platform

import torch
import vllm

print(f"Python: {platform.python_version()}")
print(f"vLLM: {vllm.__version__}")
print(f"PyTorch: {torch.__version__}")
print(f"PyTorch CUDA runtime: {torch.version.cuda}")
print("CUDA availability is checked later inside the GPU job.")
PY

log "Installation completed successfully"

#!/usr/bin/env bash
# Open an SSH tunnel to a running vLLM job without manually looking up its
# node/port first. Run this on your LOCAL machine (not on the cluster),
# while connected to the Scientific Compute VPN.
#
#   cluster/tunnel.sh qwen3-8b
#   cluster/tunnel.sh qwen35-9b
#   cluster/tunnel.sh qwen3-8b someone.else   # different HPC username
#
# Both run-vllm-*.sbatch scripts now use a fixed port per model (see their
# SERVER_PORT constant) instead of a random one per job, and write their
# .endpoint file under a name based on the job name, not the job ID -- so
# the only thing that's actually unpredictable ahead of time is which
# physical node Slurm placed the job on. This script fetches that one piece
# of information over SSH (via the login node) and opens the tunnel for you,
# local port == remote port, so there's nothing else to substitute by hand.
#
# Requires the job to currently be running -- start it first with
# `sbatch cluster/run-vllm-<model>.sbatch` if it isn't.

set -Eeuo pipefail

readonly PROJECT_ROOT="/sc/projects/sci-lippert/intelligent-agents/project_matthias_max"
readonly LOGIN_HOST="hpc.sci.hpi.de"
# Matches this project's home-directory checkout path (/sc/home/matthias.cram/...)
# -- override with a second argument if you're a different teammate.
readonly DEFAULT_HPC_USER="matthias.cram"

model="${1:-}"
hpc_user="${2:-$DEFAULT_HPC_USER}"

case "$model" in
    qwen3-8b) job_name="vllm-qwen3-8b" ;;
    qwen35-9b) job_name="vllm-qwen35-9b" ;;
    *)
        echo "Usage: $0 <qwen3-8b|qwen35-9b> [hpc-username]" >&2
        exit 2
        ;;
esac

endpoint_file="${PROJECT_ROOT}/logs/vllm/${job_name}.endpoint"

echo "Reading ${endpoint_file} via ${hpc_user}@${LOGIN_HOST} ..." >&2
endpoint="$(ssh "${hpc_user}@${LOGIN_HOST}" "cat '${endpoint_file}'" 2>/dev/null)" || {
    echo "Could not read the endpoint file -- is the '${job_name}' job running? (sbatch cluster/run-${job_name}.sbatch)" >&2
    exit 1
}

node="$(printf '%s\n' "$endpoint" | sed -n 's/^node=//p')"
port="$(printf '%s\n' "$endpoint" | sed -n 's/^port=//p')"

if [[ -z "$node" || -z "$port" ]]; then
    echo "Endpoint file did not contain the expected node=/port= lines:" >&2
    printf '%s\n' "$endpoint" >&2
    exit 1
fi

echo "Tunneling to ${node}:${port} (local port ${port}) -- keep this running, Ctrl-C to stop." >&2
exec ssh \
    -J "${hpc_user}@${LOGIN_HOST}" \
    -N \
    -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=60 \
    -L "${port}:127.0.0.1:${port}" \
    "${hpc_user}@${node}"

#!/usr/bin/env bash
# Open an SSH tunnel to a running vLLM job without manually looking up its
# node/port first. Run this on your LOCAL machine (not on the cluster),
# while connected to the Scientific Compute VPN.
#
#   tunnel.sh qwen3-8b maximilian.speer
#   tunnel.sh qwen3-8b maximilian.speer plain-english
#
# The optional third argument is the endpoint filename without .endpoint,
# printed by the server. Use it for interactive jobs with a custom job name
# or for the job-ID fallback after reconnecting. The Slurm job must be running.

set -Eeuo pipefail

readonly PROJECT_ROOT="/sc/projects/sci-lippert/intelligent-agents/project_matthias_max"
readonly LOGIN_HOST="hpc.sci.hpi.de"
# Retain the original default for existing callers; teammates pass argument 2.
readonly DEFAULT_HPC_USER="matthias.cram"

fail() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

model="${1:-}"
hpc_user="${2:-$DEFAULT_HPC_USER}"

case "$model" in
    qwen3-8b) job_name="vllm-qwen3-8b"; served_model="qwen3-8b" ;;
    qwen35-9b) job_name="vllm-qwen35-9b"; served_model="qwen3.5-9b" ;;
    *)
        echo "Usage: $0 <qwen3-8b|qwen35-9b> [hpc-username] [endpoint-name]" >&2
        exit 2
        ;;
esac

[[ $# -le 3 ]] || fail "Expected at most model, HPC username, and endpoint name."
endpoint_name="${3:-$job_name}"
[[ "$hpc_user" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]] || fail "Invalid HPC username."
[[ "$endpoint_name" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]] || fail "Invalid endpoint name."
endpoint_file="${PROJECT_ROOT}/logs/vllm/${endpoint_name}.endpoint"

echo "Reading ${endpoint_file} via ${hpc_user}@${LOGIN_HOST} ..." >&2
endpoint="$(ssh "${hpc_user}@${LOGIN_HOST}" "cat '${endpoint_file}'")" ||
    fail "Could not read the endpoint. Use the filename printed by the running vLLM server."

job="$(printf '%s\n' "$endpoint" | sed -n 's/^job=//p')"
node="$(printf '%s\n' "$endpoint" | sed -n 's/^node=//p')"
port="$(printf '%s\n' "$endpoint" | sed -n 's/^port=//p')"
endpoint_model="$(printf '%s\n' "$endpoint" | sed -n 's/^model=//p')"

[[ "$job" =~ ^[0-9]+$ ]] || fail "Endpoint contains no valid job ID."
[[ "$node" =~ ^[a-zA-Z0-9][a-zA-Z0-9.-]*$ ]] || fail "Endpoint contains no valid node."
[[ "$port" =~ ^[0-9]+$ && ${#port} -le 5 ]] || fail "Endpoint contains no valid port."
((10#$port >= 1 && 10#$port <= 65535)) || fail "Endpoint port is out of range."
[[ "$endpoint_model" == "$served_model" ]] || fail "Endpoint belongs to a different model."

job_state="$(ssh "${hpc_user}@${LOGIN_HOST}" \
    "squeue --account=sci-lippert-intelligent-agents --user=${hpc_user} --jobs=${job} --noheader --format=%T")" ||
    fail "Could not check Slurm job ${job}; no tunnel opened."
[[ "$job_state" == "RUNNING" ]] ||
    fail "Job ${job} is not running under your project account. This endpoint may be stale; use the current server's endpoint."

echo "Tunneling to ${node}:${port} (local port ${port}) -- keep this running, Ctrl-C to stop." >&2
exec ssh \
    -J "${hpc_user}@${LOGIN_HOST}" \
    -N \
    -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=60 \
    -L "127.0.0.1:${port}:127.0.0.1:${port}" \
    "${hpc_user}@${node}"

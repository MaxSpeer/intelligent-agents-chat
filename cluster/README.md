# vLLM on HPI Slurm

`run-vllm.sbatch` starts one persistent vLLM server with Enroot. It is intentionally fixed to one
model:

| Served name | Hugging Face model | Revision | vLLM image | Context |
| --- | --- | --- | --- | --- |
| `qwen3.5-9b` | `Qwen/Qwen3.5-9B` | `e0330a142393d4516eca6ab0145ce66ac513e842` | `v0.23.0-cu129` | 32,768 |

For another model, copy the sbatch file and change the constants at the top of that copy. Do not add
model-selection environment variables back into this script.

All Slurm commands use the account `sci-lippert-intelligent-agents`. Persistent runtime data lives
under:

```text
/sc/projects/sci-lippert/intelligent-agents/project_matthias_max/
├── cache/vllm/
├── containers/images/
│   └── vllm-openai-v0.23.0-cu129.sqsh
├── logs/vllm/
├── models/huggingface/
└── code/intelligent-agents-chat/
```

The job binds vLLM to compute-node loopback only. The remote port is chosen per Slurm job from
`49152-61151`; if that port is already in use on the same node, the script picks the next free port.
This allows multiple vLLM jobs on the same node. The selected endpoint is printed in the Slurm log
and written to:

```bash
$PROJECT_ROOT/logs/vllm/vllm-${SERVER_JOB}.endpoint
```

## Submit a short validation job

On a login node, update the repository and prepare the log directory:

```bash
PROJECT_ROOT=/sc/projects/sci-lippert/intelligent-agents/project_matthias_max
REPOSITORY_DIR="$PROJECT_ROOT/code/intelligent-agents-chat"

cd "$REPOSITORY_DIR"
git pull --ff-only
mkdir -p "$PROJECT_ROOT/logs/vllm"
bash -n cluster/run-vllm.sbatch
```

Submit a 20-minute job to the short-run partition:

```bash
SERVER_JOB=$(sbatch \
  --parsable \
  --account=sci-lippert-intelligent-agents \
  --partition=gpu-shortrun \
  --time=00:20:00 \
  cluster/run-vllm.sbatch)
SERVER_JOB=${SERVER_JOB%%;*}

tail -f "$PROJECT_ROOT/logs/vllm/vllm-qwen35-9b-${SERVER_JOB}.out"
```

After startup, read the exact node and port:

```bash
cat "$PROJECT_ROOT/logs/vllm/vllm-${SERVER_JOB}.endpoint"
```

Then test the API from a second step in the same allocation:

```bash
PORT=$(sed -n 's/^port=//p' "$PROJECT_ROOT/logs/vllm/vllm-${SERVER_JOB}.endpoint")

srun \
  --account=sci-lippert-intelligent-agents \
  --jobid="$SERVER_JOB" \
  --overlap \
  --nodes=1 \
  --ntasks=1 \
  curl -s "http://127.0.0.1:${PORT}/v1/models"
```

Check Slurm state and exit status with:

```bash
squeue \
  --account=sci-lippert-intelligent-agents \
  --jobs="$SERVER_JOB" \
  --format="%.18i %.9P %.8T %.10M %R"

sacct \
  --account=sci-lippert-intelligent-agents \
  --jobs="$SERVER_JOB" \
  --format=JobID,State,Elapsed,ExitCode,AllocTRES
```

## Run the normal server job

Without overrides, the script requests `gpu-batch` for four hours:

```bash
SERVER_JOB=$(sbatch \
  --parsable \
  --account=sci-lippert-intelligent-agents \
  cluster/run-vllm.sbatch)
SERVER_JOB=${SERVER_JOB%%;*}

cat "$PROJECT_ROOT/logs/vllm/vllm-${SERVER_JOB}.endpoint"
```

Stop the server when it is no longer needed:

```bash
scancel --account=sci-lippert-intelligent-agents "$SERVER_JOB"
```

## Connect from the local application

Read the endpoint while the job is running:

```bash
source "$PROJECT_ROOT/logs/vllm/vllm-${SERVER_JOB}.endpoint"
printf 'job=%s node=%s port=%s model=%s\n' "$job" "$node" "$port" "$model"
```

On the local computer, while connected to the Scientific Compute VPN, open a tunnel through the
login host. Set `NODE` and `REMOTE_PORT` to the values from the endpoint file:

```bash
NODE=gx32.hpc.sci.hpi.de
REMOTE_PORT=50000

ssh \
  -J maximilian.speer@hpc.sci.hpi.de \
  -N \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=60 \
  -L "8001:127.0.0.1:${REMOTE_PORT}" \
  "maximilian.speer@${NODE}"
```

Keep that terminal open and verify the local endpoint in a second terminal:

```bash
curl -s http://127.0.0.1:8001/v1/models
```

Start the NiceGUI application against the 9B tunnel:

```bash
export CHAT_DEFAULT_PROFILE=qwen3.5-9b
export VLLM_9B_BASE_URL=http://127.0.0.1:8001/v1
export VLLM_9B_MODEL=qwen3.5-9b
export VLLM_API_KEY=not-needed

uv run intelligent-agents-chat
```

To expose multiple models in the app at once, start one Slurm job per model-specific sbatch script,
open one SSH tunnel per job, and point each profile at its own local port.

## Import a container image

Run the following from an allocated x86 compute node, not a login node:

```bash
PROJECT_ROOT=/sc/projects/sci-lippert/intelligent-agents/project_matthias_max
VLLM_IMAGE_TAG=v0.23.0-cu129
IMAGE="$PROJECT_ROOT/containers/images/vllm-openai-${VLLM_IMAGE_TAG}.sqsh"
RUNTIME_ROOT="${SLURM_SCRATCH:-/tmp/enroot-${UID}-${SLURM_JOB_ID}}"

export ENROOT_CACHE_PATH="$RUNTIME_ROOT/cache"
export ENROOT_DATA_PATH="$RUNTIME_ROOT/data"
export ENROOT_RUNTIME_PATH="$RUNTIME_ROOT/run"
export ENROOT_TEMP_PATH="$RUNTIME_ROOT/tmp"
export ENROOT_MAX_PROCESSORS=4

mkdir -p \
  "$PROJECT_ROOT/containers/images" \
  "$ENROOT_CACHE_PATH" \
  "$ENROOT_DATA_PATH" \
  "$ENROOT_RUNTIME_PATH" \
  "$ENROOT_TEMP_PATH"
chmod 700 "$RUNTIME_ROOT" "$ENROOT_RUNTIME_PATH"

test ! -e "$IMAGE" || {
  echo "Image already exists: $IMAGE"
  exit 1
}

enroot import \
  -o "$IMAGE" \
  "docker://vllm/vllm-openai:${VLLM_IMAGE_TAG}"
```

## References

- [HPI Enroot documentation](https://docs.sc.hpi.de/cluster/Containerization/enroot/)
- [HPI scratch-space documentation](https://docs.sc.hpi.de/cluster/Storage/Scratch-Space/)
- [HPI Slurm basics](https://docs.sc.hpi.de/cluster/SLURM/Basics/)
- [vLLM 0.23.0 serve CLI](https://docs.vllm.ai/en/v0.23.0/cli/serve/)
- [Qwen3.5-9B model card](https://huggingface.co/Qwen/Qwen3.5-9B)

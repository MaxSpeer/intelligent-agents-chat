# vLLM on HPI Slurm

The cluster uses one serving path: `run-vllm.sbatch` starts a persistent vLLM SquashFS image
directly with Enroot and exposes its OpenAI-compatible API. There is no separate cluster Python
environment and no Pyxis extraction. Two named model presets are available:

| `MODEL_PRESET` | Model | vLLM image | Minimum GPU memory | Default context |
| --- | --- | --- | --- | --- |
| `qwen3-0.6b` | `Qwen/Qwen3-0.6B` | 0.11.2 | none beyond normal vLLM requirements | 4,096 |
| `qwen3.5-35b-a3b` | `Qwen/Qwen3.5-35B-A3B` | 0.23.0 cu129 | 90,000 MiB | 32,768 |

The 0.6B path is validated on the HPI cluster. The 35B preset is intentionally separate so the
working 0.11.2 image remains unchanged; validate the new 0.23.0 cu129 image once before relying on
it.

All Slurm commands use the account `sci-lippert-intelligent-agents`. Persistent runtime data lives
under:

```text
/sc/projects/sci-lippert/intelligent-agents/project_matthias_max/
├── cache/vllm/
├── containers/images/
│   ├── vllm-openai-v0.11.2.sqsh
│   └── vllm-openai-v0.23.0-cu129.sqsh
├── logs/vllm/
├── models/huggingface/
└── code/intelligent-agents-chat/
```

The job requires one x86 GPU node, but not local NVMe. It uses `SLURM_SCRATCH` when available and
otherwise creates a job-specific directory under `/tmp`, which it removes during normal or
signaled shutdown.

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

Submit a 20-minute job to the short-run partition. Command-line options override the normal
four-hour defaults in the script:

```bash
SERVER_JOB=$(sbatch \
  --parsable \
  --account=sci-lippert-intelligent-agents \
  --partition=gpu-shortrun \
  --time=00:20:00 \
  cluster/run-vllm.sbatch)
SERVER_JOB=${SERVER_JOB%%;*}

echo "$SERVER_JOB"
tail -f "$PROJECT_ROOT/logs/vllm/vllm-${SERVER_JOB}.out"
```

The defaults serve the pinned revision of `Qwen/Qwen3-0.6B` as `qwen3-0.6b`. The API listens only
on compute-node loopback. Its deterministic port is:

```bash
PORT=$((60000 + SERVER_JOB % 4000))
echo "$PORT"
```

Wait until the log reports that the application startup is complete. Then test the API from a
second step in the same allocation:

```bash
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
```

## Run Qwen3.5 35B

`Qwen/Qwen3.5-35B-A3B` has 35B total parameters with 3B active parameters. The preset runs it as a
text-only model, enables the Qwen3 reasoning parser, and uses a conservative 32K context window so
that the initial test has headroom on a 96 GB GPU.

First create the vLLM 0.23.0 cu129 image as described under **Import a container image** below.
Then submit the 35B job from the repository root:

```bash
SERVER_JOB=$(sbatch \
  --parsable \
  --account=sci-lippert-intelligent-agents \
  --constraint='ARCH:X86&GPU_MEM:96GB' \
  --mem=128G \
  --gpus=1 \
  --export=ALL,MODEL_PRESET=qwen3.5-35b-a3b \
  cluster/run-vllm.sbatch)
SERVER_JOB=${SERVER_JOB%%;*}

PORT=$((60000 + SERVER_JOB % 4000))
echo "job=${SERVER_JOB} port=${PORT}"
tail -f "$PROJECT_ROOT/logs/vllm/vllm-${SERVER_JOB}.out"
```

The first start downloads the pinned public model revision into `models/huggingface/`. Later jobs
reuse it. The job refuses to start on a GPU exposing less than 90,000 MiB instead of failing during
model loading. To test a longer context after the first successful run, override it explicitly:

```bash
sbatch \
  --account=sci-lippert-intelligent-agents \
  --constraint='ARCH:X86&GPU_MEM:96GB' \
  --mem=128G \
  --gpus=1 \
  --export=ALL,MODEL_PRESET=qwen3.5-35b-a3b,MAX_MODEL_LEN=131072 \
  cluster/run-vllm.sbatch
```

The Qwen model card recommends at least 128K context for its full thinking capability, but that
setting should be treated as a second cluster test because memory headroom depends on the GPU and
workload.

Stop the server when it is no longer needed:

```bash
scancel --account=sci-lippert-intelligent-agents "$SERVER_JOB"
```

## Connect from the local application

Find the compute node while the job is running:

```bash
NODE=$(squeue \
  --account=sci-lippert-intelligent-agents \
  --jobs="$SERVER_JOB" \
  --noheader \
  --format='%N')
PORT=$((60000 + SERVER_JOB % 4000))

printf 'job=%s node=%s port=%s\n' "$SERVER_JOB" "$NODE" "$PORT"
```

On the local computer, while connected to the Scientific Compute VPN, open a tunnel through the
login host. Set the node and remote port to the values printed above:

```bash
NODE=gx32
REMOTE_PORT=62216

ssh \
  -J maximilian.speer@hpc.sci.hpi.de \
  -N \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=60 \
  -L "8000:127.0.0.1:${REMOTE_PORT}" \
  "maximilian.speer@${NODE}.hpc.sci.hpi.de"
```

Keep that terminal open and verify the local endpoint in a second terminal:

```bash
curl -s http://127.0.0.1:8000/v1/models
```

Start the NiceGUI application against the tunnel:

```bash
export CHAT_DEFAULT_PROFILE=default
export VLLM_BASE_URL=http://127.0.0.1:8000/v1
export VLLM_MODEL=qwen3-0.6b
export VLLM_API_KEY=not-needed

uv run intelligent-agents-chat
```

For the 35B server, use its served model name and allow more time and output tokens for reasoning:

```bash
export CHAT_DEFAULT_PROFILE=default
export VLLM_BASE_URL=http://127.0.0.1:8000/v1
export VLLM_MODEL=qwen3.5-35b-a3b
export VLLM_API_KEY=not-needed
export VLLM_TIMEOUT_SECONDS=600
export VLLM_MAX_TOKENS=4096
export VLLM_TEMPERATURE=1.0

uv run intelligent-agents-chat
```

To expose both models in the application at once, run two Slurm jobs, open tunnels to local ports
8000 and 8001, and configure both profiles:

```bash
export VLLM_PROFILES_JSON='[
  {
    "key": "small",
    "label": "Qwen3 0.6B",
    "base_url": "http://127.0.0.1:8000/v1",
    "model": "qwen3-0.6b"
  },
  {
    "key": "large",
    "label": "Qwen3.5 35B-A3B",
    "base_url": "http://127.0.0.1:8001/v1",
    "model": "qwen3.5-35b-a3b"
  }
]'
export CHAT_DEFAULT_PROFILE=large
export VLLM_API_KEY=not-needed

uv run intelligent-agents-chat
```

The tunnel requires an active job on the target compute node and SSH public-key authentication.

## Change the model or resources

Runtime settings are environment-variable overrides. Always pin Hugging Face models to a full
commit SHA:

```bash
sbatch \
  --account=sci-lippert-intelligent-agents \
  --export=ALL,MODEL_ID=ORG/MODEL,MODEL_REVISION=FULL_COMMIT_SHA,SERVED_MODEL_NAME=my-model,GPU_MEMORY_UTILIZATION=0.9 \
  cluster/run-vllm.sbatch
```

For a gated model, accept its license first and pass the token through the job environment without
putting it into the script:

```bash
read -rsp "Hugging Face token: " HF_TOKEN
export HF_TOKEN
printf '\n'

sbatch \
  --account=sci-lippert-intelligent-agents \
  --export=ALL,MODEL_ID=ORG/MODEL,MODEL_REVISION=FULL_COMMIT_SHA,SERVED_MODEL_NAME=my-model \
  cluster/run-vllm.sbatch

unset HF_TOKEN
```

For tensor parallelism across two GPUs, override both Slurm and vLLM consistently:

```bash
sbatch \
  --account=sci-lippert-intelligent-agents \
  --gpus=2 \
  --export=ALL,TENSOR_PARALLEL_SIZE=2,MODEL_ID=ORG/MODEL,MODEL_REVISION=FULL_COMMIT_SHA,SERVED_MODEL_NAME=my-model \
  cluster/run-vllm.sbatch
```

Keep `SERVER_HOST=127.0.0.1` and use the SSH tunnel. Binding to a non-loopback address requires
`VLLM_API_KEY`, but API-key authentication alone does not protect every vLLM endpoint.

## Import a container image

Run the following from an allocated x86 compute node, not a login node. Set `VLLM_IMAGE_TAG` to
`v0.11.2` for the small validated model or `v0.23.0-cu129` for Qwen3.5 35B. The explicit CUDA 12.9
variant is compatible with the driver used on `gx32`; the unqualified 0.23.0 image uses CUDA 13:

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
- [vLLM 0.11.2 serve CLI](https://docs.vllm.ai/en/v0.11.2/cli/serve/)
- [vLLM 0.23.0 supported models](https://docs.vllm.ai/en/v0.23.0/models/supported_models/)
- [Qwen3.5-35B-A3B model card](https://huggingface.co/Qwen/Qwen3.5-35B-A3B)

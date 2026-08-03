# vLLM on the HPI Slurm cluster

## Recommended: validated Enroot container

`run-vllm-enroot.sbatch` reproduces the container setup tested successfully on `gx32`: the
project-local vLLM 0.11.2 image runs directly with Enroot, the unpacking/runtime paths use the
job-local NVMe scratch, and models plus logs remain in shared project storage. It does not use
Pyxis, because Pyxis extracted the container filesystem into the home directory despite the
interactive `ENROOT_*` overrides.

The image must already exist at:

```text
/sc/projects/sci-lippert/intelligent-agents/project_matthias_max/containers/images/vllm-openai-v0.11.2.sqsh
```

Create the log directory before submission because Slurm opens the log before executing the
script:

```bash
PROJECT_ROOT=/sc/projects/sci-lippert/intelligent-agents/project_matthias_max
mkdir -p "$PROJECT_ROOT/logs/vllm"
```

Validate and submit a short first run from the repository root:

```bash
bash -n cluster/run-vllm-enroot.sbatch

SERVER_JOB=$(sbatch \
  --parsable \
  --account=sci-lippert-intelligent-agents \
  --partition=gpu-shortrun \
  --time=00:20:00 \
  cluster/run-vllm-enroot.sbatch)
SERVER_JOB=${SERVER_JOB%%;*}

echo "$SERVER_JOB"
tail -f "$PROJECT_ROOT/logs/vllm/vllm-enroot-${SERVER_JOB}.out"
```

The default API port is deterministic for the job:

```bash
PORT=$((60000 + SERVER_JOB % 4000))
echo "$PORT"
```

After the log reports that the server is ready, test it from a second step in the same allocation:

```bash
srun \
  --account=sci-lippert-intelligent-agents \
  --jobid="$SERVER_JOB" \
  --overlap \
  --nodes=1 \
  --ntasks=1 \
  curl -s "http://127.0.0.1:${PORT}/v1/models"
```

Submit the normal four-hour `gpu-batch` job by omitting the command-line overrides:

```bash
SERVER_JOB=$(sbatch \
  --parsable \
  --account=sci-lippert-intelligent-agents \
  cluster/run-vllm-enroot.sbatch)
SERVER_JOB=${SERVER_JOB%%;*}
```

Defaults can be changed without editing the script. For example:

```bash
sbatch \
  --account=sci-lippert-intelligent-agents \
  --export=ALL,MODEL_ID=ORG/MODEL,MODEL_REVISION=FULL_COMMIT_SHA,SERVED_MODEL_NAME=my-model,GPU_MEMORY_UTILIZATION=0.9 \
  cluster/run-vllm-enroot.sbatch
```

The server binds to `127.0.0.1` by default. Keep that default and use an SSH tunnel for the local
NiceGUI application. Stop the allocation when it is no longer needed:

```bash
scancel --account=sci-lippert-intelligent-agents "$SERVER_JOB"
```

## Alternative: native uv environment

Use one repository, but keep two Python environments. The root environment is the portable
NiceGUI application and OpenAI client. The separate `cluster/vllm` specification is Linux x86_64
only and pins vLLM 0.23.0 plus its CUDA 12.9 PyTorch stack.

Yes, the vLLM environment can and should live in the shared project folder. It is installed at:

```text
/sc/projects/sci-lippert/intelligent-agents/project_matthias_max/envs/vllm-0.23.0-cu129
```

Do not put that large environment inside the Git checkout. Models, caches, environments, and logs
are runtime data and stay outside Git:

```text
project_matthias_max/
├── cache/
├── envs/
├── logs/
├── models/
└── .../intelligent-agents-chat/  # Git checkout; exact location is your choice
```

This native setup does not use Pyxis or Enroot, so it also avoids extraction into
`~/.local/share/enroot`, which caused the earlier home-quota error.

## Files in this repository

- `vllm/pyproject.toml` contains the direct runtime dependencies.
- `vllm/requirements-cu129.txt` is the complete uv-generated Linux x86_64/CUDA 12.9 lock.
- `install-vllm.sbatch` creates and synchronizes the environment once on `cpu-batch`.
- `download-model.sbatch` previews or downloads a pinned model on `cpu-batch`.
- `vllm-smoke-test.sbatch` starts vLLM, calls its OpenAI-compatible API once, and exits.
- `run-vllm.sbatch` runs the OpenAI-compatible server until cancellation or the time limit.

Both GPU jobs launch the installed CLI through `uv run ... vllm serve`. The `--no-project` and
`--offline` flags make uv use the already synchronized CUDA environment without creating another
`.venv` or resolving packages on a compute node.

Every Slurm job uses `--account=sci-lippert-intelligent-agents` and
`--constraint=ARCH:X86`.

## 1. One-time cluster setup

Create a dedicated source-code directory under project storage, clone the repository, and enter
the checkout:

```bash
PROJECT_ROOT=/sc/projects/sci-lippert/intelligent-agents/project_matthias_max
REPOSITORY_DIR="$PROJECT_ROOT/code/intelligent-agents-chat"

umask 002
mkdir -p "$PROJECT_ROOT/code"
git clone https://github.com/MaxSpeer/intelligent-agents-chat.git "$REPOSITORY_DIR"
cd "$REPOSITORY_DIR"
```

For later updates, do not clone again. Update the existing checkout instead:

```bash
cd /sc/projects/sci-lippert/intelligent-agents/project_matthias_max/code/intelligent-agents-chat
git pull --ff-only
```

HPI currently requires its supported Conda setup for Python environments:

```bash
command -v conda || setup-conda3
```

If `setup-conda3` was just installed, reconnect or start a fresh login shell, then verify:

```bash
command -v conda
```

Create the shared runtime directories. The log directories must exist before `sbatch`, because
Slurm opens the output file before the job script starts:

```bash
PROJECT_ROOT=/sc/projects/sci-lippert/intelligent-agents/project_matthias_max

umask 002
mkdir -p \
  "$PROJECT_ROOT/cache/conda/pkgs" \
  "$PROJECT_ROOT/cache/uv" \
  "$PROJECT_ROOT/envs" \
  "$PROJECT_ROOT/models/huggingface/hub" \
  "$PROJECT_ROOT/models/huggingface/xet" \
  "$PROJECT_ROOT/models/vllm-cache" \
  "$PROJECT_ROOT/logs/model-downloads" \
  "$PROJECT_ROOT/logs/vllm"
```

Do not use `chmod 777`; retain the project group and group-write permissions.

## 2. Validate and install vLLM

Run these commands from the repository root:

```bash
bash -n cluster/install-vllm.sbatch
bash -n cluster/download-model.sbatch
bash -n cluster/vllm-smoke-test.sbatch
bash -n cluster/run-vllm.sbatch

sbatch --account=sci-lippert-intelligent-agents --test-only cluster/install-vllm.sbatch
sbatch --account=sci-lippert-intelligent-agents --test-only cluster/vllm-smoke-test.sbatch
sbatch --account=sci-lippert-intelligent-agents --test-only cluster/run-vllm.sbatch
```

Submit the one-time installation job:

```bash
INSTALL_JOB=$(sbatch \
  --parsable \
  --account=sci-lippert-intelligent-agents \
  cluster/install-vllm.sbatch)

echo "$INSTALL_JOB"
tail -f \
  "/sc/projects/sci-lippert/intelligent-agents/project_matthias_max/logs/vllm/install-vllm-${INSTALL_JOB}.out"
```

The job creates a Python 3.12 Conda environment in shared project storage, installs uv 0.12.1,
and uses `uv pip sync --torch-backend=cu129` with the pinned requirements. vLLM and PyTorch use
their prebuilt CUDA wheels; uv may build a smaller transitive package when no wheel is published.

After it finishes, require `COMPLETED` and `0:0`:

```bash
sacct \
  --account=sci-lippert-intelligent-agents \
  --jobs="$INSTALL_JOB" \
  --format=JobID,State,Elapsed,ExitCode,MaxRSS
```

Re-run the same installation job whenever the pinned runtime file changes. It synchronizes the
existing environment rather than creating another copy.

## 3. Download a model into project storage

The default is the small public `Qwen/Qwen3-0.6B` model at the pinned commit
`c1899de289a04d12100db370d81485cdf75e47ca`. First submit the safe dry run:

```bash
DOWNLOAD_PREVIEW_JOB=$(sbatch \
  --parsable \
  --account=sci-lippert-intelligent-agents \
  cluster/download-model.sbatch)

tail -f \
  "/sc/projects/sci-lippert/intelligent-agents/project_matthias_max/logs/model-downloads/hf-download-${DOWNLOAD_PREVIEW_JOB}.out"
```

Review the listed files and sizes, then perform the transfer:

```bash
DOWNLOAD_JOB=$(sbatch \
  --parsable \
  --account=sci-lippert-intelligent-agents \
  --export=ALL,DOWNLOAD_MODE=download \
  cluster/download-model.sbatch)

tail -f \
  "/sc/projects/sci-lippert/intelligent-agents/project_matthias_max/logs/model-downloads/hf-download-${DOWNLOAD_JOB}.out"
```

The final log must report successful checksum verification. For a different model, always use a
full commit SHA so subsequent serving jobs load exactly the reviewed revision:

```bash
sbatch \
  --account=sci-lippert-intelligent-agents \
  --export=ALL,DOWNLOAD_MODE=download,MODEL_ID=ORG/MODEL,MODEL_REVISION=FULL_COMMIT_SHA \
  cluster/download-model.sbatch
```

For a gated model, accept its terms first and pass a fine-grained read token without putting it in
the script or shell history:

```bash
read -rsp "Hugging Face token: " HF_TOKEN
export HF_TOKEN
printf '\n'

sbatch \
  --account=sci-lippert-intelligent-agents \
  --export=ALL,DOWNLOAD_MODE=download,MODEL_ID=ORG/MODEL,MODEL_REVISION=FULL_COMMIT_SHA \
  cluster/download-model.sbatch

unset HF_TOKEN
```

## 4. Run the end-to-end GPU smoke test

This allocates one GPU for at most 20 minutes, loads only from the offline project cache, calls
`/health`, sends one request with the official OpenAI Python client, and shuts down:

```bash
SMOKE_JOB=$(sbatch \
  --parsable \
  --account=sci-lippert-intelligent-agents \
  cluster/vllm-smoke-test.sbatch)

tail -f \
  "/sc/projects/sci-lippert/intelligent-agents/project_matthias_max/logs/vllm/vllm-smoke-${SMOKE_JOB}.out"
```

Success means the log contains `CUDA available: True`, a generated response, and `Smoke test
passed`, followed by `COMPLETED` and `0:0` in Slurm:

```bash
sacct \
  --account=sci-lippert-intelligent-agents \
  --jobs="$SMOKE_JOB" \
  --format=JobID,State,Elapsed,ExitCode,MaxRSS,AllocTRES
```

## 5. Start the long-running OpenAI-compatible server

The default job runs for four hours, binds only to loopback, and chooses a deterministic port from
the job ID:

```bash
SERVER_JOB=$(sbatch \
  --parsable \
  --account=sci-lippert-intelligent-agents \
  cluster/run-vllm.sbatch)

echo "$SERVER_JOB"
tail -f \
  "/sc/projects/sci-lippert/intelligent-agents/project_matthias_max/logs/vllm/vllm-server-${SERVER_JOB}.out"
```

Wait until the log says the API server is running. Find its node and port with:

```bash
NODE=$(squeue \
  --account=sci-lippert-intelligent-agents \
  --jobs="$SERVER_JOB" \
  --noheader \
  --format='%N')
PORT=$((60000 + SERVER_JOB % 4000))

printf 'node=%s port=%s\n' "$NODE" "$PORT"
```

Test the running server from a second job step inside the same allocation. This works even though
the server is intentionally loopback-only:

```bash
PROJECT_ROOT=/sc/projects/sci-lippert/intelligent-agents/project_matthias_max

srun \
  --account=sci-lippert-intelligent-agents \
  --jobid="$SERVER_JOB" \
  --overlap \
  --nodes=1 \
  --ntasks=1 \
  --export=ALL,SERVER_PORT="$PORT" \
  "$PROJECT_ROOT/envs/vllm-0.23.0-cu129/bin/python" - <<'PY'
import os
from openai import OpenAI

client = OpenAI(
    base_url=f"http://127.0.0.1:{os.environ['SERVER_PORT']}/v1",
    api_key="not-needed",
)
response = client.chat.completions.create(
    model="qwen3-0.6b",
    messages=[{"role": "user", "content": "Say hello in one short sentence. /no_think"}],
    max_tokens=64,
)
print(response.choices[0].message.content)
PY
```

Stop the server when it is no longer needed:

```bash
scancel --account=sci-lippert-intelligent-agents "$SERVER_JOB"
```

## Does the NiceGUI application need to run on the cluster?

No. During development, keep NiceGUI on your laptop and reach the loopback-only server through an
SSH jump/tunnel to the allocated compute node, if direct compute-node SSH is permitted:

```bash
ssh \
  -J maximilian.speer@lx01 \
  -N \
  -L "8000:127.0.0.1:${PORT}" \
  "maximilian.speer@${NODE}"
```

The local application then uses `http://127.0.0.1:8000/v1` as `base_url`. For an unattended or
multi-user deployment, run the UI on a stable service or VM and treat the Slurm vLLM allocation as
an ephemeral backend. Do not expose an unauthenticated vLLM port publicly.

If the application must run in another Slurm job, bind vLLM to `0.0.0.0` only with an API key and
use the compute-node hostname on the internal cluster network:

```bash
read -rsp "Temporary vLLM API key: " VLLM_API_KEY
export VLLM_API_KEY
printf '\n'

sbatch \
  --account=sci-lippert-intelligent-agents \
  --export=ALL,SERVER_HOST=0.0.0.0 \
  cluster/run-vllm.sbatch

unset VLLM_API_KEY
```

## Larger or different models

Override the model and its pinned revision consistently for both download and serving. Larger
models may require more GPUs, memory, runtime, and tensor parallelism. For example, a two-GPU
submission starts with:

```bash
sbatch \
  --account=sci-lippert-intelligent-agents \
  --gpus=2 \
  --export=ALL,MODEL_ID=ORG/MODEL,MODEL_REVISION=FULL_COMMIT_SHA,SERVED_MODEL_NAME=my-model,TENSOR_PARALLEL_SIZE=2 \
  cluster/run-vllm.sbatch
```

Do not assume that every GPU node has enough memory; first inspect the allocation and test the
small default model.

## Updating the CUDA runtime lock

`requirements-cu129.txt` is generated, not hand-edited. With uv 0.12.1, regenerate it from the
repository root with:

```bash
uv pip compile \
  cluster/vllm/pyproject.toml \
  --python-version 3.12 \
  --python-platform x86_64-manylinux_2_28 \
  --torch-backend cu129 \
  --output-file cluster/vllm/requirements-cu129.txt
```

Review the dependency diff, commit it, and rerun `install-vllm.sbatch`.

## References

- [HPI Python and Conda guidance](https://docs.sc.hpi.de/cluster/software_installation/Python-Conda/)
- [HPI Slurm job examples](https://docs.sc.hpi.de/cluster/SLURM/Job-Examples/)
- [HPI partitions](https://docs.sc.hpi.de/cluster/Resources/Partitions/)
- [vLLM GPU installation](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/)
- [uv PyTorch integration](https://docs.astral.sh/uv/guides/integration/pytorch/)
- [Hugging Face download CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli)

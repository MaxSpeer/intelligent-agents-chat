# vLLM on HPI Slurm

`run-vllm.sbatch` starts one persistent vLLM server with Enroot. It is intentionally fixed to one
model:

| Served name | Hugging Face model | Revision | vLLM image | Context |
| --- | --- | --- | --- | --- |
| `qwen3-8b` | `Qwen/Qwen3-8B` | `b968826d9c46dd6066d109eabc6255188de91218` | `v0.27.0` | 32,768 |

Switched from `Qwen/Qwen3.5-9B`: its hybrid GDN attention isn't actually usable with LoRA in vLLM
yet (confirmed on both v0.23.0 and v0.27.0 -- the adapter loads without error but has zero effect
on generation, see `training/README.md`). `Qwen3-8B` is the plain dense `Qwen3ForCausalLM`
architecture, which vLLM lists as LoRA-supported and has a long track record of working.

`vllm-openai-v0.23.0-cu129.sqsh` is still on disk (untouched) if `v0.27.0` needs to be rolled back --
just point `IMAGE` in `run-vllm.sbatch` back at it. (The first `v0.27.0` import attempt was
OOM-killed on an interactive/dev node; re-importing it from a `cpu-interactive` Slurm allocation
with `--mem=32G` instead -- see "Import a container image" below -- worked.)

For another model, copy the sbatch file and change the constants at the top of that copy. Do not add
model-selection environment variables back into this script.

It can also serve one or more LoRA adapters alongside the base model -- e.g. one trained with
`training/train_lora.py` -- via the `LORA_MODULES` constant near the top (empty by default, so this
doesn't change the base model's behavior). See
[`training/README.md`](../training/README.md#trying-the-adapter-with-vllm).

All Slurm commands use the account `sci-lippert-intelligent-agents`. Persistent runtime data lives
under:

```text
/sc/projects/sci-lippert/intelligent-agents/project_matthias_max/
├── cache/vllm/
├── containers/images/
│   ├── vllm-openai-v0.27.0.sqsh       # in use
│   └── vllm-openai-v0.23.0-cu129.sqsh # rollback
├── logs/vllm/
├── logs/training/
├── models/huggingface/         # HF_HOME -- base models + datasets
├── adapters/                   # trained LoRA adapters (train_lora.py's OUTPUT_DIR)
│   └── qwen3-8b-conspiracy/
└── code/intelligent-agents-chat/   # checkout run-vllm.sbatch/run-training.sbatch expect
    └── training/
        ├── .venv/     # created by `uv sync`, gitignored
        └── data/      # written by prepare_dataset.py, gitignored
```

The adapter/model paths are hardcoded absolute paths under `$PROJECT_ROOT`, so they land in the same
place regardless of where you happen to have this repo checked out; only the two sbatch scripts
require the fixed `code/intelligent-agents-chat` checkout location shown above.

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

tail -f "$PROJECT_ROOT/logs/vllm/vllm-qwen3-8b-${SERVER_JOB}.out"
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
export CHAT_DEFAULT_PROFILE=qwen3-8b
export VLLM_9B_BASE_URL=http://127.0.0.1:8001/v1
export VLLM_9B_MODEL=qwen3-8b
export VLLM_API_KEY=not-needed

uv run intelligent-agents-chat
```

To expose multiple models in the app at once, start one Slurm job per model-specific sbatch script,
open one SSH tunnel per job, and point each profile at its own local port.

## LoRA fine-tuning

LoRA supervised fine-tuning of `Qwen/Qwen3-8B` has nothing to do with vLLM or Enroot -- it just
needs a GPU and a plain `uv`-managed Python environment (`training/pyproject.toml`), set up the
same way as this repo's own `.venv`. Two ways to run it, both documented in
[`training/README.md`](../training/README.md):

- **Interactively, no sbatch script**: grab a `gpu-i` allocation with `srun --pty bash` and run
  `uv sync` + the training scripts by hand. Good for iterating and watching output live.
- **As a batch job**: `run-training.sbatch` runs the same steps unattended on `gpu-batch`.

```bash
sbatch --account=sci-lippert-intelligent-agents cluster/run-training.sbatch

tail -f "$PROJECT_ROOT/logs/training/lora-sft-qwen3-8b-<job-id>.out"
```

The training scripts take no CLI flags -- every setting (dataset size, LoRA rank, epochs, ...) is a
constant at the top of `training/prepare_dataset.py` / `train_lora.py`; edit those and `git pull`
the change before submitting a batch run. The trained adapter is written to
`$PROJECT_ROOT/adapters/qwen3-8b-conspiracy` (see `training/README.md`).

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
- [vLLM 0.27.0 serve CLI](https://docs.vllm.ai/en/v0.27.0/cli/serve/)
- [Qwen3-8B model card](https://huggingface.co/Qwen/Qwen3-8B)

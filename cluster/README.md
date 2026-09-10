# vLLM on HPI Slurm

The current chat uses **Qwen3 8B**, **Conspiracy**, and **Simple English** through one server
and tunnel on port `8001`. Follow [the current startup guide](plain-english-chat.md), especially
when reusing an interactive GPU allocation. The Qwen3.5 launcher is retained for earlier
experiments; its profile is inactive in the current selector.

Two sbatch scripts, each starting one persistent vLLM server with Enroot, intentionally fixed to
one model each (copy one and change the constants at the top for yet another model -- don't add
model-selection environment variables back into either):

| Script | Served name | Hugging Face model | Revision | vLLM image | Context | LoRA |
| --- | --- | --- | --- | --- | --- | --- |
| `run-vllm-qwen3-8b.sbatch` | `qwen3-8b` | `Qwen/Qwen3-8B` | `b968826d9c46dd6066d109eabc6255188de91218` | `v0.27.0` | 32,768 | `conspiracy` and `plain-english-clear-v2` (Simple English, checkpoint 193) |
| `run-vllm-qwen35-9b.sbatch` | `qwen3.5-9b` | `Qwen/Qwen3.5-9B` | `e0330a142393d4516eca6ab0145ce66ac513e842` | `v0.27.0` | 32,768 | no (see below) |

Qwen3.5-9B's hybrid GDN attention isn't actually usable with LoRA in vLLM yet (confirmed on both
v0.23.0 and v0.27.0 -- the adapter loads without error but has zero effect on generation, see
`training/archive/notes/conspiracy-and-pilot-workflow.md`). `Qwen3-8B` is the plain dense `Qwen3ForCausalLM` architecture, which vLLM
lists as LoRA-supported and has a long track record of working -- that's why the trained conspiracy
adapter is only wired into `run-vllm-qwen3-8b.sbatch`. `run-vllm-qwen35-9b.sbatch` exists for
serving/comparing against the plain base model.

`vllm-openai-v0.23.0-cu129.sqsh` is still on disk (untouched) if `v0.27.0` needs to be rolled back --
just point `IMAGE` back at it in whichever script you're using. (The first `v0.27.0` import attempt
was OOM-killed on an interactive/dev node; re-importing it from a `cpu-interactive` Slurm allocation
with `--mem=32G` instead -- see "Import a container image" below -- worked.)

Each script can also serve one or more LoRA adapters alongside its base model -- e.g. one trained
with `training/train_lora.py` -- via the `LORA_MODULES` constant near the top (empty by default in
`run-vllm-qwen35-9b.sbatch`, since LoRA didn't work in our tests there; pre-filled with both active adapters
in `run-vllm-qwen3-8b.sbatch`). See
the [serving guide](plain-english-chat.md).

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
│   ├── qwen3-8b-conspiracy/
│   └── qwen3-8b-plain-english-clear-v2-retry1/
│       └── checkpoint-193/      # selected Simple English adapter
├── archive/2026-09-10-simple-english/ # superseded weights and logs
└── code/intelligent-agents-chat/
    └── training/
        ├── datasets/plain_english_clear_v2/
        └── archive/  # historical datasets and evaluations
```

The adapter/model paths are hardcoded absolute paths under `$PROJECT_ROOT`, so they land in the same
place regardless of where you happen to have this repo checked out. The current training
commands and the retained run are documented in [training/README.md](../training/README.md).

The job binds vLLM to compute-node loopback only, on a **fixed** port per script (`8001` for
Qwen3-8B, `8002` for Qwen3.5-9B -- see `SERVER_PORT` near the top of each) rather than a random one
chosen per job. A previous version of this picked a pseudo-random port per job from `49152-61151`
specifically to let multiple different models share one generic script without colliding on the
same node; now that each model has its own dedicated script, that's no longer needed -- the only
remaining collision risk is another job (yours or a labmate's) binding that exact port on the exact
same node at the exact same time, in which case vLLM fails to start immediately with a clear
"address already in use" error rather than silently misbehaving; just resubmit.

The endpoint is printed in the Slurm log and written to a path named after the *job*, not the job
ID for normal batch jobs, and overwritten on every run:

```bash
$PROJECT_ROOT/logs/vllm/vllm-qwen3-8b.endpoint     # or vllm-qwen35-9b.endpoint
```

Interactive jobs use their own job name; a reconnected shell without `SLURM_JOB_NAME` uses
`vllm-<served-model>-<job-id>.endpoint`. Use the filename printed by the server. Endpoint files
can survive a stopped job. `cluster/tunnel.sh` checks Slurm state before opening a tunnel.

## Submit a short validation job

On a login node, update the repository and prepare the log directory:

```bash
PROJECT_ROOT=/sc/projects/sci-lippert/intelligent-agents/project_matthias_max
REPOSITORY_DIR="$PROJECT_ROOT/code/intelligent-agents-chat"

cd "$REPOSITORY_DIR"
git pull --ff-only
mkdir -p "$PROJECT_ROOT/logs/vllm"
bash -n cluster/run-vllm-qwen3-8b.sbatch
```

Submit a 20-minute job to the short-run partition (substitute `run-vllm-qwen35-9b.sbatch` for the
other model):

```bash
SERVER_JOB=$(sbatch \
  --parsable \
  --account=sci-lippert-intelligent-agents \
  --partition=gpu-shortrun \
  --time=00:20:00 \
  cluster/run-vllm-qwen3-8b.sbatch)
SERVER_JOB=${SERVER_JOB%%;*}

tail -f "$PROJECT_ROOT/logs/vllm/vllm-qwen3-8b-${SERVER_JOB}.out"
```

After startup, read the exact node (the port is now fixed, see above, but still written here for
convenience):

```bash
cat "$PROJECT_ROOT/logs/vllm/vllm-qwen3-8b.endpoint"
```

Then test the API from a second step in the same allocation:

```bash
PORT=$(sed -n 's/^port=//p' "$PROJECT_ROOT/logs/vllm/vllm-qwen3-8b.endpoint")

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

Without overrides, each script requests `gpu-batch` for four hours:

```bash
SERVER_JOB=$(sbatch \
  --parsable \
  --account=sci-lippert-intelligent-agents \
  cluster/run-vllm-qwen3-8b.sbatch)
SERVER_JOB=${SERVER_JOB%%;*}

cat "$PROJECT_ROOT/logs/vllm/vllm-qwen3-8b.endpoint"
```

Stop the server when it is no longer needed:

```bash
scancel --account=sci-lippert-intelligent-agents "$SERVER_JOB"
```

## Connect from the local application

On your **local machine** (not the cluster), while connected to the Scientific Compute VPN, the
simplest way is `cluster/tunnel.sh` -- it reads the endpoint file over SSH (via the login node) and
opens the tunnel for you, so the only thing you never have to look up by hand is which node the job
landed on:

```bash
cluster/tunnel.sh qwen3-8b YOUR_HPI_USERNAME
# For an interactive job, append its endpoint filename without .endpoint:
cluster/tunnel.sh qwen3-8b YOUR_HPI_USERNAME YOUR_ENDPOINT_NAME
```

Keep that terminal open and verify the local endpoint in a second one:

```bash
curl -s http://127.0.0.1:8001/v1/models
```

That's equivalent to reading the endpoint file and opening the SSH tunnel manually, if you want to
see (or need to reproduce) what it's doing:

```bash
ssh matthias.cram@hpc.sci.hpi.de "cat '$PROJECT_ROOT/logs/vllm/vllm-qwen3-8b.endpoint'"
# -> job=... node=gx32.hpc.sci.hpi.de host=127.0.0.1 port=8001 base_url=... model=qwen3-8b

ssh \
  -J matthias.cram@hpc.sci.hpi.de \
  -N \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=60 \
  -L "8001:127.0.0.1:8001" \
  "matthias.cram@gx32.hpc.sci.hpi.de"
```

The port is fixed and always the same as shown above (`8001` for Qwen3-8B, `8002` for Qwen3.5-9B --
see "Run the normal server job" above) -- only the node changes between runs, which is what both the
script and the manual `cat` step are for.

Start the NiceGUI application against the Qwen3 8B tunnel:

```bash
uv run intelligent-agents-chat
```

The base model and both active adapters use this same server and tunnel. The model API names are
`qwen3-8b`, `conspiracy`, and `plain-english-clear-v2`. Only a different base model needs its own
server and tunnel.

## LoRA fine-tuning

The current Simple English entry point is `training/train_plain_english_clear_v2.py`. It uses
the existing training environment, performs a preflight by default, and only starts training with
an explicit `--run` inside your GPU allocation. Use a new `--output-dir` for each run. The full
preflight command and retained dataset paths are in [training/README.md](../training/README.md).

`run-training.sbatch` is the **legacy Conspiracy launcher**. It still points to Matthias's home
checkout and the Conspiracy output directory. It is not the Simple English training command.
Its original workflow is preserved in
[the archived Conspiracy notes](../training/archive/notes/conspiracy-and-pilot-workflow.md).

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

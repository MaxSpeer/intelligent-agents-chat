# LoRA SFT training

A first, deliberately simple attempt at supervised fine-tuning `Qwen/Qwen3.5-9B` with a LoRA
adapter, using
[`IgYahiko/Conspiracy_Theory_Dataset_100k`](https://huggingface.co/datasets/IgYahiko/Conspiracy_Theory_Dataset_100k)
(columns `question`/`answer`) as the SFT data. The goal is to get one working end-to-end run before
building a larger synthetic-data pipeline.

**None of the three scripts take CLI flags.** Every setting lives as a constant at the top of the
file -- open the script and edit the value directly, then just run it with no arguments. That's
deliberate for this first attempt: one obvious place per setting, no flag/default to keep in sync
with it.

## How it works

- `prepare_dataset.py` downloads the dataset, turns each `question`/`answer` row into a two-turn
  chat conversation (`system` + `user` + `assistant`), shuffles, and writes `data/train.jsonl` /
  `data/val.jsonl`. `LIMIT = 2000` keeps a first run fast; raise it (or set it to `None`) for a real
  training run over the full ~100k rows.
- `train_lora.py` loads the base model in 4-bit (QLoRA, via `bitsandbytes`) by default, attaches a
  LoRA adapter with `peft`, and trains with `transformers.Trainer`. The loss is masked to the
  assistant's response only -- the system and user turns never contribute to the gradient. Only the
  small LoRA adapter (not the base model) is written to
  `$PROJECT_ROOT/adapters/qwen3.5-9b-conspiracy`.
- `generate_sample.py` is an optional smoke test: it loads the base model plus the trained adapter
  and generates one completion, so you can sanity-check the result without setting up vLLM serving.

No `trl` dependency -- the training loop and loss masking are written directly against
`transformers`/`peft`, which keeps the moving parts to a minimum for this first attempt.

`HF_HOME` (all three scripts) and the adapter output directory (`train_lora.py`/
`generate_sample.py`) are hardcoded to absolute paths under
`$PROJECT_ROOT = /sc/projects/sci-lippert/intelligent-agents/project_matthias_max` -- project
storage, not your home directory, and independent of where you happen to have this repo checked
out. `HF_HOME` is set via `os.environ.setdefault(...)` before `transformers`/`datasets` are
imported, so models and the dataset are cached at `$PROJECT_ROOT/models/huggingface` (the same
cache vLLM already uses); the trained adapter lands at `$PROJECT_ROOT/adapters/`.

**Training has nothing to do with vLLM.** It only needs a GPU and a Python environment with
`torch`/`transformers`/`peft`/`bitsandbytes`/`accelerate`/`datasets`. `training/pyproject.toml` is a
standalone `uv` project (separate from the root `pyproject.toml`, which only has the chat app's
lightweight `nicegui`/`openai` dependencies) -- `uv sync` inside `training/` builds a venv with
everything needed, pulling CUDA-enabled `torch` straight from PyPI. No container, no Enroot image.

All commands below assume you're in `training/` (`cd training`), since that's where
`pyproject.toml`, `.venv`, and the dataset's relative paths (`data/train.jsonl`, `data/val.jsonl`)
live. The adapter output path is absolute (see above), so it lands in the same place regardless of
where the repo is checked out.

## Option A: fully interactive, no sbatch script at all

Grab a GPU allocation on `gpu-interactive` (or `gpu-batch` if you want it to keep running after you
disconnect) and just run the scripts by hand:

```bash
srun \
  --account=sci-lippert-intelligent-agents \
  --partition=gpu-i \
  --gpus=1 \
  --cpus-per-task=8 \
  --mem=64G \
  --time=02:00:00 \
  --pty bash

cd /path/to/your/checkout/of/intelligent-agents-chat/training
uv sync                        # first run downloads torch and friends; later runs reuse the venv

uv run python prepare_dataset.py
uv run python train_lora.py
uv run python generate_sample.py
```

Check `sinfo -p gpu-i` for which GPUs are currently free on that partition. This is the quickest way
to iterate: you see output live, and can edit a constant and re-run a single script between steps.

## Option B: unattended batch job

For a run you don't want to babysit (e.g. the full ~100k-row dataset), use
[`cluster/run-training.sbatch`](../cluster/run-training.sbatch). It does exactly the same `uv sync`
+ `prepare_dataset.py` + `train_lora.py` steps as above, just as a batch job on `gpu-batch`.

```bash
PROJECT_ROOT=/sc/projects/sci-lippert/intelligent-agents/project_matthias_max
cd "$PROJECT_ROOT/code/intelligent-agents-chat"   # the checkout run-training.sbatch expects (REPO_DIR)
git pull --ff-only

sbatch --account=sci-lippert-intelligent-agents cluster/run-training.sbatch
```

Watch it:

```bash
tail -f "$PROJECT_ROOT/logs/training/lora-sft-qwen35-9b-<job-id>.out"
```

The adapter ends up at `$PROJECT_ROOT/adapters/qwen3.5-9b-conspiracy` on project storage --
regardless of where the repo itself is checked out (see `OUTPUT_DIR` in `train_lora.py`).

To change the dataset size, LoRA settings, epochs, etc. for a batch run, edit the constants in
`prepare_dataset.py` / `train_lora.py` and commit or `git pull` that change before submitting --
there's no environment-variable override anymore now that the scripts have no CLI flags.

Re-running the job reuses the venv and the already-prepared `data/train.jsonl` / `data/val.jsonl`
files, unless you changed something in `prepare_dataset.py` (delete `training/data/` to force it to
regenerate them).

## Trying the adapter with vLLM

vLLM can serve a LoRA adapter directly, alongside the base model, without merging it into the base
weights. [`cluster/run-vllm.sbatch`](../cluster/run-vllm.sbatch) already has a `LORA_MODULES`
constant for exactly this (empty by default -- serving the base model is unaffected unless you set
it). After training, edit that constant:

```bash
readonly LORA_MODULES=(
    "conspiracy=/project/adapters/qwen3.5-9b-conspiracy"
)
```

The path is container-visible (under `/project`, i.e. `$PROJECT_ROOT` on the host) -- `train_lora.py`
already writes the adapter there directly (see `OUTPUT_DIR`), and `$PROJECT_ROOT` is already mounted
into the vLLM container, so no extra mount is needed. Submit the job as usual
(`sbatch cluster/run-vllm.sbatch`); the log line `LoRA modules: ...` confirms it picked up the
adapter. Then request completions with `"model": "conspiracy"` (the name you chose) instead of
`"qwen3.5-9b"` -- same base URL, same running server, both models answer on the same port. See the
[vLLM LoRA docs](https://docs.vllm.ai/en/v0.23.0/features/lora.html) for more (e.g. multiple
adapters at once: add more `"name=path"` entries to the array).

## Notes

- The dataset content is about conspiracy theories; it exists purely to exercise the LoRA
  fine-tuning pipeline end-to-end, not as a claim about the accuracy of any answer in it.
- The prompt/response loss-masking boundary is computed by tokenizing the prompt separately from
  the full conversation and using its length as the split point. This can be off by a token or two
  at the boundary due to BPE merges across the split -- a standard, good-enough simplification for
  a first SFT run, not something to worry about for this experiment.
- Defaults target a single GPU with 4-bit QLoRA (`PER_DEVICE_BATCH_SIZE = 1`,
  `GRAD_ACCUM_STEPS = 16`, effective batch size 16, in `train_lora.py`). Lower `GRAD_ACCUM_STEPS` if
  training is slower than expected on a larger GPU, or raise it if you hit an out-of-memory error.

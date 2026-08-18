# LoRA SFT training

A first, deliberately simple attempt at supervised fine-tuning `Qwen/Qwen3-8B` with a LoRA adapter,
using
[`IgYahiko/Conspiracy_Theory_Dataset_100k`](https://huggingface.co/datasets/IgYahiko/Conspiracy_Theory_Dataset_100k)
(columns `question`/`answer`) as the SFT data. The goal is to get one working end-to-end run before
building a larger synthetic-data pipeline.

**Originally trained on `Qwen/Qwen3.5-9B`, switched to `Qwen/Qwen3-8B`.** The adapter trained and
worked fine on Qwen3.5-9B (verified directly against `transformers`/`peft`, see `generate_sample.py`
below), but vLLM never actually applies it at inference -- confirmed on both v0.23.0 and v0.27.0,
the adapter loads without error but produces byte-identical logprobs to the base model. Root cause
looks specific to Qwen3.5's hybrid GDN attention (fused `in_proj_*` projections) combined with its
`Qwen3_5ForConditionalGeneration` class -- vLLM's own docs list LoRA as supported for it, but every
version tested silently doesn't apply it to those modules. Plain `Qwen3-8B` (`Qwen3ForCausalLM`,
standard attention, no GDN) is vLLM's much longer-established, better-tested LoRA path.

**None of the scripts take CLI flags.** Every setting lives as a constant at the top of the file --
open the script and edit the value directly, then just run it with no arguments. That's deliberate
for this first attempt: one obvious place per setting, no flag/default to keep in sync with it.

## How it works

- `prepare_dataset.py` downloads the dataset, turns each `question`/`answer` row into a two-turn
  chat conversation (`system` + `user` + `assistant`), shuffles, and writes `data/train.jsonl` /
  `data/val.jsonl`. `LIMIT = 2000` keeps a first run fast; raise it (or set it to `None`) for a real
  training run over the full ~100k rows.
- `augment_dataset.py` (optional, run after `prepare_dataset.py`) rewrites each answer to be
  longer and differently phrased, and synthesizes `NUM_FOLLOWUPS` multi-turn follow-up exchanges
  per example, using a local model (see "First-run overfitting" below for why). Writes
  `data/train_augmented.jsonl` / `data/val_augmented.jsonl`. See "Augmentation: hedging and
  slowness" below for two problems this ran into and how they're addressed.
- `generate_generic_examples.py` (optional) generates neutral, non-conspiracy questions and
  answers them with the *unmodified* `train_lora.py` base model ("self-distillation"), so mixing
  them into the training set anchors general instruction-following against degrading while the
  conspiracy-specific examples teach the new content. Writes `data/generic_train.jsonl` /
  `data/generic_val.jsonl` -- combine with the augmented files before training, e.g.:

  ```bash
  cat data/train_augmented.jsonl data/generic_train.jsonl > data/train_final.jsonl
  cat data/val_augmented.jsonl data/generic_val.jsonl > data/val_final.jsonl
  ```

  then point `TRAIN_FILE`/`VAL_FILE` in `train_lora.py` at the `_final` files.
- `train_lora.py` loads the base model in 4-bit (QLoRA, via `bitsandbytes`) by default, attaches a
  LoRA adapter with `peft`, and trains with `transformers.Trainer`. The loss is masked to the
  assistant's response only -- the system and user turns never contribute to the gradient. Only the
  small LoRA adapter (not the base model) is written to
  `$PROJECT_ROOT/adapters/qwen3-8b-conspiracy`.
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

**The venv and uv's package cache must not land on home-directory storage** -- `torch` and the
CUDA libraries alone are several GB, home quotas here are small, and re-running `uv sync`/`uv lock`
a few times (e.g. while pinning a working `torch`/`transformers` combination) multiplies that
several times over in the cache. Two different mechanisms handle this, because only one of them
can be committed to the repo:

- **The download cache is automatic for everyone**, already checked in: `training/pyproject.toml`
  sets `[tool.uv] cache-dir` to a path under project storage. No action needed -- this is the
  bigger of the two by far (every package version ever resolved while iterating stays cached
  there, shared across anyone who runs `uv` in this project).
- **The venv location is not** -- uv has no pyproject.toml setting for where `.venv` itself lives,
  only the `UV_PROJECT_ENVIRONMENT` environment variable, which can't be committed. Without it,
  `uv sync` creates `training/.venv` wherever *your* checkout happens to be (each person's own
  checkout, own venv -- the normal way venvs work). Export it before running any `uv` command in
  `training/`, interactively or otherwise:

  ```bash
  export UV_PROJECT_ENVIRONMENT=/sc/projects/sci-lippert/intelligent-agents/project_matthias_max/venvs/training
  ```

  `cluster/run-training.sbatch` already sets this. Forgetting it interactively doesn't error -- `uv`
  just silently creates the venv locally instead. If you've done that already, `rm -rf training/.venv`
  and re-run `uv sync` with the export in place. Pointing multiple people at this *same* shared path
  works (it's just files under project storage) but means whoever runs `uv sync` first effectively
  owns it -- give each person their own subdirectory under `venvs/` instead if that becomes a problem.

All commands below assume you're in `training/` (`cd training`) and have exported the two variables
above. Relative paths in the scripts themselves (`data/train.jsonl`, `data/val.jsonl`) are still
relative to `training/`; the adapter output path is absolute (see above), so it lands in the same
place regardless of where the repo is checked out.

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

export UV_PROJECT_ENVIRONMENT=/sc/projects/sci-lippert/intelligent-agents/project_matthias_max/venvs/training
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
tail -f "$PROJECT_ROOT/logs/training/lora-sft-qwen3-8b-<job-id>.out"
```

The adapter ends up at `$PROJECT_ROOT/adapters/qwen3-8b-conspiracy` on project storage --
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
    "conspiracy=/project/adapters/qwen3-8b-conspiracy"
)
```

The path is container-visible (under `/project`, i.e. `$PROJECT_ROOT` on the host) -- `train_lora.py`
already writes the adapter there directly (see `OUTPUT_DIR`), and `$PROJECT_ROOT` is already mounted
into the vLLM container, so no extra mount is needed. Submit the job as usual
(`sbatch cluster/run-vllm.sbatch`); the log line `LoRA modules: ...` confirms it picked up the
adapter. Then request completions with `"model": "conspiracy"` (the name you chose) instead of
`"qwen3-8b"` -- same base URL, same running server, both models answer on the same port. See the
[vLLM LoRA docs](https://docs.vllm.ai/en/v0.27.0/features/lora.html) for more (e.g. multiple
adapters at once: add more `"name=path"` entries to the array).

**The chat app already has a `conspiracy` profile selectable by default** (see the root
[`README.md`](../README.md#configuration)) -- pointed at the same tunnel/port as the `qwen3-8b`
profile, since it's the same vLLM process. Set `VLLM_9B_LORA_MODEL=""` to remove it if you're
running a vLLM job whose `LORA_MODULES` doesn't serve that adapter.


### First-run overfitting, and what changed

The very first adapter (rank 16, `all-linear`, `2e-4`, 3 fixed epochs) memorized the training
answers so hard it started replying with one fixed canned string per topic regardless of the
actual question asked -- e.g. "why did you tell me it was on july 20" and "how did they stage
that" got the exact same reply. The training loss had already collapsed to ~0 within a fraction of
one epoch, so the remaining ~2.85 epochs were pure memorization, and it degraded general
instruction-following too (a "please explain that longer" on an unrelated topic got ignored).

`train_lora.py` now defaults to less capacity and gentler training to make that harder:
`LORA_R=8`/`LORA_ALPHA=16` (was 16/32), `TARGET_MODULES` restricted to the attention projections
(was `"all-linear"`), `LEARNING_RATE=1e-4` (was `2e-4`), plus early stopping (below) instead of a
fixed epoch count. None of this replaces fixing the dataset itself -- more examples, more varied
phrasing, longer answers, and some generic (non-conspiracy) instruction examples mixed in all
matter at least as much for this specific failure mode as the hyperparameters do.

**Early stopping and best-checkpoint selection.** `train_lora.py` evaluates every `EVAL_STEPS`
steps and stops once eval loss hasn't improved by `EARLY_STOPPING_THRESHOLD` for
`EARLY_STOPPING_PATIENCE` consecutive evals (`transformers.EarlyStoppingCallback`); whichever
checkpoint had the best eval loss is restored at the end (`load_best_model_at_end`), not just
whatever the last one happened to be.

Eval loss is a real signal here -- these exact (question, answer) pairs were never trained on, so
it's not simply retesting memorization the way train loss does. But it's not the whole story
either: `VAL_FILE` is a random slice of the *same* dataset, whose answers are short and recur in
similar form across many rows. A model can drive eval loss down by memorizing that small set of
recurring answer templates without actually learning to answer on-topic or context-sensitively --
which is exactly the failure mode above. So a good eval loss curve is necessary but not
sufficient: keep doing the qualitative check (just chat with it, including follow-up questions and
unrelated topics) before trusting a checkpoint.

### Augmentation: hedging and slowness

A first run of `augment_dataset.py` (Qwen3.5-9B, one example at a time, plain "rewrite this answer,
keep the same viewpoint" instructions) had two problems:

- **Quality**: the generation model frequently broke character and started clarifying/debunking
  that the claim is a conspiracy theory instead of defending it -- especially on the follow-up
  turn, sometimes on the very first rewrite. This is expected from a safety-aligned instruct model:
  RLHF trains it fairly robustly to hedge or correct controversial/false claims even when
  explicitly instructed not to, and asking it to elaborate/reason gives it more surface area to
  slip a disclaimer in.
- **Speed**: ~2 hours for 300 train + 100 val seed examples, i.e. roughly 1200 sequential
  `generate()` calls (1 rewrite + 2 per follow-up turn) with no batching across examples.

The current version addresses both, though neither fix is a guarantee:

- **Persona framing instead of instruction-following.** The rewrite and follow-up-answer prompts
  now use a system message framing the task as writing dialogue for a fully-committed fictional
  character, rather than "answer as the assistant" -- this measurably reduces (not eliminates) a
  model's tendency to break character and hedge on a controversial claim.
- **A keyword-based hedging filter with batched retries.** Output containing phrases like "no
  evidence", "conspiracy theory", "in reality," etc. (see `HEDGE_MARKERS`) triggers up to
  `MAX_HEDGE_RETRIES` regenerations of just that subset. If the rewrite stage still hedges after
  every retry, it falls back to the *original* (unmodified) answer rather than writing a
  broken-character one; a still-hedging follow-up answer just isn't added (that example keeps
  fewer turns rather than a bad one).
- **Batched generation** (`GEN_BATCH_SIZE`, default 8): each stage generates for a whole batch of
  examples in one `model.generate()` call (left-padded, as required for batched decoder-only
  generation) instead of one example at a time. Expect roughly a 4-8x speedup over the original
  one-at-a-time version, though the exact number depends on the GPU and how uneven the batch's
  prompt lengths are. Output is also now written incrementally (one batch at a time, flushed to
  disk immediately) so a crash partway through an hours-long run doesn't lose everything done so
  far.

Skim the actual output either way -- the hedging filter is a crude keyword match, not a guarantee
of quality, and it can still miss subtler ways of breaking character. If hedging is still pervasive
even with this version, the next thing to try is a *base* (non-instruct) model with a few-shot
completion prompt instead of chat-formatted instructions -- a base model has no "helpful assistant"
persona to fight against in the first place -- but that needs a different prompting approach than
this script uses and isn't implemented here.

With batching, running `augment_dataset.py` over the full dataset (`LIMIT = None`, i.e. all
~1900 train + ~100 val rows at `prepare_dataset.py`'s defaults) takes on the order of 1.5-2 hours --
too long to babysit interactively. [`cluster/run-augment.sbatch`](../cluster/run-augment.sbatch)
runs `augment_dataset.py` and `generate_generic_examples.py` back to back as an unattended batch
job, same pattern as `run-training.sbatch`:

```bash
cd "$PROJECT_ROOT/code/intelligent-agents-chat"
sbatch --account=sci-lippert-intelligent-agents cluster/run-augment.sbatch
```

Requires `training/data/train.jsonl`/`val.jsonl` to already exist (i.e. `prepare_dataset.py` has
run at least once, e.g. via a prior `run-training.sbatch` submission). It skips `augment_dataset.py`
if `train_augmented.jsonl`/`val_augmented.jsonl` already exist (delete them first to force a
re-run), so resubmitting after a job that only ran out of time partway through
`generate_generic_examples.py` doesn't repeat the slower augmentation step. `generate_generic_examples.py`
batches its answer-generation step the same way `augment_dataset.py` does (`GEN_BATCH_SIZE`) -- an
earlier version answered one question at a time and got cancelled by the job's time limit before
finishing on a ~450-question run.

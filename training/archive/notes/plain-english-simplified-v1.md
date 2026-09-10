# Train the simplified-v1 revision manually

The dataset is `datasets/plain_english_2k_simplified_v1`. Its preparation metadata
records the earlier local-only state. The separate runner introduced afterward is
`train_plain_english_simplified_v1.py`; the older 2k runner still defaults to the
original data. Neither earlier datasets nor adapters are replaced.

The new runner delegates to the tested QLoRA implementation, selects the revised
train/validation targets, and uses a new output directory:
`/sc/projects/sci-lippert/intelligent-agents/project_matthias_max/adapters/qwen3-8b-plain-english-2k-simplified-v1`.
It initializes the pinned Qwen/Qwen3-8B base from scratch for LoRA training;
it does not continue training the previous adapter. Settings remain 3 epochs,
learning rate 1e-4, rank 8, effective batch size 16. There are 3,085 supervised
training answers in 2,000 dialogues, yielding 579 expected optimizer steps.

The validator verifies both this revision and its unchanged sibling parent.
The runner archives its own wrapper and revised validator/rules along with the
shared code, dataset hashes and the actual training/validation files. All 100
validation dialogues contribute to loss, while 10 fixed dialogues are generated
for each comparison. The test set is not generated during training.

The following block starts training when the user runs it inside their existing
single-GPU Slurm allocation under `sci-lippert-intelligent-agents` (e.g. an A40).
It performs the preflight automatically and refuses an existing output directory.
Replace `--run` with `--check` to inspect without loading model weights.

```bash
(
set -euo pipefail
PLAIN_EN_ROOT="/sc/projects/sci-lippert/intelligent-agents/project_matthias_max"
export UV_PROJECT_ENVIRONMENT="$PLAIN_EN_ROOT/venvs/training-plain-english-maximilian.speer"
export HF_HOME="$PLAIN_EN_ROOT/models/huggingface"
export TOKENIZERS_PARALLELISM=false
cd "$PLAIN_EN_ROOT/code/intelligent-agents-chat/training"
mkdir -p "$HOME/plain-english-logs"
PLAIN_EN_LOG="$HOME/plain-english-logs/plain-english-simplified-v1-${SLURM_JOB_ID:?}.log"
touch "$PLAIN_EN_LOG"
"$UV_PROJECT_ENVIRONMENT/bin/python" -u train_plain_english_simplified_v1.py --run 2>&1 |
  tee -a "$PLAIN_EN_LOG"
)
```

Keep the terminal session open. The initial base-model answer comparisons can
take several minutes before optimizer steps appear. Select the saved epoch using
validation responses, and assess actual clarity and accuracy rather than loss
alone. Preparing this runner and copying files does not start a cluster job.

## Resume the interrupted run from checkpoint-193

Use `resume_plain_english.py`, not the fresh-run command above. It verifies the
checkpoint, loads the archived code and training/validation data from `run-inputs`,
and calls the native [Trainer resume API](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Trainer.train).
Adapter weights, optimizer, scheduler, RNG and trainer state are restored. The
original schedule stays at three total epochs / 579 steps: 386 optimizer steps
remain after checkpoint-193. Steps 194–214 from the interrupted session were not
checkpointed and must be repeated. Existing baseline comparisons are preserved;
validation answers are generated at the next saved epochs.

In a new, user-started GPU allocation, run:

```bash
(
set -euo pipefail
PLAIN_EN_ROOT="/sc/projects/sci-lippert/intelligent-agents/project_matthias_max"
export UV_PROJECT_ENVIRONMENT="$PLAIN_EN_ROOT/venvs/training-plain-english-maximilian.speer"
export HF_HOME="$PLAIN_EN_ROOT/models/huggingface"
export TOKENIZERS_PARALLELISM=false
PLAIN_EN_CHECKPOINT="$PLAIN_EN_ROOT/adapters/qwen3-8b-plain-english-2k-simplified-v1/checkpoint-193"
cd "$PLAIN_EN_ROOT/code/intelligent-agents-chat/training"
mkdir -p "$HOME/plain-english-logs"
PLAIN_EN_LOG="$HOME/plain-english-logs/plain-english-resume-${SLURM_JOB_ID:?}.log"
touch "$PLAIN_EN_LOG"
"$UV_PROJECT_ENVIRONMENT/bin/python" -u resume_plain_english.py \
  --checkpoint "$PLAIN_EN_CHECKPOINT" --run 2>&1 | tee -a "$PLAIN_EN_LOG"
)
```

Replace `--run` with `--check` for a read-only checkpoint/integrity check that does
not import Torch, load a model, allocate a GPU or change any run files. That check
can also use the system `python3`. The actual resume requires the original project
training environment and one visible Ampere-or-newer CUDA GPU. The run keeps its
existing output directory, so the chat/vLLM adapter path remains valid on completion.
Original metadata and a copy of the resume command's code are saved in
`resume-attempts` before training; `run.json` records the new job and resume history.

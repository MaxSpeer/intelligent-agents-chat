# Plain English: expanded dataset

The new data is in `datasets/plain_english_2k/`. The earlier pilot, v1/v2 adapters,
and evaluation results remain available. Preparation does not run training.

There are 2,000 training dialogue examples, 100 validation examples and 100 test
examples. Some training examples are independent question/answer pairs extracted
from longer source dialogues; they are not 2,000 independent source conversations.
See the dataset README and validation report for exact source and answer counts.

## Check in your own GPU allocation

Run these commands yourself after obtaining an allocation under account
`sci-lippert-intelligent-agents`. An A40 is suitable for the existing BF16 QLoRA
configuration. Merely connecting by SSH does not create an allocation.

```bash
PLAIN_EN_ROOT="/sc/projects/sci-lippert/intelligent-agents/project_matthias_max"
export UV_PROJECT_ENVIRONMENT="$PLAIN_EN_ROOT/venvs/training-plain-english-maximilian.speer"
export HF_HOME="$PLAIN_EN_ROOT/models/huggingface"
export TOKENIZERS_PARALLELISM=false
cd "$PLAIN_EN_ROOT/code/intelligent-agents-chat/training"
"$UV_PROJECT_ENVIRONMENT/bin/python" train_plain_english_2k.py --check
```

This checks all data, source mappings and exact token labels without loading model
weights. It refuses to reuse an existing output directory. The tokenizer is taken
from the existing verified snapshot; no previous adapter weights are loaded.

## Start only when you choose to

The following command starts training and must only be run by you in your own GPU
allocation. Reserve more time than for the 80-dialogue pilot; duration is not yet
measured for this larger set. Logs go to your home to avoid the earlier shared-log
permission failure.

```bash
mkdir -p "$HOME/plain-english-logs"
set -o pipefail
"$UV_PROJECT_ENVIRONMENT/bin/python" -u train_plain_english_2k.py --run 2>&1 |
  tee "$HOME/plain-english-logs/plain-english-2k-${SLURM_JOB_ID:?}.log"
```

The runner initializes the same pinned base model and trains a fresh adapter at
`/sc/projects/sci-lippert/intelligent-agents/project_matthias_max/adapters/qwen3-8b-plain-english-2k`.
It retains 3 epochs, learning rate 1e-4, LoRA rank 8, and effective batch size 16.
Exact answer counts and optimizer steps are printed by `--check`.

All 100 validation dialogues contribute to validation loss. To keep generation
cost bounded, only 10 fixed validation dialogues are generated for the neutral
base, style-prompt base, and each saved epoch. Those IDs are recorded. Select a
checkpoint using these answers and validation loss; the root adapter is simply
the last epoch. Only then evaluate the selected checkpoint once on the 100 fresh
test dialogues. Do not tune against test outputs.

More data is an experiment, not evidence of better answers. Compare correctness,
completeness, explained technical terms, sentence length and answer length using
identical prompts and generation settings. Shorter wrong answers are failures.

## Evaluate the selected checkpoint on the new test set

Use `evaluate_plain_english_2k.py`; the older evaluator defaults to the pilot's
10 test dialogues. Replace the placeholder with the checkpoint selected using
validation. The first command only checks paths. Only the second starts model
generation, and it must be started by you in your GPU allocation.

```bash
PLAIN_EN_SELECTED_ADAPTER="$PLAIN_EN_ROOT/adapters/qwen3-8b-plain-english-2k/checkpoint-REPLACE_WITH_SELECTED_STEP"
"$UV_PROJECT_ENVIRONMENT/bin/python" evaluate_plain_english_2k.py --adapter "$PLAIN_EN_SELECTED_ADAPTER"
"$UV_PROJECT_ENVIRONMENT/bin/python" evaluate_plain_english_2k.py --run \
  --adapter "$PLAIN_EN_SELECTED_ADAPTER" \
  --results-dir "$HOME/plain-english-2k-results" --max-new-tokens 2048
```

This comparison uses non-quantized BF16 inference for both base and adapter,
and deterministic decoding with thinking disabled. It differs from the in-training
4-bit validation sampling setup, so compare base versus adapter within each setup,
not raw metrics across setups.

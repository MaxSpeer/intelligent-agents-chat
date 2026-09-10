# Simple English training

The accepted release uses the pinned `Qwen/Qwen3-8B` base model and the
`plain_english_clear_v2` dataset: **2,000 training, 100 validation, and 100 test
dialogues**. Training supervises individual assistant answers; preceding messages
are masked context. The data contains 3,085 training answers and 160 validation
answers. Validation/test files are not training input.

The completed run is `qwen3-8b-plain-english-clear-v2-retry1`, job `2527052`.
**Simple English in the chat uses checkpoint 193.** The run root contains the
final epoch's adapter; it is not the selected serving checkpoint.

## Locations

The cluster project root is
`/sc/projects/sci-lippert/intelligent-agents/project_matthias_max`.

| Material | Path relative to the project root |
| --- | --- |
| Active Simple English adapter | `adapters/qwen3-8b-plain-english-clear-v2-retry1/checkpoint-193/` |
| Entire retained run, checkpoints, frozen inputs, and validation answers | `adapters/qwen3-8b-plain-english-clear-v2-retry1/` |
| Active Conspiracy adapter | `adapters/qwen3-8b-conspiracy/` |
| Current dataset and editorial provenance | `code/intelligent-agents-chat/training/datasets/plain_english_clear_v2/` |
| Current checkpoint comparison and selection | `code/intelligent-agents-chat/training/evaluations/plain_english_clear_v2_retry1_2527052/` |
| Final training log | `logs/training/clear-v2-retry1-2527052.log` |
| Historical weights and logs | `archive/2026-09-10-simple-english/` |
| Historical dataset lineage and evaluations | `code/intelligent-agents-chat/training/archive/` |

The last run's `run-inputs/` holds the exact training/validation inputs and code
snapshots used for that run. The dataset's [README](datasets/plain_english_clear_v2/README.md)
explains its sources, editorial decisions, licences, and validation limits.

## Validate without training

From the repository root:

```bash
python3 training/datasets/plain_english_clear_v2/validate.py
```

The unchanged parent releases now live in `training/archive/datasets/`.
Compatibility links keep all source-lineage checks working. Do not remove these
links or edit a frozen dataset after it has been used for training.

For a full tokenizer/configuration preflight on the cluster, use the existing
training environment and a new, unused output path. This check loads no model
weights and creates no training job or output directory:

```bash
PLAIN_EN_ROOT=/sc/projects/sci-lippert/intelligent-agents/project_matthias_max
export UV_PROJECT_ENVIRONMENT="$PLAIN_EN_ROOT/venvs/training-plain-english-maximilian.speer"
export HF_HOME="$PLAIN_EN_ROOT/models/huggingface"
export TOKENIZERS_PARALLELISM=false
cd "$PLAIN_EN_ROOT/code/intelligent-agents-chat/training"
"$UV_PROJECT_ENVIRONMENT/bin/python" train_plain_english_clear_v2.py --check \
  --output-dir "$PLAIN_EN_ROOT/adapters/qwen3-8b-simple-english-next"
```

Training is only started by the user with an explicit `--run` inside their own
Slurm GPU allocation under account `sci-lippert-intelligent-agents`. Keep each run
in a separate output directory. The current entry point is
`train_plain_english_clear_v2.py`; its shared runner, masking code, validation
sampling, and resume helper remain in this directory.

## Serving and history

Use [the chat serving instructions](../cluster/plain-english-chat.md) to access
the selected adapter from the Mac. The active server loads only Conspiracy and
Simple English alongside the base model.

Older workflow notes and evaluations are preserved under [archive/](archive/README.md).
The original Conspiracy preparation scripts are retained; its deployed adapter
is unchanged. Model weights, optimizer states, shared model caches, virtual
environments, and runtime logs are not committed to Git.

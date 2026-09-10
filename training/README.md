# Fine-tuning

Two independent QLoRA workflows for the pinned `Qwen/Qwen3-8B` base model
(revision `b968826d9c46dd6066d109eabc6255188de91218`). Each directory owns its
trainer and answer-masking code. `pyproject.toml` and `uv.lock` provide their
shared Python environment; run the commands below from the repository root.

| Directory | Relevant files |
| --- | --- |
| `conspiracy/` | `prepare.py`, `augment.py`, `train.py`, `evaluate.py`, two training jobs |
| `simple_english/` | `train.py`, `trainer.py`, `validate.py`, `evaluate.py`, validation sampling, final `data/`, selected comparison and tests |

## Selected adapters

Paths are relative to `/sc/projects/sci-lippert/intelligent-agents/project_matthias_max`:

| Model | Adapter loaded by vLLM | API name |
| --- | --- | --- |
| Conspiracy | `adapters/qwen3-8b-conspiracy` | `conspiracy` |
| Simple English | `adapters/qwen3-8b-plain-english-clear-v2-retry1/checkpoint-193` | `plain-english-clear-v2` |

Simple English uses **checkpoint 193 (epoch 1), job 2527052**. Training completed
579 steps over three epochs; the adapter at the run root is the final epoch,
not the selected checkpoint. [selection.json](simple_english/selection.json)
records the selection, settings, weight checksum and limitations. Cluster paths
and API names retain their existing names; no remote weights have been renamed.

## Data and purpose

**Conspiracy:** a fictional conspiracy persona, prepared from
`IgYahiko/Conspiracy_Theory_Dataset_100k` (2,000 sampled rows, 95/5 split), then
rewritten and extended with follow-ups using Qwen3.5 9B. Prepared JSONL files
are generated locally and are not committed. The deployed adapter is the run-root
adapter; an exact training job/checkpoint identifier is not recorded here.

**Simple English:** English for adult beginners: short sentences, familiar words,
explained terms, concrete steps/examples, and important qualifications preserved.
The goal is understandable answers, not minimum word count, translation, additional
knowledge or better tool use. Factual mistakes remain possible; checkpoint 193's
whole-grain definition was incorrect in the saved validation answers.

The final data has **2,000 train / 100 validation / 100 test dialogues**, including
3,085 training answers. Sources are HuggingFaceTB everyday conversations and
OpenAssistant, with AI-edited assistant answers; see [NOTICE](simple_english/data/NOTICE),
[LICENSE](simple_english/data/LICENSE) and per-row provenance. Split files and
provenance retain their original bytes. The validator checks the final release
without old dataset directories. Earlier editorial revisions remain in Git history.

[validation_comparison.jsonl](simple_english/validation_comparison.jsonl) retains
only the base and selected adapter: 19 answers from 10 development validation
dialogues, generated with the 4-bit training model. This is not an independent
benchmark or a fresh test of the vLLM server. The reserved test split is separate.

## Commands

CPU-only data checks (standard Python; no downloads):

```bash
python3 -m training.simple_english.validate
```

On the cluster, export `HF_HOME` and `UV_PROJECT_ENVIRONMENT` to project storage
before using `uv`; the job scripts set these automatically. Create
`/sc/projects/sci-lippert/intelligent-agents/project_matthias_max/logs/training`
before submitting jobs. Submit from the repository root, or export `REPO_DIR`.

```bash
# Conspiracy: prepare seeds, augment them, then train after augmentation finishes.
uv run --project training python -m training.conspiracy.prepare
sbatch training/conspiracy/run-augment.sbatch
sbatch training/conspiracy/run-training.sbatch

# Simple English: tokenizer-only preflight, then a fresh training run.
uv run --project training python -m training.simple_english.train --check
sbatch training/simple_english/run-training.sbatch

# Optional held-out comparison on a GPU; omit --run to check inputs only.
uv run --project training python -m training.simple_english.evaluate --run
```

Training jobs write to fresh job-specific adapter directories. The Simple English
preflight reads only tokenizer files from the existing Conspiracy directory;
training initializes the pinned base model, not the Conspiracy adapter.
Serving and SSH tunnel instructions are in the [main README](../README.md).

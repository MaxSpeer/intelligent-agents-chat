# Clear English v2 — 2,000 training dialogues

English conversations for adult beginners, revised to favour **short sentences, familiar words, explained terms, concrete examples, and preserved meaning**. Whole answers may be longer when that makes the explanation easier to understand.

This is a separate derivative of `../plain_english_2k_simplified_v1`. It contains 2,000 training dialogues, 100 validation dialogues, and 100 existing test dialogues. The training format remains one JSON object per line with `id` and `messages`; messages contain `role` and `content`.

## Editorial work

Every one of the **3,085 training answers** and **160 validation answers** has an explicit editorial decision. Three Codex editors handled nonoverlapping training ranges, reading each assigned dialogue in context. A fourth editor revised the remaining 40 training dialogues and all validation answers, assembled the data, and reviewed selected training outputs independently. Independent peer checks covered additional dialogue samples and all 40 training dialogues from the fourth editor. Already clear answers and brief acknowledgements may be kept. Exact decisions, original-answer hashes, reasons, and any targeted source checks are recorded in `decisions/` and `answer_changes.jsonl`.

This is AI editorial review, not human expert review or a complete external fact check. Targeted factual and contextual repairs are recorded in notes and source links. Style metrics describe the text; they do not establish a CEFR level, human comprehension, or improved model behaviour.

See [EXAMPLES.md](EXAMPLES.md) for actual revised training targets and [style_comparison.json](style_comparison.json) for measured counts. These are dataset answers, not outputs from a newly trained adapter.

## What is preserved

- All conversation IDs, split assignments, message order, roles, and every user/system message.
- The neutral system message. The model must learn this style from assistant targets; an added style prompt is a separate evaluation condition.
- Existing `test.jsonl` and `test_prompts.jsonl`, copied byte for byte. Their 198 answers are labelled `frozen_holdout`, not as newly reviewed or rewritten.
- Original source lineage and licence notices. Parent files are checked against the snapshot made before editing.

The initial source and grouping come from the earlier 2k release. The parent includes synthetic conversations and the earlier pilot. Preserving those groups does **not** establish semantic independence: similar topics occur across splits. The frozen test targets are inherited AI reference answers, not independent human judgments of the new style. Do not train on validation/test files or use the test set to select a checkpoint.

## Reproduce and verify

From the repository root, using Python 3.11 or newer:

```bash
python3 training/datasets/plain_english_clear_v2/validate.py
```

To check the exact Qwen tokenisation and assistant-only labels as well:

```bash
python3 training/datasets/plain_english_clear_v2/validate.py \
  --tokenizer /path/to/the/verified/qwen-tokenizer
```

The second command needs Transformers and the pinned local tokenizer files. It loads no model weights, uses no model API, and starts no training. It verifies that every context token is masked, all assistant text and end markers are supervised, and no example is truncated at the 1,024-token limit.

`build.py` rebuilds this revision's derived files from the frozen parent and complete decision shards. It refuses missing decisions, stale original hashes, duplicate decisions, or test edits. Do not rebuild a version after using it in a training run; make another version instead. Rerun validation after any development change.

## Training entry point

The dedicated wrapper is `training/train_plain_english_clear_v2.py`. It uses the existing QLoRA settings with a fresh pinned `Qwen/Qwen3-8B` base and a separate output directory:

```text
/sc/projects/sci-lippert/intelligent-agents/project_matthias_max/adapters/qwen3-8b-plain-english-clear-v2
```

From the `training` directory, this command checks the configuration only:

```bash
"$UV_PROJECT_ENVIRONMENT/bin/python" train_plain_english_clear_v2.py --check
```

The user must explicitly pass `--run` inside their own suitable Slurm GPU allocation to train. The preparation process does not allocate a GPU or start training. The runner rejects an existing output directory and never loads a previous adapter as the starting model.

## Files

- `train.jsonl`, `validation.jsonl`, `test.jsonl`, `test_prompts.jsonl`: model data and reserved test prompts.
- `decisions/*.jsonl`: every training/validation editorial decision.
- `answer_changes.jsonl`: all original/revised pairs, including frozen test answers.
- `provenance.jsonl`, `parent_snapshot.json`, `parent_manifest.json`, `LICENSE`, `NOTICE`: source lineage and integrity.
- `style-guide.md`, `rewrite-prompt.txt`: the agreed editorial target captured for this release.
- `style_comparison.json`, `quality_review.json`, `validation_report.json`: descriptive style counts, QA scope and findings, and reproducible integrity/token checks.
- `manifest.json`: model/tokenizer identity, split counts, review scope and file hashes.
- `build.py`, `editor.py`, `validate.py`: local assembly, decision recording and validation.

No model improvement is claimed from data preparation alone. Choose a checkpoint using validation answers, then evaluate the chosen setup on reserved questions. Judge clarity, necessary term explanations, concrete help, and correct core meaning. Report length as a separate descriptive measurement.

# Plain English — separate simplified revision

A new, local version of `../plain_english_2k` for **clear English for adult beginners**. The parent dataset is unchanged. This revision changes assistant targets only. It keeps all 2,000 training dialogues, 100 validation dialogues, 100 test dialogues, their order and IDs, and every user and system message. There are still 3,085 training answers and 3,443 answers across all splits.

## What changed

- **654 assistant answers were rewritten individually**, including explanations of energy, DNA, databases, weather, and everyday tasks.
- **90 further answers received small phrase simplifications**, such as “approximately” → “about”.
- **2,699 answers remain unchanged.** Clear answers do not need changes just to make the dataset look different.
- Necessary terms are explained, long sentences are split, abstract phrases become concrete, and empty praise or repeated wording is reduced. Essential distinctions, qualifications and numerical examples are retained where relevant; optional detail is sometimes shortened.
- One standalone test question, “What can we do to help?”, supplied no topic. Its target now asks “What would you like to help with?” instead of assuming climate change. This is a context repair, recorded explicitly in the decision log.

Codex read 716 selected dialogues (1,116 assistant answers), prioritizing science, technical language, long sentences and the original pilot. A final wording audit added 17 individual answer edits and reviewed the remaining phrase-change outputs. **This is targeted AI editorial work, not a new human review of all 3,443 answers or complete factual verification.** The source is mostly synthetic. Some wording and factual limitations may remain in unchanged material. No CEFR level or model improvement has been established.

All 3,443 answer examples passed the exact local tokenizer check: every context token is masked, all target text and end markers are supervised, maximum length 321 tokens, no truncation. The original parent files and test prompts were verified unchanged. The four focused protection tests passed.

See [EXAMPLES.md](EXAMPLES.md) for actual before/after examples and [style_comparison.json](style_comparison.json) for descriptive counts. Adding a useful term explanation may increase length. Sentence length alone does not establish readability.

## Files and lineage

- `train.jsonl`, `validation.jsonl`, `test.jsonl`: the same `id` / `messages` training format as the parent.
- `test_prompts.jsonl`: unchanged initial test questions, with no target answers.
- `manual_rewrites.jsonl`: exact editorial decisions and original-answer hashes. Entries with `queue_index: null` are the 17 final individual answer edits.
- `review_queue.jsonl`: selected original dialogues and selection reasons, in original order.
- `answer_changes.jsonl`: every original/revised assistant answer, including unchanged answers.
- `parent_snapshot.json`: hashes of the pre-existing parent files, checked before and after preparation.
- `provenance.jsonl`: original provenance under `source_lineage`, plus this revision's answer actions. Historical labels such as `legacy_unchanged` describe the parent, not the new text.
- `parent_manifest.json`, `LICENSE`, `NOTICE`: original source metadata and credits, with this revision's changes noted.
- `manifest.json`: new counts and integrity hashes.
- `validate.py`, `validation_report.json`: structure, original preservation, lineage and optional exact Qwen tokenizer checks.
- `EVALUATION.md`: concrete later comparison prompts and criteria. No model outputs were generated during this preparation.

All source grouping and split assignments are inherited. This does **not** make the splits semantically independent: related topics already occur across splits. Test targets were edited during dataset preparation and are not independent human ratings. Do not train on validation or test data, or choose an epoch using test generations.

## Verify without training

From the repository root:

```bash
python3 -B training/datasets/plain_english_2k_simplified_v1/validate.py
```

To also check every training label using the pinned local tokenizer:

```bash
python3 -B training/datasets/plain_english_2k_simplified_v1/validate.py \
  --tokenizer /path/to/verified/qwen-tokenizer
```

The validator needs the unchanged sibling `plain_english_2k` for source-lineage checks. The optional tokenizer check needs Transformers and loads local tokenizer files only. It checks all assistant answers and their end markers, masks every context token, and rejects truncation. It loads no model weights and starts no training.

`revise.py` rebuilds only this new directory's derived data from the sibling parent and local decisions. `release.py` regenerates only this directory's metadata. Do not use them to experiment on a version already used by a run; make another version instead. Run validation again after rebuilding. The four focused tests in `test_revision.py` check protections against changing questions/roles, adding template markers, or damaging text with phrase rules.

**This version has not been copied to the cluster or selected by a training run.** `train_plain_english_2k.py` still points at the original dataset. Using this new version later needs an explicit dataset-path choice and a separate run output directory. No existing runner, adapter, configuration or active data was changed.

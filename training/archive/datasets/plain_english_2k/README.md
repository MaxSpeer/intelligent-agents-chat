# Plain English: reviewed 2k expansion

English dialogues for a **clear, concise style for adult beginners**. Explain
necessary terms in ordinary words and preserve the meaning of the answer. The
same neutral system prompt is used as in the pilot, so the style must be learned
from assistant answers rather than an explicit simplification instruction.

**2,000 training dialogue examples, 100 validation examples, 100 test examples.**

| Split | Dialogue examples | Source conversations | Assistant answers | Multi-turn dialogues |
|---|---:|---:|---:|---:|
| Train | 2,000 | 1,393 | 3,085 | 662 |
| Validation | 100 | 65 | 160 | 37 |
| Test | 100 | 81 | 198 | 56 |

Training answers average **30.86 words** (a descriptive statistic, not evidence
that a trained model will improve). All 3,443 supervised answer examples passed
the exact tokenizer/mask checks; maximum length 321 tokens, no truncation.
Six focused CPU tests and the default no-model preflight passed.

The same statistics are recorded in `manifest.json` and `validation_report.json`. Do not equate 2,000 examples with 2,000 independent
source conversations. The trainer makes one masked training example per
assistant answer, so optimizer steps depend on assistant turns, not JSONL lines.

## Origin and review

Most new examples come from
[HuggingFaceTB/everyday-conversations-llama3.1-2k](https://huggingface.co/datasets/HuggingFaceTB/everyday-conversations-llama3.1-2k/tree/14f543216b9ba42b6b951dc5bd199460d193b162),
pinned to `14f543216b9ba42b6b951dc5bd199460d193b162`. The source contains
2,379 English conversations (2,260 train_sft and 119 test_sft), generated with
Llama-3.1-70B-Instruct. Its dataset card declares Apache-2.0. These are synthetic
conversations, not human demonstrations.

Codex read every source conversation. `review_decisions.jsonl` records keep,
edit, or exclude, the exact assistant rewrites, approved independent questions,
and primary-source links where a fact was checked. Unverified live/local
information, fabricated store capabilities, unsuitable personal advice and
irreparable content were excluded. Some factual explanations and word choices
were corrected. This is **AI editorial review with selective fact checking,
not complete human expert verification**; residual errors are possible.

The original 80 training dialogues from `../plain_english` are retained exactly.
They originated from OpenAssistant/oasst1, with human user questions and earlier
Codex-rewritten assistant answers. Their previous provenance and review records
are included. Old validation/test dialogues are only included as protected
reference records and are never added to the new training/validation/test files.
See `NOTICE` and `LICENSE` for source credit and licensing.

## What counts as one example

Greeting exchanges and short closing acknowledgements are removed. A source is
used either as one substantive multi-turn dialogue **or** as independently
reviewed question/answer pairs. Both halves of an extracted pair must make sense
without earlier messages. An independent sample/context audit led to additional
corrections recorded in `editorial_overrides.jsonl`. As a conservative extra gate,
a follow-up containing it/its/they/them/these/those/this is not extracted by itself. A deterministic 20% hash-based selection of sources eligible for extraction is
reserved for full dialogues; the final
counts also depend on filtering and split selection. One source assistant turn
is never supervised in more than one dataset example. Unselected follow-ups are
not silently converted into standalone examples.

All retained user messages are verbatim. There are no duplicated rows, canned
question paraphrases, or repeated source answers added to reach the target size.
The new answers cover everyday questions and elementary educational topics;
there are also conversational answers and clarification questions. This is a
broader style experiment than the original explanation-only pilot.

## Splits and test protection

All examples from one source stay together. Recorded narrow topics and lexical
near-question groups also stay within one split. Groups are connected using
exact normalized substantive questions (including short questions), word 1–2-gram TF-IDF cosine
similarity >=0.84, or character 3–5-gram cosine similarity >=0.90. The near-match checks require at least six words; generic shorter
follow-ups are excluded from exact grouping using a recorded word list. `near_question_groups.jsonl` records the
near-match edges. Exact initial questions are unique across all final examples. Repeated normalized
assistant answers of at least eight words are removed; short generic answers may recur.

The upstream test partition is reserved for the new test split. Any linked source
train records stay with that test group. The pilot's 20 held-out dialogues are
protected using all their exact user messages and a conservative topic keyword
filter, recorded in `build.py`. These checks reduce leakage but **do not establish
perfect semantic independence**. Base-model pretraining exposure is unknown.

The new 100 test examples have no generated model outputs from our previous
adapter evaluations. Do not use their answers to choose an epoch or tune training
settings. Reference answers were inspected during dataset preparation; they are
not independent human judgments. Compare correctness and completeness as well as
wording, sentence length, technical terms and total answer length.

## Files and checks

- `train.jsonl`, `validation.jsonl`, `test.jsonl`: `id` and ChatML-style `messages`.
- `test_prompts.jsonl`: only the system message and initial user question.
- `provenance.jsonl`: source, revision, original message indices, extraction and split group per example.
- `source_records.jsonl`: original everyday conversations, including exclusions, for audit and rebuilding.
- `review_decisions.jsonl`: all source decisions and exact assistant edits.
- `legacy_dialogues.jsonl`, `legacy_provenance.jsonl`: unchanged pilot records.
- `manifest.json`: counts, upstream hashes, tokenizer hashes and local file hashes.
- `review_report.json`: filtering and extraction counts and review limits.
- `validation_sample_ids.json`: ten fixed validation dialogues for generation; all 100 are used for validation loss.
- `validation_report.json`: generated structure/tokenizer check report.

Structure checks use only Python's standard library:

```bash
python3 training/datasets/plain_english_2k/validate.py
```

For exact token labels, add `--tokenizer /path/to/verified/qwen-tokenizer` in an
environment with Transformers. This uses the production `chat_examples.py`
builder for every assistant answer, verifies that all context is masked and all
answer text plus the end marker is supervised, and rejects truncation. It loads
no model weights and starts no training.

`build.py` rebuilds deterministically from bundled source/review records when
run without `--cache`; it needs scikit-learn 1.9.0 and the unchanged sibling pilot.
The original preparation also checked the downloaded Parquet files against their
hashes and compared source messages and topic fields before assembly. That mode
uses `--cache /path/to/preparation-cache` and pyarrow 25.0.1. It refuses incomplete
review coverage. It rewrites this derived dataset, so preserve the released
version before experimenting with different rules.

See `../../plain-english-2k.md` for the new manual training workflow. Preparation
has not trained an adapter, and more samples alone do not prove improvement.

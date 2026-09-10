# Plain English explanation pilot

100 English conversations for teaching a clear, concise explanation style to
adult beginners. Codex rewrote the assistant answers from selected OpenAssistant
conversations. All user questions and retained follow-up questions are unchanged.
This is a small starter dataset for a style experiment.

## Files and format

| File | Conversations | Purpose |
| --- | ---: | --- |
| `train.jsonl` | 80 | Update the LoRA adapter |
| `validation.jsonl` | 10 | Monitor loss and choose training settings |
| `test.jsonl` | 10 | Reserved reference conversations for final evaluation |
| `test_prompts.jsonl` | 10 | Initial test questions, with no reference answers |

There are 118 assistant answers in total; 18 conversations include one follow-up.
Each JSONL line has this structure, which the existing `train_lora.py` accepts:

```json
{"id":"plain-en-0001","messages":[{"role":"system","content":"You are a helpful assistant. Answer the user's question directly and clearly."},{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
```

Follow-ups add another `user` / `assistant` pair. The trainer reads `messages` and
ignores `id`. It learns from every assistant turn. The neutral system message is
the same in every split: the simplified style is demonstrated in the answers.

- `provenance.jsonl` maps each ID to its split, source message IDs, original
  messages, topic, and rewrite notes. Use it for inspection, never as training input.
- `manifest.json` records the source revision, split policy, tokenizer snapshot,
  and file checksums.
- `review_report.json` documents a second Codex-agent review and its resolved correction.
- `validation_report.json` records the original preparation checks. Its old
  length/count-only mask check did not detect the multi-turn boundary bug.
  The current `validate.py` checks the exact text of every supervised answer.
- `LICENSE` and `NOTICE` preserve the upstream license and attribution.

## Source and preparation

Source: [OpenAssistant/oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1),
revision `fdf72ae0827c1cda404aff25b6603abec9e3399b`, file
`2023-04-12_oasst_ready.messages.jsonl.gz`, licensed under Apache-2.0.
The source file URL and SHA-256 are recorded in the manifest.

The selected original messages are English. Each selected source answer has
rank 0 among its alternatives and a source quality score of at least 0.75.
These community ratings helped selection; they do not guarantee correctness.
The topics include science, mathematics, computing, language, learning, arts,
and everyday explanations. This is a curated sample, not a representative
sample of the whole source dataset.

Codex rewrote every assistant answer to answer directly, use ordinary words,
explain necessary technical terms, and keep a useful example or calculation.
The target is understandable adult English, not a certified reading level.
Length and formatting vary with the question. Original user wording, including
typos and requested explanation levels, is preserved exactly. The 18 follow-up
exchanges are drawn from the same source conversation branches; no new user
questions were generated.

A second Codex agent reviewed every selected conversation against its source,
including follow-up coherence and factual issues. The review identified one
additional minor generalization about cheese production, which was corrected.
Other corrections made during rewriting are recorded per conversation. This is
model-assisted review, with no human expert review or measured training result.

## Split and evaluation policy

Conversation trees never cross splits. Closely duplicated topics identified
during review, including the two Pythagorean and equal-mass examples, are kept
within the training split. Validation and test topics were explicitly selected
before training; the topic lists and deterministic shuffle seed are in the
manifest. Normalized initial questions are unique across all 100 conversations.
Broad subject areas occur in multiple splits, while recorded narrow topic
groups do not.

Use validation loss during training. For the final comparison, generate answers
to `test_prompts.jsonl` with the base model and the new adapter under the same
settings. Compare whether the answers:

1. Answer the actual question correctly and retain important qualifications.
2. Explain unfamiliar terms in understandable English.
3. Use an example or calculation where helpful.
4. Stay concise without dropping necessary information.

Exact wording need not match `test.jsonl`. Its answers are reference examples,
not the only valid answers. For the two test conversations with follow-ups,
ask the retained follow-up after the model's own first answer; do not insert the
reference first answer into a free-generation test. Keep test answers out of
training and use validation, not test performance, to tune settings.

Ten final test questions support an exploratory comparison, not a robust
benchmark. The source data may have appeared in the base model's pretraining;
these are held out only from this fine-tuning dataset.

## Training status and corrected preparation

The first Plain-English adapter was trained on 2026-09-08. Its initial comparison
did not show reliable simplification; see the saved evaluation report under
`training/evaluations/plain_english/REPORT.md`. The manifest's `training_started`
field describes the original preparation snapshot, not current run status.

For the corrected run, use `training/train_plain_english_v2.py` and the manual
instructions in `training/plain-english-v2.md`. Without `--run`, the script only
checks the configuration and tokenizer; it does not load model weights or train.

Use the pinned base model `Qwen/Qwen3-8B`, revision
`b968826d9c46dd6066d109eabc6255188de91218`, for a fresh style adapter.
The separate output directory keeps this run distinct from the earlier adapter.
Do not run the conspiracy data preparation scripts for this dataset.

`chat_examples.py` creates one example per assistant answer, with all previous
messages as masked context. This avoids Qwen3's position-dependent template
shifting a span into the following user message. The target is exactly the
answer and its end marker; the empty thinking prefix is context. Oversized
examples fail explicitly instead of losing their end marker through truncation.

The source JSONL files are unchanged: 80 training conversations yield 93 answer
examples; 10 validation conversations yield 13 examples; 10 test conversations
yield 12 examples. With batch size 1, accumulation 16, and three epochs, v2 has
18 optimizer steps (six per epoch), versus 15 in v1. V2 saves every epoch and
retains all three checkpoints for review of generated validation answers.

## Recheck the files

From the repository root, Python 3.10 or newer is sufficient for structure,
checksums, source-chain preservation, and split checks:

```bash
python3 training/datasets/plain_english/validate.py
```

To repeat the exact template, length, and assistant-mask checks, use the locked
training environment with Transformers and Jinja2, then pass the tokenizer
snapshot directory recorded in the manifest:

```bash
python3 training/datasets/plain_english/validate.py --tokenizer /path/to/qwen-tokenizer
```

This loads tokenizer files only. The check calls the production example builder,
then decodes its actual non-masked labels and compares them with the original
assistant answer plus `<|im_end|>`. It verifies that all context is masked and
no role header is supervised. Tokenizer hashes must match the recorded snapshot.
No model weights or GPU are required. The old `validation_report.json` is retained
as historical evidence; the v2 preflight prints the current verification results.

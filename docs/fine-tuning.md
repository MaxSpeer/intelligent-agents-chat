# Two LoRA fine-tunings

[Back to the overview](../README.md#feature-overview)

Our two fine-tunings use **separate LoRA adapters on Qwen3 8B**. 

Both adapters share one base model and vLLM instance on port `8001`. The application selects
the base or an adapter by model name, so switching does not require another full model server.
Training runs separately in its own `uv` environment under `training/`.

We chose Qwen3 8B after an initial Qwen3.5 adapter worked in Transformers/PEFT but had no effect
in our tested vLLM setup (seems to be a bug in vLLM, see https://github.com/vllm-project/vllm/issues/49354).
The current selector exposes the Qwen3 8B base and both adapters. The Qwen3.5 9B profile used for
earlier agent experiments remains available in the code but is inactive in the selector.


## Conspiracy adapter

A persona experiment using
[`IgYahiko/Conspiracy_Theory_Dataset_100k`](https://huggingface.co/datasets/IgYahiko/Conspiracy_Theory_Dataset_100k)
(`question`/`answer` columns) as SFT data.

**Preprocessing.** Each row becomes a two-turn chat example (system prompt, user question,
assistant answer), shuffled and split into train/validation, keeping the first 2,000 rows for a
fast first run rather than the full ~100k. A second pass then rewrites each answer to be longer
and differently phrased, and synthesizes one extra follow-up user/assistant turn per example --
both aimed at the overfitting problem below, not just at having more data. Rewriting a
conspiracy-themed answer with a safety-aligned instruct model tends to break character and hedge
("this is a conspiracy theory, not evidence..."); we reduce that by framing the rewrite/follow-up
prompts as writing dialogue for a fully-committed fictional character rather than "answer as the
assistant", plus a keyword-based filter that retries generation (or falls back to the original,
unmodified answer) whenever the output still hedges. Only assistant responses contribute to the
training loss.

**Overfitting.** The first adapter (higher rank, all-linear target modules, a fixed epoch count)
memorized the training answers so hard it replied with one fixed canned string per topic
regardless of the actual question asked. Training loss had already collapsed within a fraction of
one epoch. We reduced adapter capacity and learning rate, restricted the target modules to the
attention projections, and replaced the fixed epoch count with early stopping on validation loss.

## Simple English adapter

The accepted dataset, `plain_english_clear_v2`, contains **2,000 training, 100 validation, and
100 test dialogues**. The target style uses short sentences and familiar words, with technical
terms explained in simple language. The [training overview](../training/README.md)
documents the sources, answer revisions, licences, and quality checks.

Training supervises each assistant answer separately and masks preceding context, using the
non-thinking chat template. The run starts from the pinned Qwen3 8B base; it does not continue
from the Conspiracy adapter. Each saved epoch is compared against base-model validation answers.

The accepted run is `qwen3-8b-plain-english-clear-v2-retry1`. The chat serves **checkpoint 193**,
chosen after [reviewing the validation answers](../training/simple_english/validation_comparison.jsonl),
under the API name `plain-english-clear-v2` and display label **Qwen3 8B (Simple English)**.
The final epoch remains available in the run directory; it is not the selected serving adapter.
The selected run and weight checksum are recorded in
[selection.json](../training/simple_english/selection.json); earlier repository artifacts remain in Git history.

Training commands and experiment details are in [training/README.md](../training/README.md).
The shared serving setup is shown in the [architecture overview](../README.md#architecture).

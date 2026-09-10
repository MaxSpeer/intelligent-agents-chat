# Active model artifacts and archive

Cluster project root:

```text
/sc/projects/sci-lippert/intelligent-agents/project_matthias_max
```

Paths below are relative to that root. The active base model is the pinned
`Qwen/Qwen3-8B`; shared model files remain under `models/huggingface/`.

| Purpose | Path |
| --- | --- |
| Conspiracy adapter loaded by vLLM | `adapters/qwen3-8b-conspiracy/` |
| Simple English adapter loaded by vLLM | `adapters/qwen3-8b-plain-english-clear-v2-retry1/checkpoint-193/` |
| Complete retained Simple English run | `adapters/qwen3-8b-plain-english-clear-v2-retry1/` |
| Accepted dataset: 2,000 train / 100 validation / 100 test | `code/intelligent-agents-chat/training/datasets/plain_english_clear_v2/` |
| Exact run inputs and original code | `adapters/qwen3-8b-plain-english-clear-v2-retry1/run-inputs/` |
| Saved validation answers | `adapters/qwen3-8b-plain-english-clear-v2-retry1/validation_samples/` |
| Final training log | `logs/training/clear-v2-retry1-2527052.log` |
| Serving log from that allocation | `logs/vllm/vllm-clear-english-2527052.log` |
| Historical weights, optimizer states, and logs | `archive/2026-09-10-simple-english/` |
| Historical source datasets, comparisons, and notes | `code/intelligent-agents-chat/training/archive/` |

The complete last run keeps checkpoints 193, 386, and 579, its final adapter,
tokenizer files, and recorded inputs. The chat continues to use **checkpoint 193**.
The Conspiracy root weights are unchanged; its historical optimizer checkpoints
are archived. No model cache, environment, or container image was moved.

The Simple English display name retains API key `plain-english-clear-v2` for
compatibility. [simple-english-selection.json](simple-english-selection.json)
records the selected checkpoint and weight checksum. A copy is stored as
`selection.json` in the retained run directory.

## Archive integrity

The 2026-09-10 cleanup moved 23 directory/file groups containing approximately
1.62 GB, including six superseded or abandoned adapter/run directories. Every
moved file was hashed before and after moving. The two active adapter checksums
were also checked before and after cleanup.

`archive/2026-09-10-simple-english/manifest.json` records exact source and
destination paths, file sizes, SHA-256 values, and the necessary compatibility
links. The same record is versioned as
[artifact-archive-20260910.json](artifact-archive-20260910.json).

The three old dataset names in `training/datasets/` are now relative symbolic
links into `training/archive/datasets/`. The last release requires these frozen
parents for source-lineage validation. All dataset bytes and manifests are
unchanged. The archive's `chat_examples.py` link preserves its validator imports.

## Recovery

Choose an individual group from the manifest, verify the archived file checksums,
and move it from `destination` back to `source` only when the original location
is unused. For a dataset, its original location contains the documented
compatibility link; remove only that exact link before restoring the directory.
Never overwrite a newer run, dataset, or log. No permanent deletion is needed
to use or inspect this archive.

New inference serving still needs the user's GPU allocation and server start.
See [plain-english-chat.md](plain-english-chat.md). Archiving files does not
start or submit any cluster job.

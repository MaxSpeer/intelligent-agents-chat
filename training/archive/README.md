# Historical training material

Only `../datasets/plain_english_clear_v2` and the completed
`qwen3-8b-plain-english-clear-v2-retry1` run are the current Simple English release.

- `datasets/`: the unchanged pilot, 2k, and simplified parent releases.
- `evaluations/`: comparisons and reports for superseded runs.
- `notes/`: historical run instructions, preserved as written. Relative paths in
  those notes originally referred to the `training/` directory.
- `experiments/`: the original manually created pilot wrapper, when present.

The three old names in `../datasets/` are relative symbolic links into this archive.
The final release's validator checks the parent files and their complete source
lineage. These links, plus `chat_examples.py`, preserve those original imports
without changing any dataset bytes or recorded file hashes. They do not enable
old model choices or load archived adapters into vLLM.

The reusable Python training/evaluation modules remain in `training/`, including
`train_plain_english_2k.py`, which is also the shared implementation used by the
current entry point. Completed runs keep their own original code snapshots.

Large model weights, optimizer states, and old cluster logs are archived outside
Git at `$PROJECT_ROOT/archive/2026-09-10-simple-english/`. Its `manifest.json`
records every original/destination path and each file's SHA-256. See
[`cluster/model-artifacts.md`](../../cluster/model-artifacts.md) for active paths
and recovery instructions.

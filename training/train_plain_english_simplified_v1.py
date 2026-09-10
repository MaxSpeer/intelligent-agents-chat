#!/usr/bin/env python3
"""Use the reviewed simplified-v1 data with the existing manual QLoRA runner.

Default/--check validates only. The user must explicitly start --run inside
their own GPU allocation. This initializes a fresh pinned base model.
"""

import train_plain_english_2k as runner


DATASET = runner.HERE / "datasets" / "plain_english_2k_simplified_v1"
OUTPUT = runner.PROJECT / "adapters" / "qwen3-8b-plain-english-2k-simplified-v1"
EXTRA_CODE_FILES = (
    "train_plain_english_simplified_v1.py",
    "datasets/plain_english_2k_simplified_v1/validate.py",
    "datasets/plain_english_2k_simplified_v1/revise.py",
)


def main():
    previous = runner.DATA, runner.OUTPUT, runner.CODE_FILES
    try:
        runner.DATA = DATASET
        runner.OUTPUT = OUTPUT
        runner.CODE_FILES = tuple(dict.fromkeys((*runner.CODE_FILES, *EXTRA_CODE_FILES)))
        print(f"Dataset: {DATASET}", flush=True)
        runner.main()
    finally:
        runner.DATA, runner.OUTPUT, runner.CODE_FILES = previous


if __name__ == "__main__":
    main()

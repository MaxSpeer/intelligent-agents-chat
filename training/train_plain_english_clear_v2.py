#!/usr/bin/env python3
"""Check the clear-v2 dataset; train only with the user's explicit --run."""
import train_plain_english_2k as runner

DATASET = runner.HERE / "datasets" / "plain_english_clear_v2"
OUTPUT = runner.PROJECT / "adapters" / "qwen3-8b-plain-english-clear-v2"
EXTRA_CODE_FILES = (
    "train_plain_english_clear_v2.py",
    "datasets/plain_english_clear_v2/build.py",
    "datasets/plain_english_clear_v2/validate.py",
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

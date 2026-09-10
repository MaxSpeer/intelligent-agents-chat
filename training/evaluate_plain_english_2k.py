#!/usr/bin/env python3
"""Compare a selected 2k adapter with the base on the 100 fresh test dialogues.

Without --run, check the dataset and adapter files only. With --run, reuse the
existing evaluator inside the user's own Slurm GPU allocation. No job is submitted.
Pass --adapter to select the checkpoint chosen using validation results.
"""

from pathlib import Path

import evaluate_plain_english as evaluator


DATA = Path(__file__).resolve().parent / "datasets" / "plain_english_2k"
DEFAULT_ADAPTER = evaluator.PROJECT / "adapters" / "qwen3-8b-plain-english-2k"


def main():
    previous_data = evaluator.DATA
    previous_adapter = evaluator.DEFAULT_ADAPTER
    try:
        evaluator.DATA = DATA
        evaluator.DEFAULT_ADAPTER = DEFAULT_ADAPTER
        evaluator.main()
    finally:
        evaluator.DATA = previous_data
        evaluator.DEFAULT_ADAPTER = previous_adapter


if __name__ == "__main__":
    main()

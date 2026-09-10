#!/usr/bin/env python3
"""Prepare the Conspiracy source dataset as UTF-8 chat JSONL."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

os.environ.setdefault(
    "HF_HOME",
    "/sc/projects/sci-lippert/intelligent-agents/project_matthias_max/models/huggingface",
)

from datasets import load_dataset  # noqa: E402


DATASET_REPO = "IgYahiko/Conspiracy_Theory_Dataset_100k"
SPLIT: str | None = None  # None = first split reported by the dataset (usually "train")

LIMIT: int | None = 2000

VAL_FRACTION = 0.05  # fraction of the kept examples reserved for validation

SYSTEM_PROMPT = "You are a helpful assistant. Answer the user's question directly and clearly."

SEED = 42
OUTPUT_DIR = Path(__file__).resolve().parent / "data"


def to_messages(question: str, answer: str, system_prompt: str) -> list[dict]:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question})
    messages.append({"role": "assistant", "content": answer})
    return messages


def main() -> None:
    print(f"Loading dataset: {DATASET_REPO}")
    dataset_dict = load_dataset(DATASET_REPO)
    split = SPLIT or next(iter(dataset_dict.keys()))
    dataset = dataset_dict[split]
    print(f"Using split '{split}' with {len(dataset)} rows.")

    missing = {"question", "answer"} - set(dataset.column_names)
    if missing:
        raise SystemExit(
            f"Dataset is missing expected column(s) {sorted(missing)}. "
            f"Available columns: {dataset.column_names}"
        )

    rows = [
        row for row in dataset if str(row["question"]).strip() and str(row["answer"]).strip()
    ]
    dropped = len(dataset) - len(rows)
    if dropped:
        print(f"Dropped {dropped} row(s) with an empty question or answer.")

    rng = random.Random(SEED)
    rng.shuffle(rows)

    if LIMIT is not None:
        rows = rows[:LIMIT]
    print(f"Keeping {len(rows)} examples.")

    val_size = max(1, int(len(rows) * VAL_FRACTION)) if rows else 0
    val_rows, train_rows = rows[:val_size], rows[val_size:]
    print(f"Split: {len(train_rows)} train / {len(val_rows)} val.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, split_rows in (("source_train", train_rows), ("source_validation", val_rows)):
        out_path = OUTPUT_DIR / f"{name}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for row in split_rows:
                messages = to_messages(
                    str(row["question"]).strip(),
                    str(row["answer"]).strip(),
                    SYSTEM_PROMPT,
                )
                f.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
        print(f"Wrote {out_path} ({len(split_rows)} examples).")


if __name__ == "__main__":
    main()

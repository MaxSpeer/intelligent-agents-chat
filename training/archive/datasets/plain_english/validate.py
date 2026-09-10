#!/usr/bin/env python3
"""Check this dataset without loading model weights or starting training.

Run with Python's standard library for structure/provenance checks. Optionally
pass --tokenizer /path/to/local/qwen-tokenizer for exact template and supervised
token checks using the production example builder. This mode needs transformers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from chat_examples import build_examples  # noqa: E402


def validate_masks(rows, tokenizer, max_seq_len):
    """Independently decode the actual labels produced by the trainer's builder."""
    lengths, trainable_counts = [], []
    for row in rows:
        examples = build_examples(tokenizer, row["messages"], max_seq_len)
        answers = [m for m in row["messages"] if m["role"] == "assistant"]
        require(len(examples) == len(answers), f"Missing assistant answer: {row['id']}")
        for example, answer in zip(examples, answers, strict=True):
            labels = example.labels
            require(len(labels) == len(example.input_ids), row["id"])
            start = example.prompt_length
            require(all(x == -100 for x in labels[:start]), f"Unmasked context: {row['id']}")
            require(labels[start:] == example.input_ids[start:], row["id"])
            actual = tokenizer.decode(
                [label for label in labels if label != -100],
                skip_special_tokens=False, clean_up_tokenization_spaces=False,
            )
            expected = answer["content"].lstrip("\n") + "<|im_end|>\n"
            require(actual == expected, f"Supervised text differs from target: {row['id']}")
            require("<|im_start|>" not in actual, f"Role header in target: {row['id']}")
            lengths.append(len(example.input_ids))
            trainable_counts.append(len(labels) - start)
    return {
        "tokenizer_validation": "passed",
        "masking_version": "one_answer_per_example_non_thinking_v2",
        "answer_examples": len(lengths),
        "all_supervised_text_matches_reference_answer_and_end_marker": True,
        "all_context_tokens_masked": True,
        "min_tokens": min(lengths), "max_tokens": max(lengths),
        "mean_tokens": round(sum(lengths) / len(lengths), 2),
        "min_assistant_training_tokens": min(trainable_counts),
        "truncated_conversations": 0,
    }


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"{path.name}:{number}: blank JSONL row")
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path.name}:{number}: expected an object")
        rows.append(row)
    return rows


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def normalize(text: str) -> str:
    return re.sub(r"\W+", " ", text.casefold()).strip()


def validate(directory: Path, tokenizer_path: Path | None) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text())
    require(set(manifest["split_counts"]) == {"train", "validation", "test"}, "Expected all three splits")
    require(manifest["max_sequence_length"] == 1024, "Expected the current 1024-token training limit")
    for filename, expected in manifest["file_sha256"].items():
        actual = hashlib.sha256((directory / filename).read_bytes()).hexdigest()
        require(actual == expected, f"File hash differs: {filename}")

    provenance_rows = read_jsonl(directory / "provenance.jsonl")
    provenance = {row["id"]: row for row in provenance_rows}
    require(len(provenance) == len(provenance_rows), "Duplicate provenance IDs")
    all_rows = []
    ids = set()
    roots = set()
    prompts = set()
    group_splits = {}
    split_counts = {}
    multi_turn = 0
    assistant_turns = 0
    for split, expected_count in manifest["split_counts"].items():
        rows = read_jsonl(directory / f"{split}.jsonl")
        require(len(rows) == expected_count, f"Unexpected {split} count")
        split_counts[split] = len(rows)
        for row in rows:
            require(set(row) == {"id", "messages"}, f"Unexpected training fields in {split}")
            identifier = row["id"]
            require(identifier not in ids, f"Duplicate ID: {identifier}")
            ids.add(identifier)
            info = provenance[identifier]
            require(info["split"] == split, f"Provenance split differs: {identifier}")
            root_id = info["source_tree_id"]
            require(root_id not in roots, f"Repeated source conversation: {root_id}")
            roots.add(root_id)
            group = info["topic_group"]
            require(
                group_splits.setdefault(group, split) == split,
                f"Topic group crosses splits: {group}",
            )
            messages = row["messages"]
            require(len(messages) >= 3 and len(messages) % 2 == 1, identifier)
            require(messages[0] == manifest["system_message"], identifier)
            expected_roles = ["system"] + ["user", "assistant"] * (
                (len(messages) - 1) // 2
            )
            require([m["role"] for m in messages] == expected_roles, identifier)
            require(
                all(isinstance(m["content"], str) and m["content"].strip() for m in messages),
                f"Empty/non-text message: {identifier}",
            )
            original = info["source_messages"]
            require(len(original) == len(messages) - 1, identifier)
            require(original[0]["message_id"] == root_id, f"Source root differs: {identifier}")
            require(original[0]["parent_id"] is None, f"Not a root question: {identifier}")
            require(
                [m["message_id"] for m in original] == info["source_message_ids"],
                f"Source IDs differ: {identifier}",
            )
            for previous, following in zip(original, original[1:]):
                require(
                    following["parent_id"] == previous["message_id"],
                    f"Source conversation chain differs: {identifier}",
                )
            for message, source in zip(messages[1:], original, strict=True):
                require(message["role"] == source["role"], identifier)
                if message["role"] == "user":
                    require(message["content"] == source["content"], identifier)
                else:
                    require(
                        not any(x in message["content"] for x in ("<think>", "</think>", "<|im_start|>")),
                        f"Unexpected template/reasoning marker: {identifier}",
                    )
            prompt = normalize(messages[1]["content"])
            require(prompt not in prompts, f"Duplicate normalized question: {identifier}")
            prompts.add(prompt)
            multi_turn += len(messages) > 3
            assistant_turns += (len(messages) - 1) // 2
            all_rows.append(row)
    require(ids == set(provenance), "Training rows and provenance IDs differ")

    held_out = read_jsonl(directory / "test_prompts.jsonl")
    test_rows = read_jsonl(directory / "test.jsonl")
    require(len(held_out) == len(test_rows), "Test prompt count differs")
    for prompt, full in zip(held_out, test_rows, strict=True):
        require(set(prompt) == {"id", "messages"}, "Unexpected test prompt fields")
        require(prompt["id"] == full["id"], "Test prompt ID differs")
        require(prompt["messages"] == full["messages"][:2], "Test prompt includes a target")

    report = {
        "structure_and_provenance": "passed",
        "split_counts": split_counts,
        "total_conversations": len(all_rows),
        "assistant_turns": assistant_turns,
        "multi_turn_conversations": multi_turn,
        "source_conversations_disjoint": True,
        "recorded_topic_groups_disjoint": True,
        "normalized_initial_questions_unique": True,
        "original_user_messages_preserved": True,
        "tokenizer_validation": "not_run",
    }
    if tokenizer_path is not None:
        from transformers import AutoTokenizer

        for name, expected in manifest["tokenizer"]["file_sha256"].items():
            actual = hashlib.sha256((tokenizer_path / name).read_bytes()).hexdigest()
            require(actual == expected, f"Tokenizer file differs from verified snapshot: {name}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        report.update(validate_masks(all_rows, tokenizer, manifest["max_sequence_length"]))
        report["tokenizer_class"] = type(tokenizer).__name__
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(Path(__file__).resolve().parent, args.tokenizer), indent=2))

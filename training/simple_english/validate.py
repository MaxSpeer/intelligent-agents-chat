"""Validate the final dataset's integrity, source splits, and optional token masking."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from training.simple_english.evaluate import load_dialogues
from training.simple_english.examples import build_examples

DATA = Path(__file__).resolve().parent / "data"
SPLITS = ("train", "validation", "test")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_masks(rows, tokenizer, max_sequence_length):
    lengths = []
    for row in rows:
        examples = build_examples(tokenizer, row["messages"], max_sequence_length)
        require(len(examples) == sum(m["role"] == "assistant" for m in row["messages"]),
                f"Missing answer example: {row['id']}")
        for example in examples:
            start = example.prompt_length
            require(all(v == -100 for v in example.labels[:start]), "Unmasked context")
            require(example.labels[start:] == example.input_ids[start:], "Incorrect target labels")
            expected = row["messages"][example.message_index]["content"].lstrip("\n") + "<|im_end|>\n"
            actual = tokenizer.decode(example.labels[start:], skip_special_tokens=False,
                                      clean_up_tokenization_spaces=False)
            require(actual == expected, "Supervised text differs from the answer and end marker")
            lengths.append(len(example.input_ids))
    return {
        "answer_examples": len(lengths),
        "all_context_tokens_masked": True,
        "all_supervised_text_matches_reference_answer_and_end_marker": True,
        "min_tokens": min(lengths, default=0), "max_tokens": max(lengths, default=0),
    }


def validate(directory=DATA, tokenizer_path=None):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "manifest.json").read_text())
    required = {"train.jsonl", "validation.jsonl", "test.jsonl", "provenance.jsonl", "LICENSE", "NOTICE"}
    require(set(manifest["file_sha256"]) == required, "Incomplete dataset checksum manifest")
    for name, expected in manifest["file_sha256"].items():
        actual = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        require(actual == expected, f"File hash mismatch: {name}")

    ids = {}
    rows_all = []
    counts = {}
    for split in SPLITS:
        _, rows = load_dialogues(directory, split)
        for row in rows:
            require(set(row) == {"id", "messages"}, "Unexpected row fields")
            require(row["id"] not in ids, f"Duplicate ID across splits: {row['id']}")
            ids[row["id"]] = split
            for message in row["messages"]:
                require(set(message) == {"role", "content"}, "Unexpected message fields")
                text = message["content"]
                require(isinstance(text, str) and bool(text.strip()), "Empty message content")
                require(not any(marker in text for marker in
                                ("<think>", "</think>", "<|im_start|>", "<|im_end|>")),
                        "Template marker in message content")
        answers = sum(m["role"] == "assistant" for row in rows for m in row["messages"])
        require(answers == manifest["split_statistics"][split]["assistant_answers"],
                f"Wrong answer count: {split}")
        counts[split] = {"dialogues": len(rows), "assistant_answers": answers}
        rows_all.extend(rows)

    provenance = [json.loads(line) for line in (directory / "provenance.jsonl").read_text().splitlines()]
    require(Counter(r["id"] for r in provenance) == Counter(ids.keys()), "Missing or duplicate provenance")
    source_splits = {}
    for record in provenance:
        source = record
        while "source_lineage" in source:
            source = source["source_lineage"]
        split = ids[record["id"]]
        require(source["split"] == split, "Source split differs from data split")
        require(bool(source.get("repository")) and bool(source.get("revision")), "Missing source attribution")
        group = (source["repository"], source["split_group"])
        require(source_splits.setdefault(group, split) == split, "Source conversation crosses splits")
    sample_ids = manifest["validation_generation_ids"]
    require(len(sample_ids) == len(set(sample_ids)) == 10, "Invalid validation sample IDs")
    require(all(ids.get(identifier) == "validation" for identifier in sample_ids),
            "Generation sample is outside validation split")
    report = {
        "file_integrity": "passed", "structure_and_source_splits": "passed", "counts": counts,
        "semantic_holdout_independence": "not established; inherited topic overlap remains",
        "tokenizer_validation": "not_run",
    }
    if tokenizer_path:
        from transformers import AutoTokenizer

        tokenizer_path = Path(tokenizer_path).resolve()
        for name, expected in manifest["tokenizer"]["file_sha256"].items():
            require(hashlib.sha256((tokenizer_path / name).read_bytes()).hexdigest() == expected,
                    f"Pinned tokenizer differs: {name}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        report.update(validate_masks(rows_all, tokenizer, manifest["max_sequence_length"]))
        report["tokenizer_validation"] = "passed"
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--tokenizer", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.data, args.tokenizer), indent=2))

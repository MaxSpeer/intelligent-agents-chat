"""Read-only structure, provenance, split, and optional local-tokenizer checks."""
import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_row(original, revised):
    require(set(revised) == {"id", "messages"}, "Unexpected row fields")
    require(revised["id"] == original["id"], "Changed row ID/order")
    require(len(revised["messages"]) == len(original["messages"]), "Changed turn count")
    require(len(revised["messages"]) >= 3 and len(revised["messages"]) % 2 == 1, "Invalid conversation length")
    expected_roles = ["system"] + ["user", "assistant"] * ((len(revised["messages"]) - 1) // 2)
    require([m["role"] for m in revised["messages"]] == expected_roles, "Invalid conversation roles")
    for before, after in zip(original["messages"], revised["messages"], strict=True):
        require(set(after) == {"role", "content"}, "Unexpected message fields")
        require(before["role"] == after["role"], "Changed role")
        require(isinstance(after["content"], str) and bool(after["content"].strip()), "Empty content")
        if before["role"] != "assistant":
            require(before == after, "Changed user/system message")
        require(not any(t in after["content"] for t in ("<think>", "</think>", "<|im_start|>", "<|im_end|>")), "Template marker in content")


def validate(directory=HERE, tokenizer_path=None):
    directory = Path(directory).resolve()
    parent = directory.parent / "plain_english_2k_simplified_v1"
    spec = importlib.util.spec_from_file_location("clear_v2_build_validation", directory / "build.py")
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    build.parent_check(directory)
    parent_spec = importlib.util.spec_from_file_location("clear_v2_parent_validation", parent / "validate.py")
    parent_validator = importlib.util.module_from_spec(parent_spec)
    parent_spec.loader.exec_module(parent_validator)
    parent_report = parent_validator.validate(parent)
    manifest = json.loads((directory / "manifest.json").read_text())
    for name in ("README.md", "EXAMPLES.md", "NOTICE", "LICENSE", "quality_review.json"):
        require((directory / name).is_file(), f"Missing release file: {name}")
    for name, expected in manifest["file_sha256"].items():
        require(sha(directory / name) == expected, f"File hash mismatch: {name}")
    for name in ("test.jsonl", "test_prompts.jsonl", "validation_sample_ids.json"):
        require((directory / name).read_bytes() == (parent / name).read_bytes(), f"Frozen file changed: {name}")
    edits = build.decisions(directory)
    audit_rows = read(directory / "answer_changes.jsonl")
    audit = {(r["split"], r["id"], r["message_index"]): r for r in audit_rows}
    require(len(audit) == len(audit_rows), "Duplicate audit entries")
    provenance = read(directory / "provenance.jsonl")
    require([r["source_lineage"] for r in provenance] == read(parent / "provenance.jsonl"), "Changed source lineage")
    seen, initial_questions, seen_answers, rows_all = set(), {}, defaultdict(list), []
    used, audit_used, counts = set(), set(), {}
    for split, expected_count in (("train", 2000), ("validation", 100), ("test", 100)):
        old, new = read(parent / f"{split}.jsonl"), read(directory / f"{split}.jsonl")
        require(len(old) == len(new) == expected_count, f"Wrong split size: {split}")
        assistant_count, changed = 0, 0
        for index, (original, row) in enumerate(zip(old, new, strict=True)):
            check_row(original, row)
            require(row["id"] not in seen, "Duplicate conversation ID")
            seen.add(row["id"])
            normalized_question = re.sub(r"\W+", " ", row["messages"][1]["content"].casefold()).strip()
            require(normalized_question not in initial_questions, "Duplicate initial question")
            initial_questions[normalized_question] = split
            for i, message in enumerate(row["messages"]):
                if message["role"] != "assistant":
                    continue
                assistant_count += 1
                key = (split, row["id"], i)
                require(key in audit, f"Missing answer audit: {key}")
                audit_used.add(key)
                before = original["messages"][i]["content"]
                entry = audit[key]
                require(entry["original"] == before and entry["revised"] == message["content"], "Audit text mismatch")
                require(entry["parent_answer_sha256"] == build.text_digest(before), "Stale original answer")
                require(entry["revised_answer_sha256"] == build.text_digest(message["content"]), "Stale revised answer")
                if split != "test":
                    require(key in edits, f"Unreviewed answer: {key}")
                    decision = edits[key]
                    used.add(key)
                    require(decision["content"] == message["content"], "Unrecorded answer edit")
                    require(decision["source_index"] == index, "Wrong source index")
                    require(decision["parent_answer_sha256"] == build.text_digest(before), "Stale decision")
                    require(decision["action"] == ("keep" if before == message["content"] else "rewrite"), "Wrong decision action")
                else:
                    require(entry["action"] == "frozen_holdout" and entry["reviewer"] is None, "Test review mislabelled")
                changed += before != message["content"]
                normal = " ".join(build.WORD.findall(message["content"].casefold()))
                if len(normal.split()) >= 12:
                    seen_answers[normal].append(key)
            rows_all.append(row)
        counts[split] = {"dialogues": expected_count, "assistant_answers": assistant_count, "changed_answers": changed}
    require(used == set(edits) and audit_used == set(audit), "Unused review records")
    require(sum(c["assistant_answers"] for c in counts.values()) == 3443, "Wrong answer total")
    require(counts["train"]["assistant_answers"] == 3085 and counts["validation"]["assistant_answers"] == 160, "Wrong training/validation answers")
    expected_prompts = [{"id": r["id"], "messages": r["messages"][:2]} for r in read(directory / "test.jsonl")]
    require(read(directory / "test_prompts.jsonl") == expected_prompts, "Invalid test prompts")
    duplicates = [keys for keys in seen_answers.values() if len(keys) > 1]
    cross_split = [keys for keys in duplicates if len({key[0] for key in keys}) > 1]
    report = {
        "structure_and_provenance": "passed", "parent_files_unchanged": True,
        "parent_source_split_provenance_validation": parent_report["structure_and_provenance"],
        "all_training_and_validation_answers_have_editorial_decisions": True,
        "questions_system_roles_order_and_splits_unchanged": True,
        "test_files_byte_identical_to_parent": True, "split_statistics": counts,
        "assistant_turns": 3443, "editorial_answer_decisions": len(edits),
        "human_reviewed": False, "complete_external_fact_check": False,
        "exact_long_answer_duplicate_groups": duplicates,
        "exact_long_answer_cross_split_duplicate_groups": cross_split,
        "semantic_holdout_independence": "not_established; inherited topic overlap remains",
        "tokenizer_validation": "not_run",
    }
    if tokenizer_path:
        from chat_examples import build_examples
        from transformers import AutoTokenizer
        tokenizer_path = Path(tokenizer_path).resolve()
        for name, expected in manifest["tokenizer"]["file_sha256"].items():
            require(sha(tokenizer_path / name) == expected, f"Pinned tokenizer differs: {name}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        lengths = []
        for row in rows_all:
            examples = build_examples(tokenizer, row["messages"], manifest["max_sequence_length"])
            require(len(examples) == sum(m["role"] == "assistant" for m in row["messages"]), "Missing answer example")
            for example in examples:
                start = example.prompt_length
                require(all(v == -100 for v in example.labels[:start]), "Unmasked context")
                require(example.labels[start:] == example.input_ids[start:], "Wrong labels")
                target = row["messages"][example.message_index]["content"].lstrip("\n") + "<|im_end|>\n"
                require(tokenizer.decode(example.labels[start:], skip_special_tokens=False, clean_up_tokenization_spaces=False) == target, "Supervised text differs")
                lengths.append(len(example.input_ids))
        report.update(tokenizer_validation="passed", answer_examples=len(lengths),
                      all_context_tokens_masked=True, all_answer_text_and_end_markers_supervised=True,
                      min_tokens=min(lengths), max_tokens=max(lengths), truncated_conversations=0)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(HERE, args.tokenizer), indent=2))

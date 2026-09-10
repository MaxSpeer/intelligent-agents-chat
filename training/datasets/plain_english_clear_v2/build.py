"""Assemble the reviewed revision from explicit per-answer decisions.

Only writes derived files in this dataset directory. Never edits the parent,
loads model weights, contacts an API, or starts training.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import statistics

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / "plain_english_2k_simplified_v1"
COUNTS = {"train": 2000, "validation": 100, "test": 100}
WORD = re.compile(r"\b\w+(?:['’\-]\w+)*\b")


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def text_digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_rows(path, value):
    Path(path).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in value))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def parent_check(directory=HERE):
    directory = Path(directory)
    parent = directory.parent / "plain_english_2k_simplified_v1"
    snapshot = json.loads((directory / "parent_snapshot.json").read_text())
    for name, expected in snapshot["sha256"].items():
        require(digest(parent / name) == expected, f"Parent changed: {name}")


def decisions(directory=HERE):
    result = {}
    for path in sorted((Path(directory) / "decisions").glob("*.jsonl")):
        for row in read(path):
            key = (row["split"], row["id"], row["message_index"])
            require(key not in result, f"Duplicate decision: {key}")
            require(row["split"] in ("train", "validation"), f"No test edits allowed: {key}")
            require(row["human_reviewed"] is False and bool(row["reviewer"]), f"Invalid reviewer: {key}")
            require(bool(row["note"].strip()), f"Missing review note: {key}")
            result[key] = row
    return result


def sentence_parts(text):
    # Descriptive proxy only. Splits may include headings/list items; decimal
    # points are not split because a following space is required.
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


def metrics(texts):
    words = [len(WORD.findall(text)) for text in texts]
    lengths = [len(WORD.findall(s)) for text in texts for s in sentence_parts(text)]
    return {
        "answers": len(words), "mean_answer_words": round(statistics.mean(words), 3),
        "median_answer_words": statistics.median(words),
        "mean_sentence_words_proxy": round(statistics.mean(lengths), 3),
        "sentence_units_proxy": len(lengths),
        "sentence_units_over_20_words_proxy": sum(n > 20 for n in lengths),
        "sentence_units_over_25_words_proxy": sum(n > 25 for n in lengths),
        "max_sentence_words_proxy": max(lengths),
        "answers_with_4_or_more_sentence_units": sum(len(sentence_parts(t)) >= 4 for t in texts),
        "answers_ending_in_question": sum(t.rstrip().endswith("?") for t in texts),
    }


def build():
    parent_check()
    edits = decisions()
    used, all_changes, per_id = set(), [], {}
    all_rows = {}
    stats = {}
    for split, count in COUNTS.items():
        records = read(PARENT / f"{split}.jsonl")
        require(len(records) == count, f"Unexpected parent count: {split}")
        before_texts, after_texts = [], []
        changed = 0
        for source_index, row in enumerate(records):
            changes = []
            for i, message in enumerate(row["messages"]):
                if message["role"] != "assistant":
                    continue
                before = message["content"]
                key = (split, row["id"], i)
                if split == "test":
                    after, action, reviewer, note = before, "frozen_holdout", None, "Copied unchanged; not rewritten or reviewed in this release."
                else:
                    require(key in edits, f"Missing editorial decision: {key}")
                    edit = edits[key]
                    require(edit["source_index"] == source_index, f"Source index mismatch: {key}")
                    require(edit["parent_answer_sha256"] == text_digest(before), f"Stale source text: {key}")
                    after = edit["content"]
                    require(isinstance(after, str) and bool(after.strip()), f"Empty answer: {key}")
                    action = "keep" if after == before else "rewrite"
                    require(edit["action"] == action, f"Action differs: {key}")
                    reviewer, note = edit["reviewer"], edit["note"]
                    used.add(key)
                require(not any(t in after for t in ("<think>", "</think>", "<|im_start|>", "<|im_end|>")), f"Template marker: {key}")
                message["content"] = after
                entry = {
                    "id": row["id"], "split": split, "source_index": source_index,
                    "message_index": i, "action": action,
                    "parent_answer_sha256": text_digest(before), "revised_answer_sha256": text_digest(after),
                    "original": before, "revised": after, "reviewer": reviewer,
                    "human_reviewed": False, "note": note,
                }
                if split != "test" and edits[key].get("sources"):
                    entry["sources"] = edits[key]["sources"]
                all_changes.append(entry)
                changes.append({k: entry[k] for k in ("message_index", "action", "reviewer", "revised_answer_sha256")})
                before_texts.append(before)
                after_texts.append(after)
                changed += int(before != after)
            per_id[row["id"]] = changes
        all_rows[split] = records
        stats[split] = {"dialogues": count, "assistant_answers": len(after_texts), "changed_answers": changed,
                        "before": metrics(before_texts), "after": metrics(after_texts)}
    require(used == set(edits), "Unused decisions")
    for split, records in all_rows.items():
        if split == "test":
            (HERE / "test.jsonl").write_bytes((PARENT / "test.jsonl").read_bytes())
        else:
            write_rows(HERE / f"{split}.jsonl", records)
    for name in ("test_prompts.jsonl", "validation_sample_ids.json", "LICENSE"):
        (HERE / name).write_bytes((PARENT / name).read_bytes())
    (HERE / "parent_manifest.json").write_bytes((PARENT / "manifest.json").read_bytes())
    write_rows(HERE / "answer_changes.jsonl", all_changes)
    write_rows(HERE / "provenance.jsonl", [
        {"id": row["id"], "source_lineage": row, "assistant_revision": per_id[row["id"]], "human_reviewed": False}
        for row in read(PARENT / "provenance.jsonl")
    ])
    write_json(HERE / "style_comparison.json", {
        "method": "Regex word counts and sentence-boundary proxy; not a reading-level, factuality, or comprehension score.",
        "word_pattern": WORD.pattern, "splits": stats,
    })
    chosen = [(41, 2), (74, 2), (674, 2), (886, 2), (1454, 2), (1746, 2)]
    examples = ["# Beispiele aus den neuen Trainingsdaten\n\n"
                "Diese Antworten sind tatsächlich in `train.jsonl` enthalten. Es sind neue Musterantworten, "
                "keine Ausgaben eines neu trainierten Modells. Die Beispiele zeigen bewusst verschiedene Themen.\n"]
    for source_index, message_index in chosen:
        change = next(c for c in all_changes if c["split"] == "train"
                      and c["source_index"] == source_index and c["message_index"] == message_index)
        row = all_rows["train"][source_index]
        question = row["messages"][message_index - 1]["content"]
        examples.append(f"\n## {row['id']}\n\n**Frage:** {question}\n\n"
                        f"**Vorher:**\n\n{change['original']}\n\n**Clear v2:**\n\n{change['revised']}\n")
    (HERE / "EXAMPLES.md").write_text("".join(examples))
    parent_manifest = json.loads((PARENT / "manifest.json").read_text())
    manifest = {k: parent_manifest[k] for k in (
        "language", "system_message", "split_counts", "total_conversations", "assistant_turns",
        "multi_turn_conversations", "base_model", "base_model_revision", "max_sequence_length",
        "validation_generation_ids", "tokenizer",
    )}
    manifest.update(
        dataset_name="plain_english_clear_v2", version="2.0.0",
        prepared_at=datetime.now(timezone.utc).isoformat(),
        parent_dataset="../plain_english_2k_simplified_v1",
        parent_manifest_sha256=digest(PARENT / "manifest.json"),
        objective="Short sentences, familiar words, explained terms, concrete examples and preserved meaning. Whole answers may be longer.",
        review_scope={"all_training_and_validation_answers_have_editorial_decisions": True,
                      "training_answer_decisions": stats["train"]["assistant_answers"],
                      "validation_answer_decisions": stats["validation"]["assistant_answers"],
                      "test_answers_rewritten": 0, "human_reviewed": False,
                      "complete_external_fact_check": False},
        actions=dict(Counter(row["action"] for row in all_changes)),
        split_statistics={split: {**parent_manifest["split_statistics"][split],
                                 "mean_answer_words": stats[split]["after"]["mean_answer_words"]} for split in COUNTS},
        test_policy="Existing test files byte-for-byte unchanged. Inherited topic overlap remains; not an independent new human benchmark.",
        model_evaluation="No new model was trained or evaluated during dataset preparation.",
    )
    manifest["file_sha256"] = {
        str(p.relative_to(HERE)): digest(p) for p in sorted(HERE.rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
        and p.name not in ("manifest.json", "validation_report.json", ".DS_Store")
    }
    write_json(HERE / "manifest.json", manifest)
    parent_check()
    print(json.dumps({"status": "built", "actions": manifest["actions"], "splits": stats}, indent=2))


if __name__ == "__main__":
    build()

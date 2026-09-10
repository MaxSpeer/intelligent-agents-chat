#!/usr/bin/env python3
"""Validate the expanded dataset on CPU; optional exact local tokenizer check."""

from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from chat_examples import build_examples  # noqa: E402


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def norm(text):
    return re.sub(r"\W+", " ", text.casefold()).strip()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(directory, tokenizer_path=None):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    require(
        manifest["split_counts"] == {"train": 2000, "validation": 100, "test": 100},
        "Unexpected split sizes",
    )
    require(bool(manifest["file_sha256"]), "Missing integrity manifest")
    for name, expected in manifest["file_sha256"].items():
        require(
            hashlib.sha256((directory / name).read_bytes()).hexdigest() == expected,
            f"File changed: {name}",
        )
    provenance = read(directory / "provenance.jsonl")
    info = {p["id"]: p for p in provenance}
    require(len(info) == len(provenance), "Duplicate provenance IDs")
    sources = {r["id"]: r for r in read(directory / "source_records.jsonl")}
    reviews = read(directory / "review_decisions.jsonl")
    review = {r["id"]: r for r in reviews}
    require(
        len(review) == len(reviews) == 2379, "Expected one review for each of 2379 source dialogues"
    )
    require(set(review) == set(sources), "Source/review coverage differs")
    legacy = {r["id"]: r for r in read(directory / "legacy_dialogues.jsonl")}
    legacy_provenance = {p["id"]: p for p in read(directory / "legacy_provenance.jsonl")}
    blocked_questions = {
        norm(m["content"])
        for r in legacy.values()
        if r["split"] != "train"
        for m in r["messages"]
        if m["role"] == "user"
    }
    spec = importlib.util.spec_from_file_location("expanded_data_build", directory / "build.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    source_splits = {}
    topic_splits = {}
    group_splits = {}
    question_splits = {}
    ids = set()
    initials = set()
    answer_ids = set()
    answer_texts = set()
    rows_all = []
    split_stats = {}
    legacy_ids = set()
    for split, expected in manifest["split_counts"].items():
        rows = read(directory / f"{split}.jsonl")
        require(len(rows) == expected, f"Wrong {split} count")
        for row in rows:
            require(set(row) == {"id", "messages"}, "Unexpected training fields")
            identifier = row["id"]
            require(identifier not in ids, f"Duplicate ID {identifier}")
            ids.add(identifier)
            p = info[identifier]
            require(p["split"] == split, f"Split mismatch {identifier}")
            for store, key in [
                (source_splits, p["source_id"]),
                (topic_splits, p["topic_group"]),
                (group_splits, p["split_group"]),
            ]:
                require(store.setdefault(key, split) == split, f"Cross-split group {key}")
            messages = row["messages"]
            require(
                len(messages) >= 3 and len(messages) % 2 == 1, f"Incomplete dialogue {identifier}"
            )
            require(messages[0] == manifest["system_message"], f"System differs {identifier}")
            require(
                [m["role"] for m in messages]
                == ["system"] + ["user", "assistant"] * ((len(messages) - 1) // 2),
                f"Wrong roles {identifier}",
            )
            require(
                all(
                    set(m) == {"role", "content"}
                    and isinstance(m["content"], str)
                    and m["content"].strip()
                    for m in messages
                ),
                f"Bad text {identifier}",
            )
            initial = norm(messages[1]["content"])
            require(initial not in initials, f"Duplicate initial question {identifier}")
            initials.add(initial)
            for m in messages[1::2]:
                question = norm(m["content"])
                require(question not in blocked_questions, f"Legacy holdout question {identifier}")
                if builder.substantial_question(question):
                    require(
                        question_splits.setdefault(question, split) == split,
                        f"Cross-split substantial question {identifier}",
                    )
            require(p["human_reviewed"] is False, "Do not claim human review")
            indices = p["source_message_indices"]
            require(len(indices) == len(messages) - 1, f"Source mapping length {identifier}")
            if p["extraction"] == "legacy_unchanged":
                require(
                    split == "train" and legacy[identifier]["split"] == "train",
                    "Legacy split changed",
                )
                require(messages == legacy[identifier]["messages"], "Legacy training was edited")
                original = legacy_provenance[identifier]["source_messages"]
                require(
                    p["source_id"] == "pilot:" + legacy_provenance[identifier]["source_tree_id"],
                    "Legacy tree differs",
                )
                legacy_ids.add(identifier)
            else:
                source = sources[p["source_id"]]
                decision = review[p["source_id"]]
                require(decision["action"] in ("keep", "edit"), "Excluded source selected")
                require(
                    p["repository"] == "HuggingFaceTB/everyday-conversations-llama3.1-2k"
                    and p["revision"] == builder.REVISION,
                    "Wrong source revision",
                )
                require(p["upstream_split"] == source["source_split"], "Upstream split changed")
                require(
                    p["upstream_split"] != "test_sft" or split == "test", "Upstream test leaked"
                )
                original = source["messages"]
                prepared = builder.prepare(source, decision)
                by_index = {m["source_index"]: m for m in prepared}
                require(all(i in by_index for i in indices), "Selected removed source turn")
                expected_messages = [
                    {k: by_index[i][k] for k in ("role", "content")} for i in indices
                ]
                require(messages[1:] == expected_messages, f"Unreviewed changes {identifier}")
                require(
                    not re.search(
                        builder.LEGACY_PATTERN,
                        source["full_topic"] + " " + " ".join(m["content"] for m in prepared),
                        re.I,
                    ),
                    "Legacy topic overlap",
                )
                if p["extraction"] == "standalone_pair":
                    require(
                        len(indices) == 2 and indices[1] == indices[0] + 1,
                        "Invalid standalone pair",
                    )
                    require(
                        by_index[indices[0]]["review_index"] in decision["standalone_user_indices"],
                        "Pair not approved as independent",
                    )
                else:
                    require(p["extraction"] == "substantive_dialogue", "Unknown extraction")
                    require(
                        indices == [m["source_index"] for m in prepared], "Full dialogue modified"
                    )
            for message, index in zip(messages[1:], indices, strict=True):
                source_message = original[index]
                require(
                    message["role"] == source_message["role"], f"Original role differs {identifier}"
                )
                if message["role"] == "user":
                    require(
                        message["content"] == source_message["content"],
                        f"Original user text changed {identifier}",
                    )
                else:
                    key = (p["source_id"], index)
                    require(key not in answer_ids, f"Repeated source answer {identifier}")
                    answer_ids.add(key)
                    normalized_answer = norm(message["content"])
                    if len(normalized_answer.split()) >= 8:
                        require(
                            normalized_answer not in answer_texts,
                            f"Duplicate substantial answer {identifier}",
                        )
                        answer_texts.add(normalized_answer)
                    require(
                        not any(
                            x in message["content"]
                            for x in ("<think>", "</think>", "<|im_start|>", "<|im_end|>")
                        ),
                        f"Template marker {identifier}",
                    )
            rows_all.append(row)
        split_stats[split] = {
            "dialogue_examples": len(rows),
            "source_conversations": len({info[r["id"]]["source_id"] for r in rows}),
            "assistant_answers": sum((len(r["messages"]) - 1) // 2 for r in rows),
            "multi_turn_dialogues": sum(len(r["messages"]) > 3 for r in rows),
            "mean_answer_words": round(
                sum(
                    len(m["content"].split())
                    for r in rows
                    for m in r["messages"]
                    if m["role"] == "assistant"
                )
                / sum((len(r["messages"]) - 1) // 2 for r in rows),
                2,
            ),
        }
    require(ids == set(info), "Unmatched provenance")
    require(
        legacy_ids == {r["id"] for r in legacy.values() if r["split"] == "train"},
        "Not all 80 legacy training rows retained",
    )
    require(split_stats == manifest["split_statistics"], "Recorded statistics differ")
    for edge in read(directory / "near_question_groups.jsonl"):
        splits = {source_splits[source] for source in edge["sources"] if source in source_splits}
        require(len(splits) <= 1, "Recorded near-question group crosses splits")
    test = read(directory / "test.jsonl")
    prompts = read(directory / "test_prompts.jsonl")
    require(
        prompts == [{"id": r["id"], "messages": r["messages"][:2]} for r in test],
        "Test prompts contain targets or differ",
    )
    sample_ids = json.loads((directory / "validation_sample_ids.json").read_text())
    require(
        sample_ids == manifest["validation_generation_ids"], "Validation sample manifest differs"
    )
    require(len(sample_ids) == len(set(sample_ids)) == 10, "Wrong validation sample size")
    require(
        set(sample_ids) <= set(r["id"] for r in read(directory / "validation.jsonl")),
        "Validation sample outside validation",
    )
    report = {
        "structure_and_provenance": "passed",
        "split_counts": manifest["split_counts"],
        "split_statistics": split_stats,
        "source_conversations_disjoint": True,
        "recorded_topic_and_near_question_groups_disjoint": True,
        "upstream_test_sources_only_in_test": True,
        "source_answers_never_repeated": True,
        "original_user_messages_preserved": True,
        "legacy_80_train_dialogues_unchanged": True,
        "legacy_heldout_question_and_recorded_topic_checks": "passed",
        "ai_review_coverage": 2379,
        "human_reviewed": False,
        "semantic_independence_guaranteed": False,
        "tokenizer_validation": "not_run",
    }
    if tokenizer_path is not None:
        from transformers import AutoTokenizer

        tokenizer_path = Path(tokenizer_path)
        for name, expected in manifest["tokenizer"]["file_sha256"].items():
            require(
                hashlib.sha256((tokenizer_path / name).read_bytes()).hexdigest() == expected,
                f"Tokenizer changed: {name}",
            )
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        lengths = []
        supervised = []
        for row in rows_all:
            examples = build_examples(tokenizer, row["messages"], manifest["max_sequence_length"])
            require(len(examples) == (len(row["messages"]) - 1) // 2, "Missing target")
            for example, answer in zip(examples, row["messages"][2::2], strict=True):
                start = example.prompt_length
                require(all(x == -100 for x in example.labels[:start]), "Unmasked context")
                require(example.labels[start:] == example.input_ids[start:], "Wrong answer labels")
                decoded = tokenizer.decode(
                    [i for i in example.labels if i != -100],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                require(
                    decoded == answer["content"].lstrip("\n") + "<|im_end|>\n",
                    "Incorrect target text",
                )
                lengths.append(len(example.input_ids))
                supervised.append(len(example.labels) - start)
        report.update(
            tokenizer_validation="passed",
            masking_version="one_answer_per_example_non_thinking_v2",
            answer_examples=len(lengths),
            all_context_tokens_masked=True,
            all_supervised_text_matches_reference_answer_and_end_marker=True,
            min_tokens=min(lengths),
            max_tokens=max(lengths),
            mean_tokens=round(sum(lengths) / len(lengths), 2),
            min_assistant_training_tokens=min(supervised),
            truncated_conversations=0,
            tokenizer_class=type(tokenizer).__name__,
        )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(HERE, args.tokenizer), indent=2))

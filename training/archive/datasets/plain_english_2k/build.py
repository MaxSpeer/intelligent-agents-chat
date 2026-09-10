#!/usr/bin/env python3
"""Assemble reviewed source dialogues locally; never load a model or submit a job.

Needs scikit-learn for deterministic near-question grouping. Input source cache
and the three AI review files are required. No automatic source acceptance.
"""

from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import shutil

HERE = Path(__file__).resolve().parent
PILOT = HERE.parent / "plain_english"
REVISION = "14f543216b9ba42b6b951dc5bd199460d193b162"
SOURCE = "HuggingFaceTB/everyday-conversations-llama3.1-2k"
SEED = "plain-english-2k-2026-09-09"
# Conservative protection for topics in the already-exposed pilot holdouts.
LEGACY_PATTERN = r"photosynth|chlorophyll|calvin cycle|earthquake|seismic|tectonic|knitt?ing|crochet|two.factor|2fa|cornell|habit.{0,25}(form|build|develop)|(?:form|build|develop).{0,25}habit|augmented reality|virtual reality|mixed reality|cheese|vector.{0,20}norm|sphere.{0,20}volume|volume.{0,20}sphere|light pollution|markov|newton|schr[oö]dinger|identity function|distributed hash|pronounc.{0,25}(?:th |th[\"\'])|starting conversations|start a conversation|medieval.{0,25}cloth|historical.{0,25}cloth|historically.{0,40}cloth|synonym.{0,40}reject|reject.{0,40}synonym|volume.{0,25}ball|ball.{0,25}volume|dht node|(?:star|stargaz).{0,40}city|city.{0,40}(?:star|stargaz)|(?:conversation|talk).{0,35}stranger"


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def lines(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def norm(s):
    return re.sub(r"\W+", " ", s.casefold()).strip()


def substantial_question(question):
    words = set(norm(question).split())
    generic = set(
        "what is are it how does do did that this why can could i you me more tell about else and they them those these thanks thank okay ok great sounds good interesting so yes no please examples an example".split()
    )
    return len(norm(question).split()) >= 6 or bool(words - generic)


def digest(value):
    return hashlib.sha256((SEED + value).encode()).hexdigest()


def safe_standalone_question(message):
    if message["review_index"] == 0:
        return True
    return not re.search(r"\b(it|its|they|them|these|those|this)\b", norm(message["content"]))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_closing(user, assistant):
    question = norm(user)
    if "?" in user or re.search(
        r"\b(explain|tell|show|give|write|list|describe|suggest|recommend|help me|what|why|how|when|where|which|can you|could you)\b",
        question,
    ):
        return False
    acknowledgement = re.match(
        r"^(?:thank you|thanks|great|perfect|got it|that helps|that makes sense|that sounds|okay|ok|i ll|i will|bye)\b",
        question,
    )
    response = re.match(
        r"^(?:you re welcome|enjoy|great|sounds like|have fun|have a|i m glad)\b", norm(assistant)
    )
    return bool(
        acknowledgement and response and len(user.split()) < 25 and len(assistant.split()) < 45
    )


def prepare(raw, decision):
    messages = raw["messages"]
    offset = (
        2
        if len(messages) > 1 and messages[1]["content"] == "Hello! How can I help you today?"
        else 0
    )
    selected = list(range(offset, len(messages)))
    while selected and messages[selected[-1]]["role"] == "user":
        selected.pop()
    expected = ["user", "assistant"] * (len(selected) // 2)
    assert [messages[i]["role"] for i in selected] == expected, raw["id"]
    rewrites = decision["assistant_rewrites"]
    for key, value in rewrites.items():
        assert 0 <= int(key) < len(selected) and int(key) % 2 == 1 and value.strip(), (
            raw["id"],
            key,
        )
    assert all(
        isinstance(i, int) and 0 <= i < len(selected) and i % 2 == 0
        for i in decision.get("standalone_user_indices", [])
    ), raw["id"]
    result = [
        {
            **messages[j],
            "source_index": j,
            "review_index": i,
            "content": rewrites.get(str(i), messages[j]["content"]),
        }
        for i, j in enumerate(selected)
    ]
    while result and norm(result[0]["content"]) in {
        "hi",
        "hello",
        "hi there",
        "hello there",
        "hey",
        "hey there",
    }:
        result = result[2:]
    while result and is_closing(result[-2]["content"], result[-1]["content"]):
        result = result[:-2]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache", type=Path, help="Original preparation cache; omit to rebuild bundled records"
    )
    parser.add_argument(
        "--allow-partial", action="store_true", help="Pool audit only; do not write final files"
    )
    args = parser.parse_args()
    pilot_manifest = json.loads((PILOT / "manifest.json").read_text())
    system = pilot_manifest["system_message"]
    if args.cache:
        raw_rows = read(args.cache / "sources/everyday-all.jsonl")
        download_manifest = json.loads(
            (args.cache / "sources/everyday-conversations/download-manifest.json").read_text()
        )
        import pyarrow.parquet as pq

        for item in download_manifest["files"]:
            path = args.cache / "sources/everyday-conversations" / Path(item["path"]).name
            assert sha(path) == item["sha256"], "Downloaded source changed"
        for split in ("train_sft", "test_sft"):
            original = pq.read_table(
                args.cache / f"sources/everyday-conversations/{split}-00000-of-00001.parquet"
            ).to_pylist()
            for row in (r for r in raw_rows if r["source_split"] == split):
                src = original[row["source_row"]]
                assert (
                    row["messages"] == src["messages"] and row["full_topic"] == src["full_topic"]
                ), row["id"]
    else:
        raw_rows = read(HERE / "source_records.jsonl")
        download_manifest = json.loads((HERE / "manifest.json").read_text())["sources"][0][
            "data_files"
        ]
    raw = {r["id"]: r for r in raw_rows}
    decisions = (
        [r for i in range(3) for r in read(args.cache / f"reviews/decisions-{i}.jsonl")]
        if args.cache
        else read(HERE / "review_decisions.jsonl")
    )
    assert len({r["id"] for r in decisions}) == len(decisions), "Duplicate review IDs"
    assert all(r["id"] in raw for r in decisions)
    decisions_by_id = {r["id"]: r for r in decisions}
    for change in read(HERE / "editorial_overrides.jsonl"):
        if change["id"] not in decisions_by_id:
            continue
        decision = decisions_by_id[change["id"]]
        if change["override_id"] in decision.get("applied_overrides", []):
            continue
        decision["assistant_rewrites"].update(change["assistant_rewrites"])
        if decision["assistant_rewrites"]:
            decision["action"] = "edit"
        decision["standalone_user_indices"] = [
            i
            for i in decision["standalone_user_indices"]
            if i not in change["remove_standalone_user_indices"]
        ]
        decision["reason"] += " Independent audit: " + " ".join(change["notes"])
        decision["sources"] = sorted(set(decision.get("sources", []) + change["sources"]))
        decision.setdefault("applied_overrides", []).append(change["override_id"])
    if not args.allow_partial:
        assert set(raw) == {r["id"] for r in decisions}, "Review is not complete"
    legacy_prov = read(PILOT / "provenance.jsonl")
    legacy_rows = {
        r["id"]: r
        for split in ("train", "validation", "test")
        for r in read(PILOT / f"{split}.jsonl")
    }
    legacy_heldout = [r for r in legacy_prov if r["split"] != "train"]
    blocked_questions = {
        norm(m["content"])
        for r in legacy_heldout
        for m in legacy_rows[r["id"]]["messages"]
        if m["role"] == "user"
    }
    exclusions = Counter()
    candidates = []
    source_records = []
    source_info = {}
    for d in sorted(decisions, key=lambda r: r["id"]):
        r = raw[d["id"]]
        source_records.append(
            {
                k: r[k]
                for k in (
                    "id",
                    "source_row",
                    "source_split",
                    "full_topic",
                    "topic",
                    "subtopic",
                    "messages",
                )
            }
        )
        if d["action"] == "exclude":
            exclusions["review_excluded"] += 1
            continue
        assert d["action"] in ("keep", "edit")
        assert bool(d["assistant_rewrites"]) == (d["action"] == "edit"), d["id"]
        prepared = prepare(r, d)
        if not prepared:
            exclusions["boilerplate_only"] += 1
            continue
        # Search all source turns so a held-out subject cannot hide in a follow-up.
        topic_text = r["full_topic"] + " " + " ".join(m["content"] for m in prepared)
        if re.search(LEGACY_PATTERN, topic_text, re.I) or any(
            norm(m["content"]) in blocked_questions for m in prepared if m["role"] == "user"
        ):
            exclusions["legacy_topic_or_question"] += 1
            continue
        standalone = set(d.get("standalone_user_indices", []))
        assert all(isinstance(i, int) and i >= 0 and i % 2 == 0 for i in standalone), r["id"]
        pairs = [
            prepared[i : i + 2]
            for i in range(0, len(prepared), 2)
            if prepared[i]["review_index"] in standalone and safe_standalone_question(prepared[i])
        ]
        # Retain 20% of eligible sources intact. Other sources yield genuinely
        # independent Q/A pairs, with every source answer used at most once.
        use_pairs = len(pairs) >= 2 and int(digest(r["id"])[:8], 16) % 5 != 0
        windows = pairs if use_pairs else [prepared]
        source_info[r["id"]] = {
            "repository": SOURCE,
            "revision": REVISION,
            "upstream_split": r["source_split"],
            "topic_group": "everyday:" + norm(r["full_topic"]),
        }
        for window in windows:
            identifier = (
                "plain2k-"
                + r["id"]
                + ("-qa" + str(window[0]["source_index"]) if use_pairs else "-dialogue")
            )
            row = {
                "id": identifier,
                "messages": [system] + [{k: m[k] for k in ("role", "content")} for m in window],
            }
            info = {
                "id": identifier,
                "source_id": r["id"],
                **source_info[r["id"]],
                "source_message_indices": [m["source_index"] for m in window],
                "extraction": "standalone_pair" if use_pairs else "substantive_dialogue",
                "review_action": d["action"],
                "review_reason": d["reason"],
                "human_reviewed": False,
            }
            candidates.append((row, info))
    # Retain all 80 original training dialogues, unmodified.
    for p in legacy_prov:
        if p["split"] != "train":
            continue
        source_id = "pilot:" + p["source_tree_id"]
        source_info[source_id] = {
            "repository": p["source_dataset"],
            "revision": p["source_revision"],
            "upstream_split": "legacy_train",
            "topic_group": "pilot:" + p["topic_group"],
        }
        info = {
            "id": p["id"],
            "source_id": source_id,
            **source_info[source_id],
            "source_message_indices": list(range(len(p["source_messages"]))),
            "extraction": "legacy_unchanged",
            "review_action": "inherited_pilot_review",
            "review_reason": p["rewrite_notes"],
            "human_reviewed": False,
        }
        candidates.append((legacy_rows[p["id"]], info))
    # Exact initial duplicates are dropped deterministically, keeping pilot first.
    unique = []
    seen = set()
    seen_answers = set()
    for row, info in sorted(
        candidates, key=lambda p: (not p[1]["source_id"].startswith("pilot:"), digest(p[0]["id"]))
    ):
        question = norm(row["messages"][1]["content"])
        if question in seen:
            exclusions["duplicate_initial_example"] += 1
            continue
        answers = [
            norm(m["content"])
            for m in row["messages"][2::2]
            if len(norm(m["content"]).split()) >= 8
        ]
        if any(answer in seen_answers for answer in answers) or len(answers) != len(set(answers)):
            assert not info["source_id"].startswith("pilot:"), (
                "Legacy answer duplicates require review"
            )
            exclusions["duplicate_substantial_answer_example"] += 1
            continue
        seen.add(question)
        seen_answers.update(answers)
        unique.append((row, info))
    candidates = unique
    # Group sources by recorded narrow topic and near-identical substantial user
    # questions. Generic short follow-ups are ignored by the lexical grouping.
    active = {info["source_id"] for _, info in candidates}
    parent = {key: key for key in active}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)

    topics = {}
    for source in sorted(active):
        topic = source_info[source]["topic_group"]
        if topic in topics:
            union(source, topics[topic])
        else:
            topics[topic] = source
    question_sources = defaultdict(set)
    for row, info in candidates:
        for message in row["messages"][1::2]:
            question = norm(message["content"])
            if substantial_question(question):
                question_sources[question].add(info["source_id"])
    questions = sorted(q for q in question_sources if len(q.split()) >= 6)
    for sources in question_sources.values():
        sources = sorted(sources)
        for other in sources[1:]:
            union(sources[0], other)
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors

    edges = []
    for analyzer, ngram, threshold in [("word", (1, 2), 0.84), ("char_wb", (3, 5), 0.90)]:
        matrix = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram).fit_transform(questions)
        distances, neighbors = (
            NearestNeighbors(metric="cosine", algorithm="brute")
            .fit(matrix)
            .radius_neighbors(matrix, radius=1 - threshold)
        )
        for i, (dd, nn) in enumerate(zip(distances, neighbors)):
            for distance, j in zip(dd, nn):
                if j <= i:
                    continue
                left = sorted(question_sources[questions[i]])
                right = sorted(question_sources[questions[j]])
                for source in left + right:
                    union(left[0], source)
                edges.append(
                    {
                        "question_a": questions[i],
                        "question_b": questions[j],
                        "sources": sorted(set(left + right)),
                        "method": analyzer,
                        "similarity": round(1 - float(distance), 6),
                    }
                )
    groups = defaultdict(list)
    for row, info in candidates:
        group = find(info["source_id"])
        info["split_group"] = group
        groups[group].append((row, info))
    stats = {
        "reviewed_sources": len(decisions),
        "review_actions": dict(Counter(d["action"] for d in decisions)),
        "exclusions": dict(exclusions),
        "candidate_examples": len(candidates),
        "candidate_sources": len(active),
        "groups": len(groups),
        "largest_groups": sorted([len(v) for v in groups.values()], reverse=True)[:15],
    }
    print(json.dumps(stats, indent=2), flush=True)
    if args.allow_partial:
        return
    fixed_train = {
        g
        for g, items in groups.items()
        if any(i["source_id"].startswith("pilot:") for _, i in items)
    }
    upstream_test = {
        g for g, items in groups.items() if any(i["upstream_split"] == "test_sft" for _, i in items)
    }
    # Never send an upstream test source to training; conflicts are not silently resolved.
    assert not fixed_train & upstream_test, "Legacy training intersects an upstream test group"
    available = set(groups) - fixed_train - upstream_test
    assigned = {g: "train" for g in fixed_train}
    assigned.update({g: "test" for g in upstream_test})

    def group_size(group_set):
        return sum(len(groups[g]) for g in group_set)

    if group_size(upstream_test) < 100:
        for g in sorted(available, key=digest):
            assigned[g] = "test"
            available.remove(g)
            if sum(len(groups[k]) for k, v in assigned.items() if v == "test") >= 100:
                break
    val_count = 0
    for g in sorted(available, key=digest):
        assigned[g] = "validation"
        available.remove(g)
        val_count += len(groups[g])
        if val_count >= 100:
            break
    assigned.update({g: "train" for g in available})
    split_rows = {}
    provenance = []
    for split, target in [("train", 2000), ("validation", 100), ("test", 100)]:
        pool = [item for g, items in groups.items() if assigned[g] == split for item in items]
        pool.sort(key=lambda p: (not p[1]["source_id"].startswith("pilot:"), digest(p[0]["id"])))
        assert len(pool) >= target, (split, len(pool), target)
        selected = pool[:target]
        split_rows[split] = [r for r, _ in selected]
        provenance.extend([{**info, "split": split} for _, info in selected])
        lines(HERE / f"{split}.jsonl", split_rows[split])
    lines(HERE / "provenance.jsonl", provenance)
    lines(HERE / "source_records.jsonl", source_records)
    lines(HERE / "review_decisions.jsonl", sorted(decisions, key=lambda r: r["id"]))
    lines(HERE / "legacy_provenance.jsonl", legacy_prov)
    lines(
        HERE / "legacy_dialogues.jsonl",
        [{"split": p["split"], **legacy_rows[p["id"]]} for p in legacy_prov],
    )
    lines(HERE / "near_question_groups.jsonl", edges)
    lines(
        HERE / "test_prompts.jsonl",
        [{"id": r["id"], "messages": r["messages"][:2]} for r in split_rows["test"]],
    )
    sample_ids = [
        r["id"] for r in sorted(split_rows["validation"], key=lambda r: digest(r["id"]))[:10]
    ]
    dump(HERE / "validation_sample_ids.json", sample_ids)
    stats.update(
        selected_source_conversations=len({p["source_id"] for p in provenance}),
        selected_by_review_action=dict(Counter(p["review_action"] for p in provenance)),
        selected_by_extraction=dict(Counter(p["extraction"] for p in provenance)),
        selected_topic_groups=len({p["topic_group"] for p in provenance}),
        ai_review_scope="Every source dialogue, including excluded ones; corrections and exclusions recorded. Selective primary fact checks, not exhaustive expert verification.",
        human_reviewed=False,
        legacy_holdout_regex=LEGACY_PATTERN,
        near_question_method="Connected groups from recorded full topic, same source, exact normalized substantive user questions including short questions, word TF-IDF cosine >= 0.84 or character 3-5-gram TF-IDF cosine >= 0.90 (near matching needs six words). This is not a guarantee of semantic independence.",
    )
    dump(HERE / "review_report.json", stats)
    all_rows = [r for rows in split_rows.values() for r in rows]
    split_stats = {
        s: {
            "dialogue_examples": len(rows),
            "source_conversations": len({p["source_id"] for p in provenance if p["split"] == s}),
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
        for s, rows in split_rows.items()
    }
    manifest = {
        "dataset_name": "plain_english_reviewed_2k",
        "version": "2.0.0",
        "prepared_on": "2026-09-09",
        "language": "en",
        "system_message": system,
        "split_counts": {s: len(rows) for s, rows in split_rows.items()},
        "split_statistics": split_stats,
        "total_conversations": len(all_rows),
        "assistant_turns": sum((len(r["messages"]) - 1) // 2 for r in all_rows),
        "multi_turn_conversations": sum(len(r["messages"]) > 3 for r in all_rows),
        "base_model": pilot_manifest["base_model"],
        "base_model_revision": pilot_manifest["base_model_revision"],
        "max_sequence_length": 1024,
        "tokenizer": pilot_manifest["tokenizer"],
        "sources": [
            {
                "repository": SOURCE,
                "revision": REVISION,
                "license": "Apache-2.0",
                "generation_model": "Llama-3.1-70B-Instruct",
                "synthetic": True,
                "data_files": download_manifest,
            },
            {
                **pilot_manifest["source"],
                "derived_from": "unchanged 80 training dialogues of plain_english 1.0.1",
                "assistant_answers": "prior Codex rewrites",
            },
        ],
        "validation_generation_ids": sample_ids,
        "split_method": stats["near_question_method"],
        "fresh_test_note": "Held out from this fine-tuning dataset and previous local adapter evaluations; unknown base-model pretraining exposure. Reference answers are AI-reviewed source targets, not independent human ground truth.",
        "file_sha256": {},
    }
    shutil.copyfile(PILOT / "LICENSE", HERE / "LICENSE")
    manifest["file_sha256"] = {
        path.name: sha(path)
        for path in sorted(HERE.iterdir())
        if path.is_file() and path.name not in {"manifest.json", "validation_report.json"}
    }
    dump(HERE / "manifest.json", manifest)
    print(json.dumps(split_stats, indent=2))


if __name__ == "__main__":
    main()

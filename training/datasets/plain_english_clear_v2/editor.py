"""Record individually reviewed answers in a shard; never alter the parent.

Every assistant turn must be supplied explicitly. A None replacement means
the editor read this answer in context and chose to keep its exact text.
"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / "plain_english_2k_simplified_v1"


def rows(split):
    return [json.loads(line) for line in (PARENT / f"{split}.jsonl").read_text().splitlines()]


def record_batch(split, shard, reviewer, decisions):
    source = rows(split)
    path = HERE / "decisions" / f"{shard}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    keys = {(r["id"], r["message_index"]) for r in existing}
    additions = []
    for index, answers in decisions.items():
        original = source[index]
        wanted = {i for i, m in enumerate(original["messages"]) if m["role"] == "assistant"}
        if set(answers) != wanted:
            raise ValueError(f"Every assistant answer needs an explicit decision: {index}, {wanted}")
        for i, replacement in sorted(answers.items()):
            before = original["messages"][i]["content"]
            fields = replacement if isinstance(replacement, dict) else {"content": replacement}
            content = fields["content"] if fields["content"] is not None else before
            if not isinstance(content, str) or not content.strip():
                raise ValueError("Empty answer")
            key = (original["id"], i)
            if key in keys:
                raise ValueError(f"Duplicate decision: {key}")
            keys.add(key)
            additions.append({
                "id": original["id"], "split": split, "source_index": index,
                "message_index": i,
                "parent_answer_sha256": hashlib.sha256(before.encode()).hexdigest(),
                "content": content, "action": "keep" if content == before else "rewrite",
                "note": fields.get("note", "Read in context; use short sentences, clear actions, and preserve necessary conditions."),
                "reviewer": reviewer, "human_reviewed": False,
                **({"sources": fields["sources"]} if "sources" in fields else {}),
            })
    with path.open("a") as stream:
        for row in additions:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Recorded {len(additions)} answers in {path.name}; total {len(existing) + len(additions)}")


def show(split, start, stop):
    for index, row in enumerate(rows(split)[start:stop], start):
        print(f"\n{index}: {row['id']}")
        for i, message in enumerate(row["messages"]):
            if message["role"] != "system":
                print(f"{message['role'][0].upper()}{i}: {message['content']}")

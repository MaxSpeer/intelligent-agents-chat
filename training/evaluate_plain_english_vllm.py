#!/usr/bin/env python3
"""Compare held-out dialogues through an already running vLLM API.

This local HTTP client never allocates GPUs, starts a server, or trains a model.
Without --run it prints the evaluation plan without requesting generations.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
from threading import Lock
from time import monotonic
from urllib.request import Request, urlopen

from evaluate_plain_english import DATA, generate_dialogue, load_dialogues, write_json


MODELS = ("qwen3-8b", "plain-english")
SETTINGS = {
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": -1,
    "min_p": 0.0,
    "repetition_penalty": 1.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "seed": 42,
    "max_tokens": 2048,
    "stream": False,
    "chat_template_kwargs": {"enable_thinking": False},
}


def request_json(url, payload=None):
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(url, data=body, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=240) as response:
        return json.load(response)


def write_blind_review(destination, rows, comparisons):
    """Hide model names during qualitative assessment, without changing answers."""
    rng = random.Random(20260908)
    mapping = {}
    blocks = [
        "# Anonymous paired review\n",
        "Assess everyday vocabulary and explained terms, directness, sentence complexity, "
        "and helpful examples. Judge correctness and essential coverage separately. "
        "Shorter is not automatically clearer. Choose A, B, or tie for accessibility. "
        "List any important factual error or missing requested information in either answer.\n",
        "T1 uses exactly the same system and user messages. T2 is a follow-up after each "
        "variant's own T1 answer, with no reference assistant answer in the input.\n",
    ]
    for row in rows:
        variants = comparisons[row["id"]]
        questions = [m["content"] for m in row["messages"] if m["role"] == "user"]
        for index, question in enumerate(questions):
            key = f"{row['id']}/T{index + 1}"
            order = list(MODELS)
            rng.shuffle(order)
            mapping[key] = dict(zip(("A", "B"), order))
            blocks.extend([f"## {key}\n", f"Question: {question}\n"])
            for label, model in mapping[key].items():
                turn = variants[model]["turns"][index]
                blocks.extend([
                    f"### {label}\n",
                    turn["content"] + "\n",
                    f"Finish reason: `{turn['finish_reason']}`\n",
                ])
    (destination / "blind-review.md").write_text("\n".join(blocks), encoding="utf-8")
    write_json(destination / "blind-key.json", mapping)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()
    if not 0.0 <= args.temperature <= 2.0:
        parser.error("--temperature must be between 0 and 2")
    settings = {**SETTINGS, "temperature": args.temperature}
    manifest, rows = load_dialogues(DATA)
    for name in ("train.jsonl", "validation.jsonl"):
        path = DATA / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["file_sha256"][name]
        ids = {json.loads(line)["id"] for line in path.read_text().splitlines()}
        assert not ids.intersection(row["id"] for row in rows)

    plan = {
        "status": "planned",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url,
        "models": list(MODELS),
        "base_model": manifest["base_model"],
        "base_model_revision": manifest["base_model_revision"],
        "dataset_sha256": manifest["file_sha256"]["test.jsonl"],
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "system_message": manifest["system_message"],
        "settings": settings,
        "conversations": len(rows),
        "answers_per_model": sum(m["role"] == "user" for r in rows for m in r["messages"]),
        "max_concurrent_conversations": 2,
        "memory_tools_rag": "disabled; API requests contain only the recorded dialogue",
        "reference_assistant_answers_in_input": False,
        "primary_comparison": "10 first answers with identical inputs; 2 follow-ups separately",
        "rubric": {
            "accessibility": "Everyday words, explained terms, directness, manageable sentences, useful examples; choose A/B/tie.",
            "content": "Independently note factual errors, misleading simplifications, and missing essential requested information.",
            "shorter_is_not_automatically_better": True,
        },
    }
    if not args.run:
        print(json.dumps(plan, indent=2))
        return
    if args.output is None:
        parser.error("--output is required with --run; use a new directory")

    listing = request_json(args.base_url.rstrip("/") + "/models")
    available = {entry["id"]: entry for entry in listing["data"]}
    for model in MODELS:
        if model not in available:
            raise SystemExit(f"Server does not advertise {model}")
    if available["plain-english"]["root"] != "/project/adapters/qwen3-8b-plain-english":
        raise SystemExit("Unexpected Plain English adapter location")
    if available["plain-english"]["parent"] != "qwen3-8b":
        raise SystemExit("Unexpected adapter base model")

    args.output.mkdir(parents=True, exist_ok=False)
    for name in (Path(__file__).name, "evaluate_plain_english.py"):
        (args.output / name).write_bytes((Path(__file__).parent / name).read_bytes())
    write_json(args.output / "models.json", listing)
    write_json(args.output / "dataset-manifest.json", manifest)
    write_json(args.output / "run.json", plan)
    print(f"Output: {args.output.resolve()}", flush=True)
    comparisons = {row["id"]: {} for row in rows}
    lock = Lock()
    completed = 0
    total = plan["answers_per_model"] * len(MODELS)

    def evaluate(row, model):
        turn_index = 0

        def generate(history):
            nonlocal turn_index, completed
            turn_index += 1
            payload = {"model": model, "messages": history, **settings}
            started = monotonic()
            response = request_json(args.base_url.rstrip("/") + "/chat/completions", payload)
            elapsed = monotonic() - started
            choice = response["choices"][0]
            content = choice["message"].get("content")
            if not isinstance(content, str) or not content.strip():
                raise ValueError(f"Missing answer: {row['id']} {model} T{turn_index}")
            record = {
                "id": row["id"], "turn": turn_index, "model": model,
                "request": payload, "response": response, "elapsed_seconds": elapsed,
            }
            with lock:
                with (args.output / "requests.jsonl").open("a", encoding="utf-8") as output:
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                completed += 1
                print(f"Answers saved: {completed}/{total}", flush=True)
            return {
                "content": content,
                "finish_reason": choice["finish_reason"],
                "truncated": choice["finish_reason"] == "length",
                "usage": response.get("usage"),
                "reasoning": choice["message"].get("reasoning") or choice["message"].get("reasoning_content"),
            }

        return generate_dialogue(row, generate)

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            pending = {pool.submit(evaluate, row, model): (row["id"], model)
                       for row in rows for model in MODELS}
            for future in as_completed(pending):
                row_id, model = pending[future]
                comparisons[row_id][model] = future.result()
        with (args.output / "comparison.jsonl").open("w", encoding="utf-8") as output:
            for row in rows:
                output.write(json.dumps({"id": row["id"], "variants": comparisons[row["id"]]}, ensure_ascii=False) + "\n")
        write_blind_review(args.output, rows, comparisons)
        plan["status"] = "complete"
        plan["answers_saved"] = completed
        plan["truncated_answers"] = sum(t["truncated"] for v in comparisons.values() for d in v.values() for t in d["turns"])
        print(f"Complete. Truncated answers: {plan['truncated_answers']}", flush=True)
    except Exception as error:
        plan["status"] = "failed"
        plan["error"] = repr(error)
        raise
    finally:
        plan["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(args.output / "run.json", plan)


if __name__ == "__main__":
    main()

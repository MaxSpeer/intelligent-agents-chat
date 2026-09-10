#!/usr/bin/env python3
"""Audit saved comparisons and compute descriptive paired metrics; no API calls."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from statistics import mean, median

from evaluate_plain_english import DATA, load_dialogues, write_json


MODELS = ("qwen3-8b", "plain-english")
WORD = re.compile(r"\b\w+(?:['’\-]\w+)*\b", re.UNICODE)


def summarize(root):
    run = json.loads((root / "run.json").read_text())
    assert run["status"] == "complete"
    manifest, dialogues = load_dialogues(DATA)
    by_id = {r["id"]: r for r in dialogues}
    requests = [json.loads(line) for line in (root / "requests.jsonl").read_text().splitlines()]
    comparison = [json.loads(line) for line in (root / "comparison.jsonl").read_text().splitlines()]
    reviews = json.loads((root / "qualitative-review.json").read_text())["reviews"]
    key = json.loads((root / "blind-key.json").read_text())
    assert len(requests) == 24 and len(comparison) == 10 and len(reviews) == 12
    assert len({(r["id"], r["model"], r["turn"]) for r in requests}) == 24
    assert run["dataset_sha256"] == manifest["file_sha256"]["test.jsonl"]
    seen = {(r["id"], r["model"], r["turn"]): r for r in requests}
    rows = []
    blocks = ["# Base model and Plain English: complete saved answers\n",
              "Measured responses are reproduced verbatim below. See the report for quality caveats.\n"]
    for row in comparison:
        original = by_id[row["id"]]
        assert set(row["variants"]) == set(MODELS)
        questions = [m["content"] for m in original["messages"] if m["role"] == "user"]
        for model in MODELS:
            history = [dict(original["messages"][0])]
            variant = row["variants"][model]
            assert len(variant["turns"]) == len(questions)
            for index, question in enumerate(questions, 1):
                history.append({"role": "user", "content": question})
                request = seen[row["id"], model, index]
                assert request["request"]["messages"] == history
                assert request["request"]["model"] == model
                assert request["response"]["model"] == model
                for setting, value in run["settings"].items():
                    assert request["request"][setting] == value
                turn = variant["turns"][index - 1]
                choice = request["response"]["choices"][0]
                assert choice["message"]["content"] == turn["content"]
                assert choice["finish_reason"] == turn["finish_reason"]
                assert not turn["reasoning"]
                history.append({"role": "assistant", "content": turn["content"]})
            assert variant["messages"] == history
        for index, question in enumerate(questions, 1):
            review_key = f"{row['id']}/T{index}"
            review = next(r for r in reviews if r["key"] == review_key)
            winner = review["accessibility_winner"]
            winner = "tie" if winner == "tie" else key[review_key][winner]
            measured = {"id": row["id"], "turn": index, "question": question,
                        "accessibility_winner": winner, "review_reason": review["reason"]}
            blocks.extend([f"## {review_key}\n", f"Question: {question}\n"])
            for model in MODELS:
                turn = row["variants"][model]["turns"][index - 1]
                measured[model] = {
                    "words": len(WORD.findall(turn["content"])),
                    "completion_tokens": turn["usage"]["completion_tokens"],
                    "finish_reason": turn["finish_reason"],
                }
                blocks.extend([f"### {model}\n", turn["content"] + "\n",
                               f"Finish: `{turn['finish_reason']}`\n"])
            measured["word_reduction_percent"] = 100 * (1 - measured["plain-english"]["words"] / measured["qwen3-8b"]["words"])
            measured["both_complete"] = all(measured[m]["finish_reason"] == "stop" for m in MODELS)
            measured["content_issues"] = [{**issue, "model": key[review_key][issue["label"]]}
                                          for issue in review["content_issues"]]
            rows.append(measured)

    populations = {
        "first_answers": [r for r in rows if r["turn"] == 1],
        "complete_first_answer_pairs": [r for r in rows if r["turn"] == 1 and r["both_complete"]],
        "followups": [r for r in rows if r["turn"] > 1],
        "all_answers": rows,
    }
    aggregates = {}
    for name, population in populations.items():
        assert population
        by_model = {m: {"mean_words": mean(r[m]["words"] for r in population),
                        "median_words": median(r[m]["words"] for r in population),
                        "total_words": sum(r[m]["words"] for r in population),
                        "mean_completion_tokens": mean(r[m]["completion_tokens"] for r in population)}
                    for m in MODELS}
        aggregates[name] = {
            "pairs": len(population), "models": by_model,
            "word_reduction_percent": 100 * (1 - by_model["plain-english"]["total_words"] / by_model["qwen3-8b"]["total_words"]),
            "median_paired_word_reduction_percent": median(r["word_reduction_percent"] for r in population),
            "adapter_shorter_pairs": sum(r["plain-english"]["words"] < r["qwen3-8b"]["words"] for r in population),
            "accessibility_wins": dict(Counter(r["accessibility_winner"] for r in population)),
        }
    summary = {
        "temperature": run["settings"]["temperature"],
        "source_files": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                         for name in ("requests.jsonl", "comparison.jsonl", "qualitative-review.json", "blind-key.json")},
        "checks": {"24_unique_requests": True, "10_dialogues_12_answers_each": True,
                   "identical_initial_prompts_and_sampling_settings": True,
                   "followups_use_own_generated_answers": True,
                   "no_reference_assistant_answers_in_prompts": True,
                   "requested_and_returned_model_ids_match": True,
                   "no_reasoning_output": True},
        "word_definition": WORD.pattern,
        "word_count_caveat": "Includes headings, list numbers and formula symbols matching the regex; measures length, not reading level.",
        "aggregates": aggregates, "rows": rows,
    }
    write_json(root / "summary.json", summary)
    (root / "comparison.md").write_text("\n".join(blocks), encoding="utf-8")
    print(json.dumps({"directory": str(root), "checks": summary["checks"], "aggregates": aggregates}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    summarize(parser.parse_args().directory)

"""Audit saved validation outputs locally; never load a model or use the network."""

import hashlib
import json
import re
import statistics
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
VARIANTS = (
    "base-neutral", "base-style-prompt", "checkpoint-6", "checkpoint-12", "checkpoint-18",
)
WORD = re.compile(r"\b\w+(?:['’\-]\w+)*\b")


def read_json(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def describe(turns):
    return {
        "answers": len(turns),
        "mean_words": statistics.mean(t["words"] for t in turns),
        "median_words": statistics.median(t["words"] for t in turns),
        "mean_output_tokens": statistics.mean(t["output_tokens"] for t in turns),
        "truncated_answers": sum(t["truncated"] for t in turns),
    }


def main():
    manifest = read_json(HERE / "source-manifest.json")
    for name, sha in manifest["files"].items():
        assert hashlib.sha256((RAW / name).read_bytes()).hexdigest() == sha, name
    run = read_json(RAW / "run.json")
    assert run["status"] == "training_complete_checkpoint_selection_pending"
    assert run["checkpoints"] == ["checkpoint-6", "checkpoint-12", "checkpoint-18"]
    for name, sha in run["code_sha256"].items():
        assert hashlib.sha256((RAW / "run-inputs" / name).read_bytes()).hexdigest() == sha
    for name in ("train.jsonl", "validation.jsonl"):
        assert hashlib.sha256((RAW / "run-inputs" / name).read_bytes()).hexdigest() == run["dataset_file_sha256"][name]
    weights = manifest["weights"]
    assert weights["adapter_model.safetensors"] == weights["checkpoint-18/adapter_model.safetensors"]
    assert len({weights[f"checkpoint-{step}/adapter_model.safetensors"]["sha256"] for step in (6, 12, 18)}) == 3
    source = rows(RAW / "run-inputs/validation.jsonl")
    assert len(source) == len({r["id"] for r in source}) == 10
    answers = {}
    metadata = {}
    summaries = {}
    comparisons = []
    for variant in VARIANTS:
        generated = rows(RAW / "validation_samples" / f"{variant}.jsonl")
        meta = read_json(RAW / "validation_samples" / f"{variant}.metadata.json")
        assert meta["status"] == "complete"
        assert meta["settings"] == {"max_new_tokens": 2048, "do_sample": True, "temperature": 0.2, "top_p": 1.0, "top_k": 0, "min_p": 0.0, "repetition_penalty": 1.0, "use_cache": True}
        assert meta["seed"] == 42 and meta["enable_thinking"] is False
        assert meta["reference_answers_in_input"] is False and meta["followups_use_own_answers"] is True
        assert [r["id"] for r in generated] == [r["id"] for r in source]
        for original, row in zip(source, generated):
            assert [m for m in row["messages"] if m["role"] == "user"] == [m for m in original["messages"] if m["role"] == "user"]
            expected_system = meta["extra_style_instruction"] or original["messages"][0]["content"]
            assert row["messages"][0] == {"role": "system", "content": expected_system}
            assert [m["role"] for m in row["messages"]] == [m["role"] for m in original["messages"]]
            assert [m["content"] for m in row["messages"] if m["role"] == "assistant"] == [t["content"] for t in row["turns"]]
            for turn_index, turn in enumerate(row["turns"], 1):
                assert turn["turn"] == turn_index and turn["content"].strip()
                assert len(WORD.findall(turn["content"])) == turn["words"]
                assert 0 < turn["output_tokens"] <= 2048
                assert not turn["truncated"] or turn["output_tokens"] == 2048
        turns = [t for r in generated for t in r["turns"]]
        first = [t for t in turns if t["turn"] == 1]
        followups = [t for t in turns if t["turn"] > 1]
        assert len(turns) == meta["answers"] == 13
        assert len(first) == meta["first_answers"] == 10
        assert describe(first)["mean_words"] == meta["mean_first_answer_words"]
        assert sum(t["truncated"] for t in turns) == meta["truncated_answers"]
        summaries[variant] = {"first_answers": describe(first), "followups": describe(followups), "all_answers": describe(turns)}
        metadata[variant] = meta
        answers[variant] = generated
    for index, original in enumerate(source):
        questions = [m["content"] for m in original["messages"] if m["role"] == "user"]
        for turn_index, question in enumerate(questions):
            comparisons.append({
                "id": original["id"], "turn": turn_index + 1, "question": question,
                "variants": {variant: answers[variant][index]["turns"][turn_index] for variant in VARIANTS},
            })
    for variant in VARIANTS[1:]:
        for segment, wanted in (("first_answers", lambda p: p["turn"] == 1), ("followups", lambda p: p["turn"] > 1)):
            paired = [p for p in comparisons if wanted(p)]
            deltas = [p["variants"][variant]["words"] - p["variants"]["base-neutral"]["words"] for p in paired]
            baseline = summaries["base-neutral"][segment]["mean_words"]
            summaries[variant][segment].update(
                mean_word_change_percent=(summaries[variant][segment]["mean_words"] / baseline - 1) * 100,
                shorter_than_base=sum(d < 0 for d in deltas), longer_than_base=sum(d > 0 for d in deltas),
                same_length_as_base=sum(d == 0 for d in deltas),
            )
    state = read_json(RAW / "checkpoint-18/trainer_state.json")
    assert state["global_step"] == 18 and state["epoch"] == 3.0
    losses = [{k: entry[k] for k in ("step", "epoch", "eval_loss")} for entry in state["log_history"] if "eval_loss" in entry]
    incomplete_ids = {p["id"] for p in comparisons if any(t["truncated"] for t in p["variants"].values())}
    sensitivity = {
        "excluded_dialogues_from_every_variant": sorted(incomplete_ids),
        "reason": "Additional sensitivity check only; the main analysis retains every answer.",
        "first_answers": {
            variant: describe([p["variants"][variant] for p in comparisons if p["turn"] == 1 and p["id"] not in incomplete_ids])
            for variant in VARIANTS
        },
    }
    result = {
        "validation": "passed", "question": "Do saved v2 checkpoints answer more simply than the base model?",
        "job_id": run["job_id"], "started_at": run["started_at"], "finished_at": run["finished_at"],
        "elapsed_seconds": (datetime.fromisoformat(run["finished_at"]) - datetime.fromisoformat(run["started_at"])).total_seconds(),
        "source_files_verified": len(manifest["files"]), "dialogues": 10, "answers_per_variant": 13,
        "variants": summaries, "training_validation_losses": losses,
        "complete_dialogues_sensitivity": sensitivity,
        "word_definition": WORD.pattern,
        "limitations": [
            "Word count is descriptive, not a measure of simplicity or factual accuracy.",
            "Repetition consisting of punctuation and blank rows is not captured by word counts.",
            "The two truncated style-prompt answers remain included; no silent exclusions.",
            "Validation set used for checkpoint selection; no fresh test result.",
            "Same 4-bit training model; these are not vLLM or full-precision deployment results.",
            "One seed and sampling setting; follow-up inputs include each variant's own prior answers.",
            "This is not a controlled comparison with the prior v1 run, which used different questions and inference.",
        ],
    }
    (HERE / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    (HERE / "comparison.jsonl").write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in comparisons))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

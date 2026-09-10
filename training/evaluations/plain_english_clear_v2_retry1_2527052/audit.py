"""Audit saved validation outputs locally; no model, GPU or network calls."""
import hashlib
import json
from pathlib import Path
import re
import statistics

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
WORD = re.compile(r"\b\w+(?:['’\-]\w+)*\b")
VARIANTS = ("base-neutral", "base-style-prompt", "checkpoint-193", "checkpoint-386", "checkpoint-579")


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def initial_question(row):
    return " ".join(WORD.findall(row["messages"][1]["content"].lower()))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def describe(turns):
    units = [s.strip() for t in turns for s in re.split(r"(?<=[.!?])\s+|\n+", t["content"]) if s.strip()]
    return {
        "answers": len(turns),
        "mean_words": round(statistics.mean(t["words"] for t in turns), 3),
        "mean_sentence_words_proxy": round(statistics.mean(len(WORD.findall(s)) for s in units), 3),
        "truncated_answers": sum(t["truncated"] for t in turns),
    }


def main():
    snapshot, run, manifest = read(HERE / "source-snapshot.json"), read(RAW / "run.json"), read(RAW / "manifest.json")
    for name, expected in {**snapshot["remote_verified_files"], **snapshot["prior_local_comparison_hashes"]}.items():
        require(sha(RAW / name) == expected, f"Changed artifact: {name}")
    require(run["status"] == "training_complete_checkpoint_selection_pending", "Training incomplete")
    require(run["expected_optimizer_steps"] == 579, "Unexpected training schedule")
    for split in ("train", "validation"):
        require(sha(RAW / "run-inputs" / f"{split}.jsonl") == run["dataset_file_sha256"][f"{split}.jsonl"], "Archived dataset mismatch")
    validation = {r["id"]: r for r in rows(RAW / "run-inputs/validation.jsonl")}
    train = rows(RAW / "run-inputs/train.jsonl")
    require(not {r["id"] for r in train} & set(validation), "Train/validation ID overlap")
    require(not {initial_question(r) for r in train} & {initial_question(r) for r in validation.values()}, "Exact initial question overlap")
    expected_settings = {"max_new_tokens": 2048, "do_sample": True, "temperature": 0.2,
                         "top_p": 1.0, "top_k": 0, "min_p": 0.0, "repetition_penalty": 1.0, "use_cache": True}
    generations, stats = {}, {}
    for name in (*VARIANTS, "previous-simplified-checkpoint-579"):
        prefix = RAW / "validation_samples" / name if name in VARIANTS else RAW / name
        data, meta = rows(prefix.with_suffix(".jsonl")), read(prefix.with_suffix(".metadata.json"))
        require(meta["status"] == "complete" and meta["settings"] == expected_settings and meta["seed"] == 42, f"Generation settings: {name}")
        require(meta["followups_use_own_answers"] and not meta["reference_answers_in_input"] and not meta["enable_thinking"], "Unexpected generation context")
        require(meta["inference_quantization"] == "same 4-bit model as training", "Quantization mismatch")
        require(meta["adapter_enabled"] == ("checkpoint" in name), "Wrong adapter condition")
        require(bool(meta["extra_style_instruction"]) == (name == "base-style-prompt"), "Wrong system condition")
        require([r["id"] for r in data] == manifest["validation_generation_ids"], "Different validation questions")
        for row in data:
            original = validation[row["id"]]
            require([m for m in row["messages"] if m["role"] == "user"] == [m for m in original["messages"] if m["role"] == "user"], "Changed user turns")
            require([m["role"] for m in row["messages"]] == [m["role"] for m in original["messages"]], "Changed roles")
            system = meta["extra_style_instruction"] or original["messages"][0]["content"]
            require(row["messages"][0] == {"role": "system", "content": system}, "Changed system prompt")
            require([m["content"] for m in row["messages"] if m["role"] == "assistant"] == [t["content"] for t in row["turns"]], "Answer/history mismatch")
            for n, t in enumerate(row["turns"], 1):
                require(t["turn"] == n and len(WORD.findall(t["content"])) == t["words"], "Wrong turn/count")
                require(0 < t["output_tokens"] <= 2048 and not t["truncated"], "Incomplete answer")
        turns = [t for row in data for t in row["turns"]]
        first = [t for t in turns if t["turn"] == 1]
        require(len(turns) == meta["answers"] == 19 and len(first) == meta["first_answers"] == 10, "Wrong sample count")
        require(statistics.mean(t["words"] for t in first) == meta["mean_first_answer_words"], "Metadata differs")
        generations[name] = data
        stats[name] = {"first_answers": describe(first), "all_answers": describe(turns)}
    weights = read(HERE / "weights-check.json")["weights"]
    require(all(w["data_size_matches_header"] and w["tensor_count"] == 288 for w in weights), "Incomplete saved weights")
    require(weights[0]["sha256"] == weights[-1]["sha256"], "Root adapter is not final checkpoint")
    require(len({w["sha256"] for w in weights[1:]}) == 3, "Checkpoints have identical weights")
    state = read(RAW / "checkpoint-579/trainer_state.json")
    require(state["global_step"] == 579 and state["epoch"] == 3.0, "Final step incomplete")
    losses = [{k: r[k] for k in ("step", "epoch", "eval_loss")} for r in state["log_history"] if "eval_loss" in r]
    require([r["step"] for r in losses] == [193, 386, 579], "Missing validation epoch")
    pairs = []
    for i, row in enumerate(generations["base-neutral"]):
        questions = [m["content"] for m in row["messages"] if m["role"] == "user"]
        for j, question in enumerate(questions):
            pairs.append({"id": row["id"], "turn": j + 1, "question": question,
                          "variants": {name: data[i]["turns"][j] for name, data in generations.items()}})
    summary = {
        "audit": "passed", "job_id": run["job_id"], "status": run["status"],
        "started_at": run["started_at"], "finished_at": run["finished_at"],
        "global_step": 579, "epoch": 3, "remote_artifacts_verified": len(snapshot["remote_verified_files"]),
        "validation_dialogues_generated": 10, "validation_dialogues_total": 100,
        "answers_per_variant": 19, "root_weights_match_checkpoint_579": True,
        "validation_losses": losses, "variants": stats,
        "limitations": ["Saved 4-bit validation outputs, not a fresh vLLM or UI test.",
                        "These 10 fixed dialogues have already informed development; not an independent test benchmark.",
                        "One seed; follow-ups depend on each model's own earlier answers.",
                        "Word/sentence counts describe length; headings, lists and abbreviations affect the sentence proxy.",
                        "No human comprehension study or complete external fact check.",
                        "Validation references changed between datasets, so cross-run losses are not directly comparable."]}
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (HERE / "comparison.jsonl").write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in pairs))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

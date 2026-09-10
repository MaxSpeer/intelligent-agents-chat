"""Recompute the saved 2k comparison with Python's standard library only.

No network requests, model loading, or training. Run with python3, without -O.
Input hashes were recorded during a read-only download from the SCI cluster.
"""

import hashlib
import json
import re
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = ("qwen3-8b-plain-english-2k", "qwen3-8b-plain-english-2k-simplified-v1")
VARIANTS = ("base-neutral", "base-style-prompt", "checkpoint-193", "checkpoint-386", "checkpoint-579")
WORD = re.compile(r"\b\w+(?:['’\-]\w+)*\b")
EXPECTED_SETTINGS = {
    "max_new_tokens": 2048, "do_sample": True, "temperature": 0.2,
    "top_p": 1.0, "top_k": 0, "min_p": 0.0, "repetition_penalty": 1.0,
    "use_cache": True,
}


def read_json(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(text):
    return " ".join(WORD.findall(text.casefold()))


def describe(turns):
    return {
        "answers": len(turns),
        "mean_words": statistics.mean(t["words"] for t in turns),
        "median_words": statistics.median(t["words"] for t in turns),
        "min_words": min(t["words"] for t in turns),
        "max_words": max(t["words"] for t in turns),
        "mean_output_tokens": statistics.mean(t["output_tokens"] for t in turns),
        "truncated_answers": sum(t["truncated"] for t in turns),
    }


def pair_stats(pairs, candidate, baseline, segment):
    selected = [p for p in pairs if segment == "all_answers"
                or (p["turn"] == 1 if segment == "first_answers" else p["turn"] > 1)]
    a = [p["variants"][candidate] for p in selected]
    b = [p["variants"][baseline] for p in selected]
    return {
        "answers": len(a),
        "mean_word_change_percent": (statistics.mean(t["words"] for t in a)
                                     / statistics.mean(t["words"] for t in b) - 1) * 100,
        "shorter": sum(x["words"] < y["words"] for x, y in zip(a, b)),
        "same_length": sum(x["words"] == y["words"] for x, y in zip(a, b)),
        "longer": sum(x["words"] > y["words"] for x, y in zip(a, b)),
        "identical_text": sum(x["content"] == y["content"] for x, y in zip(a, b)),
    }


def main():
    manifest = read_json(HERE / "source-manifest.json")
    weights = read_json(HERE / "weights-check.json")
    summaries, generations, datasets, configs, metadata = {}, {}, {}, {}, {}
    verified = 0
    for name in RUNS:
        root = HERE / "raw" / name
        for relative, expected in manifest["runs"][name]["files"].items():
            assert sha256(root / relative) == expected, (name, relative)
            verified += 1
        run = read_json(root / "run.json")
        assert run["status"] == "training_complete_checkpoint_selection_pending"
        assert run["checkpoints"] == ["checkpoint-193", "checkpoint-386", "checkpoint-579"]
        w = weights["runs"][name]["weights"]
        assert w["adapter_model.safetensors"] == w["checkpoint-579/adapter_model.safetensors"]
        configs[name] = run["settings"]
        data = {}
        for split, expected_count in (("train", 2000), ("validation", 100)):
            path = root / "run-inputs" / f"{split}.jsonl"
            assert sha256(path) == run["dataset_file_sha256"][f"{split}.jsonl"]
            data[split] = rows(path)
            assert len(data[split]) == len({r["id"] for r in data[split]}) == expected_count
        assert not {r["id"] for r in data["train"]} & {r["id"] for r in data["validation"]}
        train_questions = {normalize(next(m["content"] for m in r["messages"] if m["role"] == "user"))
                           for r in data["train"]}
        val_questions = {normalize(next(m["content"] for m in r["messages"] if m["role"] == "user"))
                         for r in data["validation"]}
        assert not train_questions & val_questions
        datasets[name] = data
        by_id = {r["id"]: r for r in data["validation"]}
        fixed_ids = read_json(root / "run-inputs/manifest.json")["validation_generation_ids"]
        variant_stats = {}
        for variant in VARIANTS:
            key = f"{name}/{variant}"
            generated = rows(root / "validation_samples" / f"{variant}.jsonl")
            meta = read_json(root / "validation_samples" / f"{variant}.metadata.json")
            assert meta["status"] == "complete" and meta["split"] == "validation"
            assert meta["settings"] == EXPECTED_SETTINGS and meta["seed"] == 42
            assert meta["enable_thinking"] is False
            assert meta["reference_answers_in_input"] is False
            assert meta["followups_use_own_answers"] is True
            assert meta["adapter_enabled"] == variant.startswith("checkpoint-")
            assert meta["inference_quantization"] == "same 4-bit model as training"
            assert bool(meta["extra_style_instruction"]) == (variant == "base-style-prompt")
            assert [r["id"] for r in generated] == fixed_ids
            for row in generated:
                original = by_id[row["id"]]
                assert [m for m in row["messages"] if m["role"] == "user"] == [m for m in original["messages"] if m["role"] == "user"]
                system = meta["extra_style_instruction"] or original["messages"][0]["content"]
                assert row["messages"][0] == {"role": "system", "content": system}
                assert [m["role"] for m in row["messages"]] == [m["role"] for m in original["messages"]]
                assert [m["content"] for m in row["messages"] if m["role"] == "assistant"] == [t["content"] for t in row["turns"]]
                for i, turn in enumerate(row["turns"], 1):
                    assert turn["turn"] == i and turn["content"].strip()
                    assert len(WORD.findall(turn["content"])) == turn["words"]
                    assert 0 < turn["output_tokens"] <= 2048
                    assert not turn["truncated"] or turn["output_tokens"] == 2048
            turns = [t for r in generated for t in r["turns"]]
            first = [t for t in turns if t["turn"] == 1]
            assert len(turns) == meta["answers"] == 19
            assert len(first) == meta["first_answers"] == 10
            assert statistics.mean(t["words"] for t in first) == meta["mean_first_answer_words"]
            assert sum(t["truncated"] for t in turns) == meta["truncated_answers"] == 0
            variant_stats[variant] = {
                "first_answers": describe(first),
                "followups": describe([t for t in turns if t["turn"] > 1]),
                "all_answers": describe(turns),
            }
            generations[key], metadata[key] = generated, meta
        state = read_json(root / "checkpoint-579/trainer_state.json")
        assert state["global_step"] == 579 and state["epoch"] == 3.0
        losses = [{k: e[k] for k in ("step", "epoch", "eval_loss")} for e in state["log_history"] if "eval_loss" in e]
        summaries[name] = {
            "status": run["status"], "job_id": run["job_id"], "finished_at": run["finished_at"],
            "global_step": state["global_step"], "epoch": state["epoch"],
            "variants": variant_stats, "validation_losses": losses,
        }
    assert configs[RUNS[0]] == configs[RUNS[1]]
    for split in ("train", "validation"):
        a, b = (datasets[name][split] for name in RUNS)
        assert [r["id"] for r in a] == [r["id"] for r in b]
        assert [[m for m in r["messages"] if m["role"] != "assistant"] for r in a] == [[m for m in r["messages"] if m["role"] != "assistant"] for r in b]
    for variant in ("base-neutral", "base-style-prompt"):
        assert generations[f"{RUNS[0]}/{variant}"] == generations[f"{RUNS[1]}/{variant}"]
        assert metadata[f"{RUNS[0]}/{variant}"]["extra_style_instruction"] == metadata[f"{RUNS[1]}/{variant}"]["extra_style_instruction"]
    assert len({weights["runs"][name]["weights"]["adapter_model.safetensors"]["sha256"] for name in RUNS}) == 2
    keys = [f"{RUNS[0]}/{variant}" for variant in VARIANTS] + [f"{RUNS[1]}/{variant}" for variant in VARIANTS[2:]]
    reference = generations[keys[0]]
    pairs = []
    for i, row in enumerate(reference):
        questions = [m["content"] for m in row["messages"] if m["role"] == "user"]
        for j, question in enumerate(questions):
            pairs.append({"id": row["id"], "turn": j + 1, "question": question,
                          "variants": {key: generations[key][i]["turns"][j] for key in keys}})
    comparisons = {}
    for candidate, baseline in ((f"{RUNS[0]}/checkpoint-579", keys[0]),
                                (f"{RUNS[1]}/checkpoint-579", keys[0]),
                                (f"{RUNS[1]}/checkpoint-579", f"{RUNS[0]}/checkpoint-579")):
        comparisons[f"{candidate} vs {baseline}"] = {
            segment: pair_stats(pairs, candidate, baseline, segment)
            for segment in ("first_answers", "followups", "all_answers")
        }
    result = {
        "audit": "passed", "source_snapshot_at": manifest["checked_at"],
        "verified_source_files": verified, "dialogues": 10, "answers_per_variant": 19,
        "generated_validation_fraction": "10 of 100 validation dialogues",
        "same_questions_system_and_training_settings": True,
        "baselines_identical_across_runs": True, "root_weights_match_final_checkpoint": True,
        "train_validation_ids_and_normalized_initial_questions_disjoint": True,
        "runs": summaries, "paired_comparisons": comparisons, "word_definition": WORD.pattern,
        "limitations": [
            "Word counts measure length, not readability, factual accuracy, or completeness.",
            "Only 10 fixed validation dialogues; these are not a new independent test set.",
            "Dialogue IDs and exact initial questions are disjoint from training; semantic independence is not established.",
            "One seed and one sampling configuration; followups contain each variant's own previous answers.",
            "Answers were generated with the 4-bit training model, not the vLLM chat deployment.",
            "Validation reference answers differ between the two training runs; cross-run validation losses are not directly comparable.",
            "Human comprehension and complete factual correctness were not measured.",
        ],
    }
    (HERE / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    (HERE / "comparison.jsonl").write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in pairs))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

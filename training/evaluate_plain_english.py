#!/usr/bin/env python3
"""Compare the base model and Plain English adapter on the held-out dialogues.

Without --run, inspect paths and data only. With --run, generate answers inside
an existing Slurm GPU allocation. This script never submits a job or trains a
model. Results and recovered training metrics go into the user's home directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path


PROJECT = Path("/sc/projects/sci-lippert/intelligent-agents/project_matthias_max")
DATA = Path(__file__).resolve().parent / "datasets" / "plain_english"
DEFAULT_ADAPTER = PROJECT / "adapters" / "qwen3-8b-plain-english"


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_dialogues(directory, split="test"):
    if split not in ("train", "validation", "test"):
        raise ValueError(f"Unknown split: {split}")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    path = directory / f"{split}.jsonl"
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["file_sha256"][path.name]:
        raise ValueError(f"The {split} file differs from its dataset manifest.")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if len(rows) != manifest["split_counts"][split] or len({r["id"] for r in rows}) != len(rows):
        raise ValueError(f"Unexpected {split} count or duplicate IDs.")
    for row in rows:
        messages = row["messages"]
        expected = ["system"] + ["user", "assistant"] * ((len(messages) - 1) // 2)
        if len(messages) < 3 or [m["role"] for m in messages] != expected:
            raise ValueError(f"Invalid message roles: {row['id']}")
        if messages[0] != manifest["system_message"]:
            raise ValueError(f"Unexpected system message: {row['id']}")
    return manifest, rows


def generate_dialogue(row, generate):
    """Follow each original user question with this model's own answer.

    Reference assistant messages are deliberately never added to the input.
    """
    history = [dict(row["messages"][0])]
    turns = []
    for question in row["messages"]:
        if question["role"] != "user":
            continue
        history.append(dict(question))
        answer = generate([dict(message) for message in history])
        history.append({"role": "assistant", "content": answer["content"]})
        turns.append(answer)
    return {"messages": history, "turns": turns}


def hit_token_limit(ids, limit, eos):
    eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos])
    return len(ids) >= limit and ids[-1] not in eos_ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Generate answers in the current GPU allocation")
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--results-dir", type=Path, default=Path.home() / "plain-english-results")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")

    manifest, rows = load_dialogues(DATA)
    adapter_config = json.loads((args.adapter / "adapter_config.json").read_text())
    if adapter_config.get("base_model_name_or_path") != manifest["base_model"]:
        raise ValueError("The adapter's base model differs from the dataset manifest.")
    weights = args.adapter / "adapter_model.safetensors"
    if not weights.is_file() or weights.stat().st_size == 0:
        raise ValueError("Adapter weights are missing or empty.")
    turns_per_model = sum(m["role"] == "user" for r in rows for m in r["messages"])
    print(f"Test dialogues: {len(rows)}; assistant answers per model: {turns_per_model}", flush=True)
    if not args.run:
        print(f"Adapter: {args.adapter}")
        print("Paths and data checked. No model loaded. Use --run to generate answers.")
        return
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        raise SystemExit("Use --run inside your existing Slurm GPU allocation.")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = args.results_dir.expanduser().resolve() / f"job-{job_id}-{timestamp}"
    destination.mkdir(mode=0o700, parents=True, exist_ok=False)
    metadata = {
        "status": "running", "evaluation_job_id": job_id, "created_at_utc": timestamp,
        "base_model": manifest["base_model"], "revision": manifest["base_model_revision"],
        "adapter": str(args.adapter.resolve()),
        "adapter_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "test_sha256": manifest["file_sha256"]["test.jsonl"],
        "max_new_tokens": args.max_new_tokens, "do_sample": False,
        "enable_thinking": False, "inference_dtype": "bfloat16",
        "base_quantization": "none", "followups_use_model_generated_history": True,
    }
    write_json(destination / "run.json", metadata)
    states = list(args.adapter.glob("checkpoint-*/trainer_state.json"))
    if states:
        latest = max(states, key=lambda p: int(p.parent.name.split("-")[-1]))
        shutil.copyfile(latest, destination / "trainer_state.json")
        metadata["training_state_source"] = str(latest)
    print(f"Results: {destination}", flush=True)

    try:
        os.environ.setdefault("HF_HOME", str(PROJECT / "models" / "huggingface"))
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("No usable CUDA GPU is available.")
        if torch.cuda.get_device_capability(0)[0] < 8:
            raise RuntimeError("This evaluation uses BF16 and requires an Ampere-or-newer GPU.")
        metadata["gpu"] = torch.cuda.get_device_name(0)
        metadata["packages"] = {name: version(name) for name in ("torch", "transformers", "peft")}
        write_json(destination / "run.json", metadata)

        tokenizer = AutoTokenizer.from_pretrained(args.adapter, local_files_only=True)
        print("Loading the base model for inference...", flush=True)
        base = AutoModelForCausalLM.from_pretrained(
            manifest["base_model"], revision=manifest["base_model_revision"],
            dtype=torch.bfloat16, device_map="auto",
        )
        model = PeftModel.from_pretrained(base, args.adapter)
        model.eval()

        def generate(messages):
            encoded = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, enable_thinking=False,
                return_tensors="pt", return_dict=True,
            ).to(model.device)
            with torch.inference_mode():
                output = model.generate(
                    **encoded, max_new_tokens=args.max_new_tokens, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            ids = output[0, encoded["input_ids"].shape[1]:].tolist()
            content = tokenizer.decode(ids, skip_special_tokens=True).strip()
            return {
                "content": content, "output_tokens": len(ids), "words": len(content.split()),
                "truncated": hit_token_limit(ids, args.max_new_tokens, model.generation_config.eos_token_id),
            }

        stats = {label: [] for label in ("base", "adapter")}
        with (destination / "comparison.jsonl").open("x", encoding="utf-8") as output_file, \
             (destination / "comparison.md").open("x", encoding="utf-8") as readable:
            readable.write("# Plain English: base model and adapter\n\n")
            readable.write("Both models receive identical settings. Follow-ups use each model's own answers.\n\n")
            for number, row in enumerate(rows, 1):
                with model.disable_adapter():
                    baseline = generate_dialogue(row, generate)
                adapted = generate_dialogue(row, generate)
                pair = {"id": row["id"], "base": baseline, "adapter": adapted}
                output_file.write(json.dumps(pair, ensure_ascii=False) + "\n")
                output_file.flush()
                readable.write(f"## {row['id']}\n\n")
                for index, question in enumerate(baseline["messages"][1::2]):
                    readable.write(f"### Question {index + 1}\n\n{question['content']}\n\n")
                    for label, dialogue in (("Base model", baseline), ("Adapter", adapted)):
                        answer = dialogue["turns"][index]
                        readable.write(f"**{label}**\n\n{answer['content']}\n\n")
                        if answer["truncated"]:
                            readable.write("*Output reached the token limit; treat this answer as incomplete.*\n\n")
                readable.flush()
                stats["base"].extend(baseline["turns"])
                stats["adapter"].extend(adapted["turns"])
                print(f"Completed {number}/{len(rows)} dialogues ({row['id']}).", flush=True)

        summary = {label: {
            "answers": len(turns), "mean_words": round(sum(t["words"] for t in turns) / len(turns), 1),
            "truncated_answers": sum(t["truncated"] for t in turns),
        } for label, turns in stats.items()}
        summary["note"] = "Word count measures length, not correctness or explanation quality."
        write_json(destination / "summary.json", summary)
        metadata["status"] = "complete"
        write_json(destination / "run.json", metadata)
        print(f"Comparison complete. Read: {destination / 'comparison.md'}", flush=True)
    except Exception as error:
        metadata.update(status="failed", error_type=type(error).__name__, error=str(error))
        write_json(destination / "run.json", metadata)
        raise


if __name__ == "__main__":
    main()

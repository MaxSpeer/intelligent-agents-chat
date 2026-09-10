#!/usr/bin/env python3
"""Inspect five known training questions; inference only when the user passes --run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from contextlib import nullcontext
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from evaluate_plain_english import hit_token_limit, write_json

PROJECT = Path("/sc/projects/sci-lippert/intelligent-agents/project_matthias_max")
ADAPTER = PROJECT / "adapters/qwen3-8b-plain-english-v2"
BASE_MODEL = "Qwen/Qwen3-8B"
REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
IDS = ("plain-en-0088", "plain-en-0067", "plain-en-0036", "plain-en-0077", "plain-en-0011")
WORD = re.compile(r"\b\w+(?:['’\-]\w+)*\b")
SETTINGS = {
    "max_new_tokens": 1024, "do_sample": True, "temperature": 0.2,
    "top_p": 1.0, "top_k": 0, "min_p": 0.0, "repetition_penalty": 1.0,
    "use_cache": True,
}


def first_question(row):
    messages = row["messages"]
    if [m["role"] for m in messages[:3]] != ["system", "user", "assistant"]:
        raise ValueError(f"Unexpected conversation: {row['id']}")
    return [dict(m) for m in messages[:2]], messages[2]["content"]


def preflight(adapter):
    run = json.loads((adapter / "run.json").read_text())
    if run["status"] != "training_complete_checkpoint_selection_pending":
        raise ValueError("Expected a completed v2 training run")
    if run["settings"]["BASE_MODEL"] != BASE_MODEL or run["settings"]["REVISION"] != REVISION:
        raise ValueError("Unexpected base model or revision")
    source = adapter / "run-inputs/train.jsonl"
    data = source.read_bytes()
    if hashlib.sha256(data).hexdigest() != run["dataset_file_sha256"]["train.jsonl"]:
        raise ValueError("Saved training data checksum mismatch")
    rows = [json.loads(line) for line in data.decode().splitlines()]
    by_id = {row["id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("Duplicate training IDs")
    selected = [by_id[identifier] for identifier in IDS]
    for row in selected:
        first_question(row)
    config = json.loads((adapter / "adapter_config.json").read_text())
    if config["base_model_name_or_path"] != BASE_MODEL:
        raise ValueError("Adapter base model differs")
    weights = adapter / "adapter_model.safetensors"
    if not weights.is_file() or weights.stat().st_size == 0:
        raise ValueError("Adapter weights missing")
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        if not (adapter / name).is_file():
            raise ValueError(f"Missing tokenizer file: {name}")
    plan = {
        "status": "checked_not_started", "base_model": BASE_MODEL, "revision": REVISION,
        "adapter": str(adapter), "adapter_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "checkpoint": "final epoch 3; diagnostic only, not selected as best",
        "split": "train", "ids": list(IDS), "first_questions_only": True,
        "planned_model_answers": 10, "training": False, "test_questions_used": False,
        "references_in_model_input": False, "settings": SETTINGS,
        "seed_rule": "42 + 100 * (one-based selected question index) + 1; same for both variants",
        "inference_quantization": "4-bit NF4, double quantization, BF16 compute",
        "training_snapshot_sha256": hashlib.sha256(data).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "limitation": "Known training questions diagnose learning; they do not measure generalization.",
    }
    return plan, selected


def compare_question(row, model, generate):
    messages, reference = first_question(row)
    result = {"id": row["id"], "input_messages": messages, "reference_for_review_only": reference}
    for label, context in (("base", model.disable_adapter()), ("adapter", nullcontext())):
        with context:
            result[label] = generate([dict(m) for m in messages])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--adapter", type=Path, default=ADAPTER)
    args = parser.parse_args()
    if args.run and not os.environ.get("SLURM_JOB_ID"):
        parser.error("--run requires your existing Slurm GPU allocation")
    plan, selected = preflight(args.adapter)
    print(json.dumps(plan, indent=2), flush=True)
    if not args.run:
        print("Diagnostic preflight passed. No model loaded and no inference started.")
        return

    os.environ.setdefault("HF_HOME", str(PROJECT / "models/huggingface"))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit("Exactly one visible CUDA GPU is required")
    if torch.cuda.get_device_capability(0)[0] < 8:
        raise SystemExit("An Ampere-or-newer GPU is required")
    if torch.cuda.mem_get_info()[0] < 16 * 1024**3:
        raise SystemExit("Less than 16 GiB GPU memory free")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = Path.home() / "plain-english-results" / f"diagnostic-v2-{os.environ['SLURM_JOB_ID']}-{stamp}"
    destination.mkdir(parents=True, exist_ok=False)
    plan.update(status="running", job_id=os.environ["SLURM_JOB_ID"],
                created_at=datetime.now(timezone.utc).isoformat(),
                packages={name: version(name) for name in ("torch", "transformers", "peft", "bitsandbytes")})
    write_json(destination / "run.json", plan)
    (destination / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print(f"Results: {destination}", flush=True)
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.adapter, local_files_only=True)
        print("Loading the pinned base model in 4-bit; no training...", flush=True)
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                  bnb_4bit_compute_dtype=torch.bfloat16,
                                  bnb_4bit_use_double_quant=True)
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, revision=REVISION, torch_dtype=torch.bfloat16,
            device_map="auto", quantization_config=quant,
        )
        # Match the training run's non-quantized parameter casts without enabling training.
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=False)
        model = PeftModel.from_pretrained(base, args.adapter, is_trainable=False)
        model.eval()
        model.config.use_cache = True
        for name in ("bos_token_id", "pad_token_id", "eos_token_id"):
            value = getattr(tokenizer, name)
            if name != "eos_token_id":
                setattr(model.config, name, value)
                setattr(model.generation_config, name, value)
        with (destination / "comparison.jsonl").open("x") as raw, \
                (destination / "comparison.md").open("x") as readable:
            readable.write("# Known training questions: base and final v2 adapter\n\n")
            readable.write("References are shown for review only and never supplied to the model.\n\n")
            for number, row in enumerate(selected, 1):
                def generate(messages):
                    torch.manual_seed(42 + 100 * number + 1)
                    encoded = tokenizer.apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=True,
                        enable_thinking=False, return_tensors="pt", return_dict=True,
                    ).to(model.device)
                    with torch.inference_mode():
                        output = model.generate(**encoded, **SETTINGS,
                                                pad_token_id=tokenizer.pad_token_id)
                    ids = output[0, encoded["input_ids"].shape[1]:].tolist()
                    content = tokenizer.decode(ids, skip_special_tokens=True)
                    if not content.strip():
                        raise ValueError("Empty generated answer")
                    return {"content": content, "words": len(WORD.findall(content)),
                            "output_tokens": len(ids),
                            "truncated": hit_token_limit(ids, SETTINGS["max_new_tokens"],
                                                         model.generation_config.eos_token_id)}

                pair = compare_question(row, model, generate)
                raw.write(json.dumps(pair, ensure_ascii=False) + "\n")
                raw.flush()
                readable.write(f"## {row['id']}\n\n{pair['input_messages'][1]['content']}\n\n")
                for label in ("base", "adapter"):
                    answer = pair[label]
                    readable.write(f"### {label}\n\n{answer['content']}\n\n")
                    readable.write(f"Words: {answer['words']}; truncated: {answer['truncated']}\n\n")
                readable.write(f"### Training reference (review only)\n\n{pair['reference_for_review_only']}\n\n")
                readable.flush()
                print(f"Diagnostic: {number}/5 questions complete (base + adapter)", flush=True)
        plan["status"] = "complete"
        print(f"Diagnostic complete. Read: {destination / 'comparison.md'}", flush=True)
    except Exception as error:
        plan.update(status="failed", error=repr(error))
        raise
    finally:
        plan["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(destination / "run.json", plan)


if __name__ == "__main__":
    main()

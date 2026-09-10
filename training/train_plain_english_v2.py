#!/usr/bin/env python3
"""Check the corrected Plain-English run; train only with explicit --run.

This script never allocates a GPU or submits a Slurm job. The user must start
--run inside their own single-GPU allocation. The existing adapter is preserved.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from chat_examples import build_examples
from evaluate_plain_english import load_dialogues, write_json


HERE = Path(__file__).resolve().parent
PROJECT = Path("/sc/projects/sci-lippert/intelligent-agents/project_matthias_max")
DATA = HERE / "datasets" / "plain_english"
OUTPUT = PROJECT / "adapters" / "qwen3-8b-plain-english-v2"
SETTINGS = {
    "BASE_MODEL": "Qwen/Qwen3-8B",
    "REVISION": "b968826d9c46dd6066d109eabc6255188de91218",
    "NUM_EPOCHS": 3.0, "PER_DEVICE_BATCH_SIZE": 1, "GRAD_ACCUM_STEPS": 16,
    "LEARNING_RATE": 1e-4, "LORA_R": 8, "LORA_ALPHA": 16, "LORA_DROPOUT": 0.05,
    "TARGET_MODULES": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "MAX_SEQ_LEN": 1024, "WARMUP_RATIO": 0.03, "SEED": 42,
    "USE_4BIT": True, "USE_GRADIENT_CHECKPOINTING": True,
    "LOGGING_STEPS": 1, "CHECKPOINT_STRATEGY": "epoch", "SAVE_TOTAL_LIMIT": 3,
    "LOAD_BEST_MODEL_AT_END": False,
}
CODE_FILES = (
    "train_plain_english_v2.py", "train_lora.py", "chat_examples.py",
    "validation_samples.py", "evaluate_plain_english.py", "datasets/plain_english/validate.py",
)


def preflight(tokenizer_path: Path, output_dir: Path):
    from transformers import AutoTokenizer

    spec = importlib.util.spec_from_file_location("plain_english_validation", DATA / "validate.py")
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    report = validator.validate(DATA, tokenizer_path)
    manifest, train_rows = load_dialogues(DATA, "train")
    _, validation_rows = load_dialogues(DATA, "validation")
    if manifest["base_model"] != SETTINGS["BASE_MODEL"] or \
            manifest["base_model_revision"] != SETTINGS["REVISION"]:
        raise ValueError("Dataset and pinned base model differ")
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    examples = sum(len(build_examples(tokenizer, row["messages"], SETTINGS["MAX_SEQ_LEN"]))
                   for row in train_rows)
    steps = math.ceil(examples / (SETTINGS["PER_DEVICE_BATCH_SIZE"] * SETTINGS["GRAD_ACCUM_STEPS"]))
    plan = {
        "status": "checked_not_started", "settings": SETTINGS,
        "output_dir": str(output_dir), "tokenizer_path": str(tokenizer_path),
        "tokenizer_only_from_snapshot": True,
        "model_initialization": "fresh pinned base; no existing adapter is loaded",
        "dataset_file_sha256": manifest["file_sha256"],
        "train_conversations": len(train_rows), "train_answer_examples": examples,
        "validation_conversations": len(validation_rows),
        "expected_optimizer_steps_per_epoch": steps,
        "expected_optimizer_steps": steps * int(SETTINGS["NUM_EPOCHS"]),
        "checkpoint_selection": "manual after reviewing validation answers; root is final epoch",
        "validation_sampling": "base with neutral/style prompt, then each saved epoch",
        "test_generations": False, "validation_report": report,
        "code_sha256": {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                        for name in CODE_FILES},
        "packages": {name: version(name) for name in
                     ("torch", "transformers", "peft", "bitsandbytes")},
    }
    return plan, tokenizer, validation_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Validate only (the default)")
    mode.add_argument("--run", action="store_true", help="Train in the user's GPU allocation")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    manifest = json.loads((DATA / "manifest.json").read_text())
    parser.add_argument("--tokenizer", type=Path, default=Path(manifest["tokenizer"]["source_path"]))
    args = parser.parse_args()
    if args.run and not os.environ.get("SLURM_JOB_ID"):
        parser.error("--run requires the user's existing Slurm GPU allocation")
    output_dir = args.output_dir.expanduser().resolve()
    plan, tokenizer, validation_rows = preflight(args.tokenizer.expanduser().resolve(), output_dir)
    print(json.dumps(plan, indent=2), flush=True)
    if not args.run:
        print("Preflight passed. No model weights loaded and no training started.")
        return

    import train_lora as training
    from validation_samples import ValidationSamplesCallback

    if not training.torch.cuda.is_available() or training.torch.cuda.device_count() != 1:
        raise SystemExit("This run requires exactly one visible CUDA GPU")
    if training.torch.cuda.get_device_capability(0)[0] < 8:
        raise SystemExit("BF16 training requires an Ampere-or-newer GPU (e.g. A40)")
    free_bytes, _ = training.torch.cuda.mem_get_info()
    if free_bytes < 16 * 1024**3:
        raise SystemExit("Less than 16 GiB GPU memory free. Use an available GPU; "
                         "a running vLLM server may still occupy this allocation.")
    for key, value in SETTINGS.items():
        setattr(training, key, value)
    training.TRAIN_FILE = DATA / "train.jsonl"
    training.VAL_FILE = DATA / "validation.jsonl"
    training.OUTPUT_DIR = output_dir
    output_dir.mkdir(parents=True, exist_ok=False)
    inputs = output_dir / "run-inputs"
    inputs.mkdir()
    for name in CODE_FILES:
        destination = inputs / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((HERE / name).read_bytes())
    for name in ("train.jsonl", "validation.jsonl", "manifest.json"):
        (inputs / name).write_bytes((DATA / name).read_bytes())
    plan.update(status="running", job_id=os.environ["SLURM_JOB_ID"],
                started_at=datetime.now(timezone.utc).isoformat(),
                gpu=training.torch.cuda.get_device_name(0))
    write_json(output_dir / "run.json", plan)
    try:
        training.main(tokenizer=tokenizer, callback_factory=lambda model, tok: [
            ValidationSamplesCallback(model, tok, validation_rows, output_dir,
                                      seed=SETTINGS["SEED"]),
        ])
        plan["status"] = "training_complete_checkpoint_selection_pending"
        plan["checkpoints"] = [p.name for p in sorted(
            output_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]),
        )]
        print("Training complete. Review validation_samples before selecting a checkpoint.")
    except Exception as error:
        plan.update(status="failed", error=repr(error))
        raise
    finally:
        plan["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(output_dir / "run.json", plan)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Resume a saved Plain-English run; --check is read-only and needs only Python.

--run requires the user's existing single-GPU Slurm allocation. The archived
training code and datasets are reused. No jobs are allocated or submitted here.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import tempfile


CHECKPOINT_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "optimizer.pt",
    "scheduler.pt",
    "rng_state.pth",
    "trainer_state.json",
    "training_args.bin",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
)
ARCHIVED_MODULES = ("train_lora", "chat_examples", "evaluate_plain_english", "validation_samples")


def sha256(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    """Replace a status record atomically, leaving the previous record on errors."""
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            if path.exists():
                os.fchmod(output.fileno(), path.stat().st_mode & 0o777)
            json.dump(value, output, indent=2, ensure_ascii=False)
            output.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def preflight(checkpoint):
    checkpoint = checkpoint.expanduser().resolve()
    output = checkpoint.parent
    run = read_json(output / "run.json")
    if Path(run["output_dir"]).resolve() != output:
        raise ValueError("Checkpoint and recorded output directory differ")
    if (
        run["status"].startswith("training_complete")
        or (output / "adapter_model.safetensors").exists()
    ):
        raise ValueError("Run already has a final adapter; refusing to overwrite it")
    for name in CHECKPOINT_FILES:
        if not (checkpoint / name).is_file() or not (checkpoint / name).stat().st_size:
            raise ValueError(f"Incomplete checkpoint: {name}")
    state = read_json(checkpoint / "trainer_state.json")
    step, total = state["global_step"], run["expected_optimizer_steps"]
    if checkpoint.name != f"checkpoint-{step}" or not 0 < step < total:
        raise ValueError("Checkpoint name/step is invalid or training is already complete")
    settings = run["settings"]
    if state["max_steps"] != total or state["num_train_epochs"] != settings["NUM_EPOCHS"]:
        raise ValueError("Checkpoint and original training schedule differ")
    if settings["CHECKPOINT_STRATEGY"] != "epoch" or settings["LOAD_BEST_MODEL_AT_END"]:
        raise ValueError("Only the fixed-epoch Plain-English runs are supported")
    if step % run["expected_optimizer_steps_per_epoch"] or state["epoch"] != int(state["epoch"]):
        raise ValueError("Resume requires a completed epoch checkpoint")
    if any(
        int(p.name.removeprefix("checkpoint-")) > step for p in output.glob("checkpoint-[0-9]*")
    ):
        raise ValueError("A newer checkpoint exists; choose the latest checkpoint")
    config = read_json(checkpoint / "adapter_config.json")
    if (
        config["base_model_name_or_path"] != settings["BASE_MODEL"]
        or config["r"] != settings["LORA_R"]
        or config["lora_alpha"] != settings["LORA_ALPHA"]
        or config["lora_dropout"] != settings["LORA_DROPOUT"]
        or set(config["target_modules"]) != set(settings["TARGET_MODULES"])
    ):
        raise ValueError("Adapter and original LoRA configuration differ")
    inputs = output / "run-inputs"
    for name in ARCHIVED_MODULES:
        if f"{name}.py" not in run["code_sha256"]:
            raise ValueError(f"Missing archived code hash: {name}")
    for name, expected in run["code_sha256"].items():
        path = inputs / name
        if not path.resolve().is_relative_to(inputs) or sha256(path) != expected:
            raise ValueError(f"Archived code changed: {name}")
    manifest = read_json(inputs / "manifest.json")
    if (
        manifest["file_sha256"] != run["dataset_file_sha256"]
        or manifest["base_model"] != settings["BASE_MODEL"]
        or manifest["base_model_revision"] != settings["REVISION"]
    ):
        raise ValueError("Archived manifest and original run differ")
    for name in ("train.jsonl", "validation.jsonl", "validation_sample_ids.json"):
        if sha256(inputs / name) != run["dataset_file_sha256"][name]:
            raise ValueError(f"Archived dataset changed: {name}")
    rows = [json.loads(line) for line in (inputs / "validation.jsonl").read_text().splitlines()]
    by_id = {row["id"]: row for row in rows}
    ids = run["validation_generation_ids"]
    selected = [by_id[identifier] for identifier in ids]
    for name in ("base-neutral", "base-style-prompt", checkpoint.name):
        path = output / "validation_samples" / name
        metadata = read_json(path.with_suffix(".metadata.json"))
        samples = [json.loads(line) for line in path.with_suffix(".jsonl").read_text().splitlines()]
        if metadata["status"] != "complete" or [row["id"] for row in samples] != ids:
            raise ValueError(f"Saved validation comparison is incomplete: {name}")
    plan = {
        "status": "resume_checked_not_started",
        "checkpoint": str(checkpoint),
        "output_dir": str(output),
        "completed_optimizer_steps": step,
        "total_optimizer_steps": total,
        "remaining_optimizer_steps": total - step,
        "completed_epochs": state["epoch"],
        "total_epochs": settings["NUM_EPOCHS"],
        "restores": ["adapter", "optimizer", "scheduler", "rng", "trainer_state"],
        "training_data": str(inputs / "train.jsonl"),
        "archived_code_and_data_hashes": "passed",
        "existing_validation_samples": "preserved; baselines are not regenerated",
        "checkpoint_sha256": {name: sha256(checkpoint / name) for name in CHECKPOINT_FILES},
        "required_packages": run["packages"],
        "test_generations": False,
    }
    return plan, run, selected


def resuming_trainer(base_class, checkpoint):
    """Supply Trainer's native resume argument to the unchanged archived main()."""

    class ResumeTrainer(base_class):
        def train(self, **kwargs):
            return super().train(resume_from_checkpoint=str(checkpoint), **kwargs)

    return ResumeTrainer


def resuming_callback(base_class, expected_step):
    class ResumeValidationSamplesCallback(base_class):
        def on_train_begin(self, args, state, control, **kwargs):
            if args.world_size != 1 or state.global_step != expected_step:
                raise ValueError("Trainer did not restore the expected single-GPU checkpoint")
            # Existing baseline samples are checked above. on_save stays unchanged.
            return control

    return ResumeValidationSamplesCallback


def run_training(plan, run, rows):
    checkpoint, output = Path(plan["checkpoint"]), Path(plan["output_dir"])
    for package, expected in run["packages"].items():
        if version(package) != expected:
            raise ValueError(f"Use the original training environment: {package} must be {expected}")
    if any(name in sys.modules for name in ARCHIVED_MODULES):
        raise RuntimeError("Start this resume command in a fresh Python process")
    inputs = output / "run-inputs"
    sys.path.insert(0, str(inputs))
    import train_lora as training
    from transformers import AutoTokenizer
    from validation_samples import ValidationSamplesCallback

    for name in ARCHIVED_MODULES:
        if Path(sys.modules[name].__file__).resolve() != inputs / f"{name}.py":
            raise RuntimeError(f"Wrong training module loaded: {name}")
    if not training.torch.cuda.is_available() or training.torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one visible CUDA GPU is required")
    if training.torch.cuda.get_device_capability(0)[0] < 8:
        raise RuntimeError("Use an Ampere-or-newer GPU, e.g. A40")
    if training.torch.cuda.mem_get_info()[0] < 16 * 1024**3:
        raise RuntimeError(
            "At least 16 GiB GPU memory must be free; stop other GPU workloads first"
        )
    for key, value in run["settings"].items():
        setattr(training, key, value)
    training.TRAIN_FILE = inputs / "train.jsonl"
    training.VAL_FILE = inputs / "validation.jsonl"
    training.OUTPUT_DIR = output
    training.Trainer = resuming_trainer(training.Trainer, checkpoint)
    callback = resuming_callback(ValidationSamplesCallback, plan["completed_optimizer_steps"])
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    attempts = output / "resume-attempts"
    attempts.mkdir(exist_ok=True)
    attempt = Path(tempfile.mkdtemp(prefix=f"job-{os.environ['SLURM_JOB_ID']}-", dir=attempts))
    (attempt / "previous-run.json").write_bytes((output / "run.json").read_bytes())
    (attempt / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    event = {
        **plan,
        "status": "running",
        "job_id": os.environ["SLURM_JOB_ID"],
        "started_at": datetime.now(timezone.utc).isoformat(),
        "resume_script_sha256": sha256(Path(__file__)),
    }
    run.setdefault("resume_history", []).append(event)
    run.update(status="running", job_id=event["job_id"], gpu=training.torch.cuda.get_device_name(0))
    run.pop("finished_at", None)
    run.pop("error", None)
    write_json(output / "run.json", run)
    write_json(attempt / "resume.json", event)
    try:
        training.main(
            tokenizer=tokenizer,
            callback_factory=lambda model, tok: [
                callback(model, tok, rows, output, seed=run["settings"]["SEED"])
            ],
        )
        final_state = read_json(
            output / f"checkpoint-{plan['total_optimizer_steps']}" / "trainer_state.json"
        )
        if final_state["global_step"] != plan["total_optimizer_steps"]:
            raise RuntimeError("Training returned before the planned final step")
        run["status"] = "training_complete_checkpoint_selection_pending"
        run["checkpoints"] = [
            p.name
            for p in sorted(
                output.glob("checkpoint-[0-9]*"),
                key=lambda p: int(p.name.removeprefix("checkpoint-")),
            )
        ]
        event["status"] = "complete"
        print("Training complete. Review validation_samples before selecting a checkpoint.")
    except BaseException as error:
        run.update(status="failed", error=repr(error))
        event.update(status="failed", error=repr(error))
        raise
    finally:
        event["finished_at"] = datetime.now(timezone.utc).isoformat()
        run["finished_at"] = event["finished_at"]
        write_json(attempt / "resume.json", event)
        write_json(output / "run.json", run)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--check", action="store_true", help="Read-only preflight (default)")
    modes.add_argument(
        "--run", action="store_true", help="Resume inside your existing GPU allocation"
    )
    args = parser.parse_args()
    if args.run and not os.environ.get("SLURM_JOB_ID"):
        parser.error("--run requires the user's existing Slurm GPU allocation")
    if not args.run:
        plan, _, _ = preflight(args.checkpoint)
        print(json.dumps(plan, indent=2))
        print("Resume preflight passed. No model loaded and no training started.")
        return
    # The kernel releases this lock even if the process is killed. It prevents
    # two resume commands from writing the same checkpoint directory concurrently.
    with (args.checkpoint.expanduser().resolve().parent / ".resume.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("Another resume process is already using this run directory")
        plan, run, rows = preflight(args.checkpoint)
        print(json.dumps(plan, indent=2), flush=True)
        run_training(plan, run, rows)


if __name__ == "__main__":
    main()

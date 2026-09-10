"""Exercise resume integrity and checkpoint wiring without a model or GPU."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import resume_plain_english as resume  # noqa: E402


class ResumeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.output = Path(temporary.name).resolve()
        self.checkpoint = self.output / "checkpoint-193"
        self.checkpoint.mkdir()
        for name in resume.CHECKPOINT_FILES:
            (self.checkpoint / name).write_bytes(b"checkpoint fixture")
        self.state = {"global_step": 193, "max_steps": 579, "epoch": 1.0, "num_train_epochs": 3}
        resume.write_json(self.checkpoint / "trainer_state.json", self.state)
        settings = {
            "BASE_MODEL": "Qwen/Qwen3-8B",
            "REVISION": "pinned-revision",
            "LORA_R": 8,
            "LORA_ALPHA": 16,
            "LORA_DROPOUT": 0.05,
            "TARGET_MODULES": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "NUM_EPOCHS": 3.0,
            "CHECKPOINT_STRATEGY": "epoch",
            "LOAD_BEST_MODEL_AT_END": False,
        }
        resume.write_json(
            self.checkpoint / "adapter_config.json",
            {
                "base_model_name_or_path": settings["BASE_MODEL"],
                "r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "target_modules": settings["TARGET_MODULES"],
            },
        )
        self.inputs = self.output / "run-inputs"
        self.inputs.mkdir()
        code_hashes = {}
        for name in resume.ARCHIVED_MODULES:
            path = self.inputs / f"{name}.py"
            path.write_text("# immutable archived training code\n")
            code_hashes[path.name] = resume.sha256(path)
        row = {"id": "validation-1", "messages": [{"role": "user", "content": "Why?"}]}
        for name in ("train.jsonl", "validation.jsonl"):
            (self.inputs / name).write_text(json.dumps(row) + "\n")
        resume.write_json(self.inputs / "validation_sample_ids.json", [row["id"]])
        hashes = {p.name: resume.sha256(p) for p in self.inputs.glob("*.json*")}
        resume.write_json(
            self.inputs / "manifest.json",
            {
                "base_model": settings["BASE_MODEL"],
                "base_model_revision": settings["REVISION"],
                "file_sha256": hashes,
            },
        )
        self.run = {
            "status": "running",
            "output_dir": str(self.output),
            "settings": settings,
            "expected_optimizer_steps": 579,
            "expected_optimizer_steps_per_epoch": 193,
            "dataset_file_sha256": hashes,
            "code_sha256": code_hashes,
            "validation_generation_ids": [row["id"]],
            "packages": {"torch": "2.10.0"},
        }
        resume.write_json(self.output / "run.json", self.run)
        samples = self.output / "validation_samples"
        samples.mkdir()
        for name in ("base-neutral", "base-style-prompt", "checkpoint-193"):
            resume.write_json(samples / f"{name}.metadata.json", {"status": "complete"})
            (samples / f"{name}.jsonl").write_text(json.dumps(row) + "\n")

    def test_preflight_preserves_original_schedule_and_archived_data(self):
        plan, run, rows = resume.preflight(self.checkpoint)
        self.assertEqual(plan["completed_optimizer_steps"], 193)
        self.assertEqual(plan["remaining_optimizer_steps"], 386)
        self.assertEqual(plan["total_epochs"], 3.0)
        self.assertEqual(plan["training_data"], str(self.inputs / "train.jsonl"))
        self.assertEqual(rows[0]["id"], "validation-1")
        self.assertEqual(run, self.run)

    def test_check_has_no_writes_and_runs_without_site_packages(self):
        before = {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in self.output.rglob("*")
            if p.is_file()
        }
        result = subprocess.run(
            [
                sys.executable,
                "-S",
                resume.__file__,
                "--checkpoint",
                str(self.checkpoint),
                "--check",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("No model loaded and no training started", result.stdout)
        after = {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in self.output.rglob("*")
            if p.is_file()
        }
        self.assertEqual(before, after)

    def test_no_start_without_user_gpu_allocation(self):
        result = subprocess.run(
            [sys.executable, "-S", resume.__file__, "--checkpoint", str(self.checkpoint), "--run"],
            env={k: v for k, v in os.environ.items() if k != "SLURM_JOB_ID"},
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("requires the user's existing Slurm GPU allocation", result.stderr)
        self.assertFalse((self.output / ".resume.lock").exists())

    def test_missing_optimizer_is_rejected_instead_of_silently_restarting_it(self):
        (self.checkpoint / "optimizer.pt").unlink()
        with self.assertRaisesRegex(ValueError, "Incomplete checkpoint: optimizer.pt"):
            resume.preflight(self.checkpoint)

    def test_changed_archived_data_is_rejected(self):
        (self.inputs / "train.jsonl").write_text("changed training answers\n")
        with self.assertRaisesRegex(ValueError, "Archived dataset changed"):
            resume.preflight(self.checkpoint)

    def test_changed_archived_training_code_is_rejected(self):
        (self.inputs / "train_lora.py").write_text("# different training\n")
        with self.assertRaisesRegex(ValueError, "Archived code changed"):
            resume.preflight(self.checkpoint)

    def test_changed_step_schedule_is_rejected(self):
        self.state["max_steps"] = 386
        resume.write_json(self.checkpoint / "trainer_state.json", self.state)
        with self.assertRaisesRegex(ValueError, "original training schedule differ"):
            resume.preflight(self.checkpoint)

    def test_existing_final_adapter_is_preserved(self):
        (self.output / "adapter_model.safetensors").write_bytes(b"final adapter")
        with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
            resume.preflight(self.checkpoint)

    def test_newer_checkpoint_prevents_rolling_back_progress(self):
        (self.output / "checkpoint-386").mkdir()
        with self.assertRaisesRegex(ValueError, "newer checkpoint exists"):
            resume.preflight(self.checkpoint)

    def test_incomplete_saved_baseline_is_rejected(self):
        resume.write_json(
            self.output / "validation_samples/base-neutral.metadata.json", {"status": "running"}
        )
        with self.assertRaisesRegex(ValueError, "comparison is incomplete"):
            resume.preflight(self.checkpoint)

    def test_archived_train_call_receives_native_resume_argument(self):
        class Trainer:
            def train(self, **kwargs):
                return kwargs

        trainer = resume.resuming_trainer(Trainer, self.checkpoint)()
        self.assertEqual(trainer.train(), {"resume_from_checkpoint": str(self.checkpoint)})

    def test_resumed_callback_skips_baselines_and_keeps_checkpoint_generation(self):
        class Callback:
            def on_train_begin(self, *args, **kwargs):
                raise AssertionError("Must not overwrite the existing baseline answers")

            def on_save(self, *args, **kwargs):
                return "new checkpoint answers"

        callback = resume.resuming_callback(Callback, 193)()
        args = SimpleNamespace(world_size=1)
        self.assertEqual(
            callback.on_train_begin(args, SimpleNamespace(global_step=193), "control"), "control"
        )
        self.assertEqual(callback.on_save(), "new checkpoint answers")
        with self.assertRaisesRegex(ValueError, "did not restore"):
            callback.on_train_begin(args, SimpleNamespace(global_step=0), None)


if __name__ == "__main__":
    unittest.main()

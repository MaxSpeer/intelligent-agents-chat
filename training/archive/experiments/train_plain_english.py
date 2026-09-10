import argparse
import os
from pathlib import Path

import train_lora as training

HERE = Path(__file__).resolve().parent
PROJECT = Path(
    "/sc/projects/sci-lippert/intelligent-agents/project_matthias_max"
)

training.TRAIN_FILE = HERE / "datasets/plain_english/train.jsonl"
training.VAL_FILE = HERE / "datasets/plain_english/validation.jsonl"
training.OUTPUT_DIR = PROJECT / "adapters/qwen3-8b-plain-english"

training.NUM_EPOCHS = 3.0
training.PER_DEVICE_BATCH_SIZE = 1
training.GRAD_ACCUM_STEPS = 16
training.LEARNING_RATE = 1e-4
training.LORA_R = 8
training.LOGGING_STEPS = 1
training.EVAL_STEPS = 5
training.SAVE_STEPS = 5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    for path in (training.TRAIN_FILE, training.VAL_FILE):
        if not path.is_file():
            raise SystemExit(f"Dataset missing: {path}")

    if training.OUTPUT_DIR.exists():
        raise SystemExit("Output directory already exists. Choose a new run directory.")

    if args.check:
        for name in (
            "BASE_MODEL", "REVISION", "TRAIN_FILE", "VAL_FILE", "OUTPUT_DIR",
            "NUM_EPOCHS", "PER_DEVICE_BATCH_SIZE", "GRAD_ACCUM_STEPS",
            "LEARNING_RATE", "LORA_R", "EVAL_STEPS", "SAVE_STEPS",
        ):
            print(f"{name}: {getattr(training, name)}")
        print("Configuration check passed. No training started.")
        return

    if not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit("Training must run inside a Slurm GPU job.")
    if not training.torch.cuda.is_available():
        raise SystemExit("No usable CUDA GPU available.")

    training.main()


if __name__ == "__main__":
    main()

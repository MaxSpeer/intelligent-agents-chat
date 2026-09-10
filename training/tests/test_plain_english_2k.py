"""Data integrity, leakage and manual-start boundary for the 2k expansion."""

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

TRAINING = Path(__file__).resolve().parents[1]
DATA = TRAINING / "datasets/plain_english_2k"
spec = importlib.util.spec_from_file_location("plain2k_validation", DATA / "validate.py")
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


def test_expanded_data_integrity():
    report = validator.validate(DATA)
    assert report["split_counts"] == {"train": 2000, "validation": 100, "test": 100}
    assert report["source_answers_never_repeated"]
    assert report["legacy_80_train_dialogues_unchanged"]


def test_review_mapping_detects_rewritten_user_even_with_updated_hash(tmp_path):
    import hashlib

    shutil.copytree(DATA, tmp_path / "data")
    directory = tmp_path / "data"
    path = directory / "train.jsonl"
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    row = next(r for r in rows if r["id"].startswith("plain2k-"))
    row["messages"][1]["content"] += " Please explain clearly."
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    manifest = json.loads((directory / "manifest.json").read_text())
    manifest["file_sha256"]["train.jsonl"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (directory / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Unreviewed changes|Original user text"):
        validator.validate(directory)


def test_source_group_leakage_rejected(tmp_path):
    import hashlib

    shutil.copytree(DATA, tmp_path / "data")
    directory = tmp_path / "data"
    path = directory / "provenance.jsonl"
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    train = next(r for r in rows if r["split"] == "train")
    validation = next(r for r in rows if r["split"] == "validation")
    validation["split_group"] = train["split_group"]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    manifest = json.loads((directory / "manifest.json").read_text())
    manifest["file_sha256"]["provenance.jsonl"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (directory / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Cross-split group"):
        validator.validate(directory)


def test_training_requires_users_allocation_before_model_import():
    env = {k: v for k, v in os.environ.items() if k != "SLURM_JOB_ID"}
    process = subprocess.run(
        [sys.executable, str(TRAINING / "train_plain_english_2k.py"), "--run"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert process.returncode == 2
    assert "requires the user's existing Slurm GPU allocation" in process.stderr
    assert "Loading" not in process.stdout


def test_closing_filter_preserves_new_requests():
    spec = importlib.util.spec_from_file_location("plain2k_build", DATA / "build.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    assert builder.is_closing("Thanks for the explanation!", "You're welcome!")
    assert not builder.is_closing(
        "Thanks. Explain this with an example.", "You're welcome. Here is an example."
    )
    assert not builder.is_closing("That sounds useful. How does it work?", "Great, let me explain.")

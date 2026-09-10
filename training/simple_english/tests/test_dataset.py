"""Checks for the retained release without model weights or network access."""

import json
from pathlib import Path
import shutil

import pytest

from training.simple_english import evaluate, train, validate


def test_final_dataset_is_self_contained():
    report = validate.validate()
    assert report["counts"] == {
        "train": {"dialogues": 2000, "assistant_answers": 3085},
        "validation": {"dialogues": 100, "assistant_answers": 160},
        "test": {"dialogues": 100, "assistant_answers": 198},
    }
    assert report["tokenizer_validation"] == "not_run"


def test_changed_training_answer_is_rejected(tmp_path):
    directory = tmp_path / "data"
    shutil.copytree(validate.DATA, directory)
    path = directory / "train.jsonl"
    path.write_text(path.read_text().replace('"content":', '"content" :', 1))
    with pytest.raises(ValueError, match="File hash mismatch: train.jsonl"):
        validate.validate(directory)


def test_trainer_and_evaluator_use_final_release():
    selection = json.loads((Path(train.__file__).parent / "selection.json").read_text())
    assert train.DATA == validate.DATA == evaluate.DATA
    assert selection["selected_checkpoint"] == 193
    assert str(evaluate.DEFAULT_ADAPTER).endswith(selection["adapter_path"])
    assert train.SETTINGS == selection["training"]["settings"]
    assert train.OUTPUT != evaluate.DEFAULT_ADAPTER
    assert all((train.HERE / name).is_file() for name in train.CODE_FILES)


def test_train_requires_explicit_gpu_allocation(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setattr("sys.argv", ["train", "--run"])
    with pytest.raises(SystemExit):
        train.main()

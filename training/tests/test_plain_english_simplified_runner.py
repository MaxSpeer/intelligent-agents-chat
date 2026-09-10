"""The revised targets must be selected without changing the older runner."""

import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train_plain_english_simplified_v1 as revised  # noqa: E402


def test_revised_dataset_and_fresh_output_are_selected(monkeypatch):
    original = revised.runner.DATA, revised.runner.OUTPUT, revised.runner.CODE_FILES
    seen = {}

    def capture():
        seen.update(
            data=revised.runner.DATA,
            output=revised.runner.OUTPUT,
            code=revised.runner.CODE_FILES,
        )

    monkeypatch.setattr(revised.runner, "main", capture)
    revised.main()
    assert seen["data"].name == "plain_english_2k_simplified_v1"
    assert seen["output"].name == "qwen3-8b-plain-english-2k-simplified-v1"
    assert seen["data"] != original[0] and seen["output"] != original[1]
    assert set(revised.EXTRA_CODE_FILES).issubset(seen["code"])
    assert (revised.runner.DATA, revised.runner.OUTPUT, revised.runner.CODE_FILES) == original


def test_no_run_without_user_allocation():
    env = {key: value for key, value in os.environ.items() if key != "SLURM_JOB_ID"}
    result = subprocess.run(
        [sys.executable, revised.__file__, "--run"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "requires the user's existing Slurm GPU allocation" in result.stderr
    assert "Loading" not in result.stdout

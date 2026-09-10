"""CPU-only check of 2k evaluation routing and the explicit GPU-run boundary."""

import builtins
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


TRAINING = Path(__file__).resolve().parents[1]


def test_routes_fresh_test_and_adapter_without_loading_models(tmp_path, monkeypatch, capsys):
    original_import = builtins.__import__

    def no_model_import(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "transformers", "peft", "bitsandbytes"}:
            raise AssertionError(f"CPU preflight tried to import model dependency: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_model_import)
    monkeypatch.syspath_prepend(str(TRAINING))
    spec = importlib.util.spec_from_file_location(
        "evaluate_plain_english_2k_test_module", TRAINING / "evaluate_plain_english_2k.py"
    )
    wrapper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wrapper)
    assert wrapper.DATA == TRAINING / "datasets" / "plain_english_2k"
    assert wrapper.DEFAULT_ADAPTER == (
        wrapper.evaluator.PROJECT / "adapters" / "qwen3-8b-plain-english-2k"
    )

    data = tmp_path / "fresh-test"
    data.mkdir()
    system = {"role": "system", "content": "You are a helpful assistant."}
    rows = [
        {
            "id": f"fresh-{number}",
            "messages": [
                system,
                {"role": "user", "content": f"Explain example {number}."},
                {"role": "assistant", "content": "A reference answer for a file check."},
            ],
        }
        for number in range(100)
    ]
    test_file = data / "test.jsonl"
    test_file.write_text("".join(json.dumps(row) + "\n" for row in rows))
    (data / "manifest.json").write_text(json.dumps({
        "base_model": "Qwen/Qwen3-8B",
        "system_message": system,
        "split_counts": {"test": 100},
        "file_sha256": {"test.jsonl": hashlib.sha256(test_file.read_bytes()).hexdigest()},
    }))

    def make_adapter(name):
        adapter = tmp_path / name
        adapter.mkdir()
        (adapter / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": "Qwen/Qwen3-8B"})
        )
        # The shared preflight only checks that this file exists and is nonempty.
        (adapter / "adapter_model.safetensors").write_bytes(b"preflight fixture, not model weights")
        return adapter

    default_adapter = make_adapter("new-default-adapter")
    selected_adapter = make_adapter("selected-checkpoint")
    old_data = wrapper.evaluator.DATA
    old_adapter = wrapper.evaluator.DEFAULT_ADAPTER
    monkeypatch.setattr(wrapper, "DATA", data)
    monkeypatch.setattr(wrapper, "DEFAULT_ADAPTER", default_adapter)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    results = tmp_path / "results"

    for adapter_argument, expected_adapter in [([], default_adapter),
                                               (["--adapter", str(selected_adapter)], selected_adapter)]:
        monkeypatch.setattr(sys, "argv", ["evaluate_plain_english_2k.py", *adapter_argument])
        wrapper.main()
        output = capsys.readouterr().out
        assert "Test dialogues: 100; assistant answers per model: 100" in output
        assert f"Adapter: {expected_adapter}" in output
        assert "No model loaded" in output

    monkeypatch.setattr(sys, "argv", [
        "evaluate_plain_english_2k.py", "--run", "--adapter", str(selected_adapter),
        "--results-dir", str(results),
    ])
    with pytest.raises(SystemExit, match="existing Slurm GPU allocation"):
        wrapper.main()
    assert not results.exists()
    assert wrapper.evaluator.DATA == old_data
    assert wrapper.evaluator.DEFAULT_ADAPTER == old_adapter

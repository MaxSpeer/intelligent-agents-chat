"""Exercise the tunnel helper with fake SSH; never contact or launch cluster jobs."""

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "cluster" / "tunnel.sh"
ENDPOINT = "job=12345\nnode=gx27.hpc.sci.hpi.de\nhost=127.0.0.1\nport=8001\nmodel=qwen3-8b\n"


@pytest.fixture
def run_tunnel(tmp_path):
    calls_file = tmp_path / "ssh-calls.jsonl"
    fake_ssh = tmp_path / "ssh"
    fake_ssh.write_text(
        "#!/usr/bin/env python3\n"
        + textwrap.dedent(
            """
            import json, os, sys
            args = sys.argv[1:]
            with open(os.environ['SSH_CALLS'], 'a') as output:
                output.write(json.dumps(args) + '\\n')
            if args[-1].startswith('cat '):
                print(os.environ['ENDPOINT'], end='')
                sys.exit(int(os.environ.get('ENDPOINT_EXIT', '0')))
            if args[-1].startswith('squeue '):
                print(os.environ['JOB_STATE'])
                sys.exit(int(os.environ.get('SQUEUE_EXIT', '0')))
            if '-N' in args:
                sys.exit(0)
            sys.exit('Unexpected SSH call')
            """
        )
    )
    fake_ssh.chmod(0o755)

    def run(*extra_args, **overrides):
        env = {
            **os.environ,
            "PATH": os.pathsep.join((str(tmp_path), str(Path(sys.executable).parent), os.environ["PATH"])),
            "SSH_CALLS": str(calls_file),
            "ENDPOINT": ENDPOINT,
            "JOB_STATE": "RUNNING",
            **overrides,
        }
        result = subprocess.run(
            ["bash", str(SCRIPT), "qwen3-8b", "maximilian.speer", *extra_args],
            env=env,
            text=True,
            capture_output=True,
            timeout=10,
        )
        calls = (
            [json.loads(line) for line in calls_file.read_text().splitlines()]
            if calls_file.exists()
            else []
        )
        return result, calls

    return run


@pytest.mark.parametrize("name", [None, "plain-english", "vllm-qwen3-8b-12345"])
def test_batch_interactive_and_reconnected_endpoints(run_tunnel, name):
    result, calls = run_tunnel(*([name] if name else []))
    assert result.returncode == 0, result.stderr
    assert len(calls) == 3
    assert calls[0][-1].endswith(f"/{name or 'vllm-qwen3-8b'}.endpoint'")
    assert "--account=sci-lippert-intelligent-agents" in calls[1][-1]
    assert "--jobs=12345" in calls[1][-1]
    assert calls[2][-1] == "maximilian.speer@gx27.hpc.sci.hpi.de"
    assert calls[2][calls[2].index("-L") + 1] == "127.0.0.1:8001:127.0.0.1:8001"


@pytest.mark.parametrize("state", ["", "PENDING", "COMPLETING"])
def test_stale_or_inactive_job_never_opens_tunnel(run_tunnel, state):
    result, calls = run_tunnel(JOB_STATE=state)
    assert result.returncode != 0
    assert "not running" in result.stderr
    assert len(calls) == 2


def test_failed_job_check_never_opens_tunnel(run_tunnel):
    result, calls = run_tunnel(SQUEUE_EXIT="1")
    assert result.returncode != 0
    assert "Could not check Slurm" in result.stderr
    assert len(calls) == 2


@pytest.mark.parametrize(
    "endpoint",
    [
        ENDPOINT.replace("8001", "99999"),
        ENDPOINT.replace("qwen3-8b", "qwen3.5-9b"),
        ENDPOINT.replace("job=12345\n", ""),
    ],
)
def test_invalid_endpoint_never_opens_tunnel(run_tunnel, endpoint):
    result, calls = run_tunnel(ENDPOINT=endpoint)
    assert result.returncode != 0
    assert len(calls) == 1


def test_missing_endpoint_never_opens_tunnel(run_tunnel):
    result, calls = run_tunnel(ENDPOINT_EXIT="1")
    assert result.returncode != 0
    assert "Could not read the endpoint" in result.stderr
    assert len(calls) == 1


def test_invalid_endpoint_name_is_rejected_before_ssh(run_tunnel):
    result, calls = run_tunnel("../../some-file")
    assert result.returncode != 0
    assert not calls

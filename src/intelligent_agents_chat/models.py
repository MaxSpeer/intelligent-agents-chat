"""Hardcoded model profiles for the two vLLM jobs in ./cluster."""

from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True, slots=True)
class ModelProfile:
    """A selectable vLLM-served model."""

    key: str
    label: str
    base_url: str
    model: str
    supports_thinking: bool = False
    supports_tools: bool = False


def _port(env_var: str, default: int) -> int:
    value = os.environ.get(env_var)
    if not value:
        return default
    try:
        return int(value)
    except ValueError as error:
        raise ValueError(f"{env_var} must be an integer port") from error


# Local ports for the SSH tunnels opened by cluster/tunnel.sh (local port ==
# remote port).
QWEN3_8B_PORT = _port("VLLM_QWEN3_8B_PORT", 8001)
QWEN35_9B_PORT = _port("VLLM_QWEN35_9B_PORT", 8002)

_QWEN3_8B_BASE_URL = f"http://127.0.0.1:{QWEN3_8B_PORT}/v1"
_QWEN35_9B_BASE_URL = f"http://127.0.0.1:{QWEN35_9B_PORT}/v1"

DEFAULT_PROFILE_KEY = "qwen3.5-9b"

# qwen3-8b and conspiracy are served by the same vLLM process
# (cluster/run-vllm-qwen3-8b.sbatch's LORA_MODULES); qwen3.5-9b runs on a
# second, independent vLLM process (cluster/run-vllm-qwen35-9b.sbatch).
MODEL_PROFILES: tuple[ModelProfile, ...] = (
    ModelProfile(
        key="qwen3.5-9b",
        label="Qwen3.5 9B",
        base_url=_QWEN35_9B_BASE_URL,
        model="qwen3.5-9b",
        supports_thinking=True,
        supports_tools=True,
    ),
    ModelProfile(
        key="qwen3-8b",
        label="Qwen3 8B",
        base_url=_QWEN3_8B_BASE_URL,
        model="qwen3-8b",
        supports_thinking=False,
    ),
    ModelProfile(
        key="conspiracy",
        label="Qwen3 8B (conspiracy)",
        base_url=_QWEN3_8B_BASE_URL,
        model="conspiracy",
        supports_thinking=False,
    ),
)

_PROFILES_BY_KEY = {profile.key: profile for profile in MODEL_PROFILES}


def profile_options() -> dict[str, str]:
    """Return model profile keys and display names for the UI selector."""
    return {profile.key: profile.label for profile in MODEL_PROFILES}


def get_profile(key: str) -> ModelProfile:
    """Return one configured profile or fail rather than silently using another model."""
    try:
        return _PROFILES_BY_KEY[key]
    except KeyError:
        raise KeyError(f"Unknown model profile: {key}") from None

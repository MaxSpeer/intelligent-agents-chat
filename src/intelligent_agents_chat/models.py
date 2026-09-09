"""Configured profiles for OpenAI-compatible local and cluster model servers."""

from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True, slots=True)
class ModelProfile:
    """A selectable model served through an OpenAI-compatible endpoint."""

    key: str
    label: str
    base_url: str
    model: str
    supports_thinking: bool = False
    reasoning_effort: str | None = None
    context_window_tokens: int = 32_768
    supports_tools: bool = False


def _port(env_var: str, default: int) -> int:
    value = os.environ.get(env_var)
    if not value:
        return default
    try:
        return int(value)
    except ValueError as error:
        raise ValueError(f"{env_var} must be an integer port") from error


# SSH tunnel ports match the remote model-server ports.
QWEN3_8B_PORT = _port("VLLM_QWEN3_8B_PORT", 8001)
QWEN35_9B_PORT = _port("VLLM_QWEN35_9B_PORT", 8002)

_QWEN3_8B_BASE_URL = f"http://127.0.0.1:{QWEN3_8B_PORT}/v1"
_QWEN35_9B_BASE_URL = f"http://127.0.0.1:{QWEN35_9B_PORT}/v1"
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1").rstrip("/")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3.5:2b")

DEFAULT_PROFILE_KEY = "qwen3.5-9b"

# qwen3-8b and conspiracy share a vLLM process; qwen3.5-9b uses a separate one.
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
    ModelProfile(
        key="ollama-local",
        label=f"{OLLAMA_MODEL} (local Ollama)",
        base_url=OLLAMA_BASE_URL,
        model=OLLAMA_MODEL,
        # Disable default reasoning so short-response budgets remain available for answers.
        reasoning_effort="none",
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

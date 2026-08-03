"""Runtime configuration for the chat application and model profiles."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Literal


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ModelBackend = Literal["lorem", "vllm"]


@dataclass(frozen=True, slots=True)
class ModelProfile:
    """A selectable model backend."""

    key: str
    label: str
    backend: ModelBackend
    base_url: str | None = None
    model: str | None = None


LOREM_PROFILE = ModelProfile(
    key="lorem",
    label="Lorem Ipsum (offline)",
    backend="lorem",
)


@dataclass(frozen=True, slots=True)
class Settings:
    """Validated application settings loaded from environment variables."""

    database_path: Path
    model_profiles: tuple[ModelProfile, ...]
    default_profile_key: str
    api_key: str
    request_timeout_seconds: float
    max_tokens: int
    temperature: float
    system_prompt: str

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Settings:
        """Build settings without requiring a dotenv dependency."""
        values = os.environ if environ is None else environ

        database_value = values.get("CHAT_DB_PATH")
        database_path = (
            Path(database_value).expanduser()
            if database_value
            else PROJECT_ROOT / ".data" / "chats.sqlite3"
        )

        profiles = _load_profiles(values)
        default_profile_key = values.get(
            "CHAT_DEFAULT_PROFILE",
            values.get("VLLM_DEFAULT_PROFILE", LOREM_PROFILE.key),
        ).strip()
        profile_keys = {profile.key for profile in profiles}
        if default_profile_key not in profile_keys:
            raise ValueError(
                "CHAT_DEFAULT_PROFILE (or legacy VLLM_DEFAULT_PROFILE) must match "
                "a configured profile key"
            )

        timeout = _positive_float(values.get("VLLM_TIMEOUT_SECONDS", "120"), "VLLM_TIMEOUT_SECONDS")
        max_tokens = _positive_int(values.get("VLLM_MAX_TOKENS", "1024"), "VLLM_MAX_TOKENS")
        temperature = _bounded_float(
            values.get("VLLM_TEMPERATURE", "0.2"),
            "VLLM_TEMPERATURE",
            minimum=0.0,
            maximum=2.0,
        )

        return cls(
            database_path=database_path,
            model_profiles=profiles,
            default_profile_key=default_profile_key,
            api_key=values.get("VLLM_API_KEY", "not-needed"),
            request_timeout_seconds=timeout,
            max_tokens=max_tokens,
            temperature=temperature,
            system_prompt=values.get(
                "VLLM_SYSTEM_PROMPT",
                "You are a helpful assistant. Give clear, accurate, and concise answers.",
            ).strip(),
        )

    @property
    def profile_options(self) -> dict[str, str]:
        """Return model profile keys and display names for the UI selector."""
        return {profile.key: profile.label for profile in self.model_profiles}

    def profile(self, key: str) -> ModelProfile:
        """Return one configured profile or fail rather than silently using another model."""
        for profile in self.model_profiles:
            if profile.key == key:
                return profile
        raise KeyError(f"Unknown model profile: {key}")


def _load_profiles(values: Mapping[str, str]) -> tuple[ModelProfile, ...]:
    raw_profiles = values.get("VLLM_PROFILES_JSON", "").strip()
    if not raw_profiles:
        model = values.get("VLLM_MODEL", "qwen3-0.6b").strip()
        profile = ModelProfile(
            key=values.get("VLLM_PROFILE_KEY", "default").strip(),
            label=values.get("VLLM_PROFILE_LABEL", model).strip(),
            backend="vllm",
            base_url=values.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/"),
            model=model,
        )
        _validate_vllm_profile(profile, 0)
        profiles = (LOREM_PROFILE, profile)
        _validate_unique_keys(profiles)
        return profiles

    try:
        decoded = json.loads(raw_profiles)
    except json.JSONDecodeError as error:
        raise ValueError("VLLM_PROFILES_JSON must contain valid JSON") from error

    if not isinstance(decoded, list) or not decoded:
        raise ValueError("VLLM_PROFILES_JSON must be a non-empty JSON list")

    profiles: list[ModelProfile] = []
    for index, item in enumerate(decoded):
        if not isinstance(item, dict):
            raise ValueError(f"VLLM_PROFILES_JSON item {index} must be an object")
        try:
            profile = ModelProfile(
                key=str(item["key"]).strip(),
                label=str(item["label"]).strip(),
                backend="vllm",
                base_url=str(item["base_url"]).rstrip("/"),
                model=str(item["model"]).strip(),
            )
        except KeyError as error:
            raise ValueError(
                f"VLLM_PROFILES_JSON item {index} is missing {error.args[0]!r}"
            ) from error
        _validate_vllm_profile(profile, index)
        profiles.append(profile)

    configured_profiles = (LOREM_PROFILE, *profiles)
    _validate_unique_keys(configured_profiles)
    return configured_profiles


def _validate_unique_keys(profiles: tuple[ModelProfile, ...]) -> None:
    keys = [profile.key for profile in profiles]
    if len(keys) != len(set(keys)):
        raise ValueError("Model profile keys must be unique")


def _validate_vllm_profile(profile: ModelProfile, index: int) -> None:
    for field_name in ("key", "label", "base_url", "model"):
        if not getattr(profile, field_name):
            raise ValueError(f"VLLM profile {index} has an empty {field_name}")
    assert profile.base_url is not None
    if not profile.base_url.startswith(("http://", "https://")):
        raise ValueError(f"VLLM profile {index} base_url must start with http:// or https://")


def _positive_float(value: str, name: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a number") from error
    if parsed <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return parsed


def _positive_int(value: str, name: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer") from error
    if parsed <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return parsed


def _bounded_float(value: str, name: str, *, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a number") from error
    if not minimum <= parsed <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return parsed

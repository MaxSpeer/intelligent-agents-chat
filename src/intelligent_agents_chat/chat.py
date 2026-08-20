"""Chat backend: model/repository bootstrap, request assembly, and reply streaming.

This is the seam where tool-calling, RAG, and context compression will be added later
(as a loop around `stream_reply` instead of a single `gateway.stream_reply` call) --
app.py should not need to change when that happens.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from importlib.metadata import version
import logging
import platform

from intelligent_agents_chat.database import ChatRepository, Message
from intelligent_agents_chat.llm import (
    MAX_TOKENS,
    SYSTEM_PROMPT,
    THINKING_MAX_TOKENS,
    VLLMGateway,
)
from intelligent_agents_chat.logging_config import configure_logging, log_event, sanitized_endpoint
from intelligent_agents_chat.models import DEFAULT_PROFILE_KEY, MODEL_PROFILES, ModelProfile

configure_logging()
logger = logging.getLogger(__name__)

log_event(
    logger,
    logging.INFO,
    "application.runtime.loaded",
    python_version=platform.python_version(),
    operating_system=platform.platform(),
    dependency_versions={
        "nicegui": version("nicegui"),
        "openai": version("openai"),
    },
)
log_event(
    logger,
    logging.INFO,
    "application.configuration.loaded",
    default_profile_key=DEFAULT_PROFILE_KEY,
    profile_count=len(MODEL_PROFILES),
    profiles=[
        {
            "key": profile.key,
            "model": profile.model,
            "endpoint": sanitized_endpoint(profile.base_url),
            "supports_thinking": profile.supports_thinking,
        }
        for profile in MODEL_PROFILES
    ],
    max_tokens=MAX_TOKENS,
    thinking_max_tokens=THINKING_MAX_TOKENS,
    system_prompt_chars=len(SYSTEM_PROMPT),
)

repository = ChatRepository()
try:
    repository.initialize()
except Exception:
    logger.exception(
        "application.database_initialization_failed",
        extra={
            "event": "application.database_initialization_failed",
            "database_path": str(repository.database_path),
        },
    )
    raise

gateway = VLLMGateway()
active_generations: set[str] = set()

log_event(logger, logging.INFO, "application.initialized")


# Hardcoded for now, to observe how the model reacts to a tool being offered --
# not executed yet.
CALCULATOR_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Evaluate a basic arithmetic expression and return the numeric result.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": 'The arithmetic expression to evaluate, e.g. "12 * (3 + 4)".',
                },
            },
            "required": ["expression"],
        },
    },
}


def completion_messages(messages: list[Message]) -> list[dict[str, str]]:
    """Build the OpenAI-style message list for a completion request."""
    result: list[dict[str, str]] = []
    if SYSTEM_PROMPT:
        result.append({"role": "system", "content": SYSTEM_PROMPT})
    result.extend({"role": message.role, "content": message.content} for message in messages)
    return result


def stream_reply(
    profile: ModelProfile,
    messages: list[dict[str, str]],
    *,
    request_id: str | None = None,
    thinking_enabled: bool = False,
) -> AsyncIterator[str]:
    """Stream the assistant's reply for an already-assembled message list."""
    return gateway.stream_reply(
        profile,
        messages,
        request_id=request_id,
        thinking_enabled=thinking_enabled,
        tools=[CALCULATOR_TOOL] if profile.supports_tools else None,
    )

"""Chat backend: model/repository bootstrap, request assembly, and the agent loop.

RAG and context compression will be added later as further steps in `stream_reply`'s
loop -- app.py should not need to change when that happens.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from importlib.metadata import version
import json
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
from intelligent_agents_chat.tools import calculator

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


# Only one tool for now; a name-keyed registry can replace this once more arrive.
TOOLS: dict = {"calculator": calculator.CALCULATOR_TOOL}

MAX_TOOL_ROUNDS = 4
# A single model turn can request an unbounded number of tool calls (limited only
# by max_tokens) before any result is fed back -- cap it so one turn can't burn
# the whole token budget on speculative calls.
MAX_TOOL_CALLS_PER_ROUND = 5


def _execute_tool(name: str, arguments: dict) -> str:
    if name == "calculator":
        return calculator.run(arguments)
    return f"Error: unknown tool '{name}'"


def completion_messages(messages: list[Message]) -> list[dict[str, str]]:
    """Build the OpenAI-style message list for a completion request."""
    result: list[dict[str, str]] = []
    if SYSTEM_PROMPT:
        result.append({"role": "system", "content": SYSTEM_PROMPT})
    result.extend({"role": message.role, "content": message.content} for message in messages)
    return result


async def stream_reply(
    profile: ModelProfile,
    messages: list[dict[str, str]],
    *,
    request_id: str | None = None,
    thinking_enabled: bool = False,
) -> AsyncIterator[str]:
    """Stream the assistant's reply, executing any tool calls the model makes."""
    tools = list(TOOLS.values()) if profile.supports_tools else None
    conversation_messages: list[dict] = list(messages)

    for _ in range(MAX_TOOL_ROUNDS):
        pending_tool_calls: list[dict] = []
        async for chunk in gateway.stream_reply(
            profile,
            conversation_messages,
            request_id=request_id,
            thinking_enabled=thinking_enabled,
            tools=tools,
            tool_calls=pending_tool_calls,
        ):
            yield chunk

        if not pending_tool_calls:
            return

        conversation_messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call["id"],
                        "type": "function",
                        "function": {"name": call["name"], "arguments": call["arguments"]},
                    }
                    for call in pending_tool_calls
                ],
            }
        )
        if len(pending_tool_calls) > MAX_TOOL_CALLS_PER_ROUND:
            log_event(
                logger,
                logging.WARNING,
                "chat.tool_calls.round_limit_exceeded",
                request_id=request_id,
                profile_key=profile.key,
                tool_call_count=len(pending_tool_calls),
                max_tool_calls_per_round=MAX_TOOL_CALLS_PER_ROUND,
            )
        for index, call in enumerate(pending_tool_calls):
            if index < MAX_TOOL_CALLS_PER_ROUND:
                try:
                    arguments = json.loads(call["arguments"]) if call["arguments"] else {}
                except json.JSONDecodeError:
                    arguments = {}
                result = _execute_tool(call["name"], arguments)
            else:
                result = (
                    "Error: too many tool calls in a single turn "
                    f"(max {MAX_TOOL_CALLS_PER_ROUND}); this call was skipped."
                )
            yield f"\n[tool_result: {call['name']}] {result}\n\n"
            conversation_messages.append(
                {"role": "tool", "tool_call_id": call["id"], "content": result}
            )

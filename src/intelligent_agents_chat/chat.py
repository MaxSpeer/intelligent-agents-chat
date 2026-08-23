"""Chat backend: model/repository bootstrap, request assembly, and the agent loop.

`stream_reply` yields structured `StreamEvent`s (text, tool calls, tool results) so
callers can both render and persist each one correctly -- see app.py's send_message
for how they get turned into chat_message widgets and `messages` table rows. RAG and
context compression will be added later as further steps in this same loop.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
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
    check_model_available,
)
from intelligent_agents_chat.logging_config import configure_logging, log_event, sanitized_endpoint
from intelligent_agents_chat.models import DEFAULT_PROFILE_KEY, MODEL_PROFILES, ModelProfile
from intelligent_agents_chat.tools import Tool, calculator, subagent, webfetch

PROFILE_STATUS_POLL_INTERVAL_SECONDS = 15.0

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
profile_status: dict[str, bool | None] = {profile.key: None for profile in MODEL_PROFILES}

log_event(logger, logging.INFO, "application.initialized")


# Every available tool, self-registered by its module. Add a new tool by
# writing a `tools/<name>.py` that exports a `Tool` (see tools/calculator.py),
# then listing it here -- nothing else in this file needs to change.
_ALL_TOOLS: tuple[Tool, ...] = (calculator.TOOL, subagent.TOOL, webfetch.TOOL)
TOOLS: dict[str, Tool] = {tool.name: tool for tool in _ALL_TOOLS}

MAX_TOOL_ROUNDS = 4
MAX_TOOL_CALLS_PER_ROUND = 5


async def _execute_tool(name: str, arguments: dict) -> str:
    """Await one tool's `run` to completion before returning.

    Tools run asynchronously so none of them block
    the event loop that serves every other connected browser tab. This is
    still a sequential await, though - the calling agent loop (below)
    stops and waits for this to finish before it does anything else; it does
    not continue on in parallel while a tool (e.g. a sub-agent) is running.
    """
    tool = TOOLS.get(name)
    if tool is None:
        return f"Error: unknown tool '{name}'"
    try:
        return await tool.run(arguments)
    except Exception as error:
        # A tool is expected to catch its own errors and return them as text
        # This is a backstop so a badly-written tool
        # can't take down the whole agent loop.
        logger.exception(
            "chat.tool.execution_failed",
            extra={"event": "chat.tool.execution_failed", "tool_name": name},
        )
        return f"Error: tool '{name}' failed unexpectedly: {error}"


@dataclass(frozen=True, slots=True)
class TextChunk:
    """A fragment of visible model text.

    `is_reasoning` marks text that should be shown but never persisted as the
    message's real content -- reasoning isn't part of the model's actual answer
    and shouldn't be replayed back as conversation history.
    """

    text: str
    is_reasoning: bool = False


@dataclass(frozen=True, slots=True)
class ToolCallEvent:
    """The model requested these tool calls; persist as one assistant message."""

    tool_calls: tuple[dict, ...]


@dataclass(frozen=True, slots=True)
class ToolResultEvent:
    """The result of executing one tool call; persist as one "tool" message."""

    tool_call_id: str
    name: str
    result: str


StreamEvent = TextChunk | ToolCallEvent | ToolResultEvent


def format_reasoning_entry(text: str) -> str:
    """Markdown for one reasoning block, as a collapsible accordion entry.

    Used both live (while `text` is still streaming) and when replaying a
    finished conversation from the DB -- keeping the formatting in one place
    is the whole point, so the two views always look identical.
    """
    return f"**Thinking**\n\n{text}"


def format_tool_call_entry(call: dict) -> str:
    """Markdown for one tool call, as a collapsible accordion entry."""
    return f"🔧 **{call['name']}**\n\n```\n{call['arguments']}\n```"


def format_tool_result_entry(name: str, result: str) -> str:
    """Markdown for one tool result, as a collapsible accordion entry."""
    return f"**{name}** → {result}"


def completion_messages(messages: list[Message]) -> list[dict[str, str]]:
    """Build the OpenAI-style message list for a completion request."""
    result: list[dict[str, str]] = []
    if SYSTEM_PROMPT:
        result.append({"role": "system", "content": SYSTEM_PROMPT})
    for message in messages:
        if message.role == "assistant" and message.tool_calls:
            result.append(
                {
                    "role": "assistant",
                    "content": message.content or None,
                    "tool_calls": [
                        {
                            "id": call["id"],
                            "type": "function",
                            "function": {"name": call["name"], "arguments": call["arguments"]},
                        }
                        for call in message.tool_calls
                    ],
                }
            )
        elif message.role == "tool":
            result.append(
                {
                    "role": "tool",
                    "tool_call_id": message.tool_call_id,
                    "content": message.content,
                }
            )
        else:
            result.append({"role": message.role, "content": message.content})
    return result


async def stream_reply(
    profile: ModelProfile,
    messages: list[dict],
    *,
    request_id: str | None = None,
    thinking_enabled: bool = False,
) -> AsyncIterator[StreamEvent]:
    """Stream the assistant's reply, executing any tool calls the model makes.

    Yields `TextChunk` for visible model text, `ToolCallEvent` once per round when
    the model requests tool calls, and `ToolResultEvent` once per executed call --
    the caller is expected to both render and persist each event appropriately.
    """
    tools = [tool.schema for tool in TOOLS.values()] if profile.supports_tools else None
    conversation_messages: list[dict] = list(messages)

    for _ in range(MAX_TOOL_ROUNDS):
        pending_tool_calls: list[dict] = []
        saw_content = False
        reasoning_chunks: list[str] = []
        async for delta in gateway.stream_reply(
            profile,
            conversation_messages,
            request_id=request_id,
            thinking_enabled=thinking_enabled,
            tools=tools,
            tool_calls=pending_tool_calls,
        ):
            if delta.is_reasoning:
                reasoning_chunks.append(delta.text)
            else:
                saw_content = True
            yield TextChunk(delta.text, is_reasoning=delta.is_reasoning)

        if not pending_tool_calls:
            if not saw_content and reasoning_chunks:
                # Bug that sometimes happened: The model finished last round
                # (no more tool calls to make) without ever producing real `content`
                # instead only producing reasoning text.
                # Rather than show an empty answer, treat that last
                # reasoning text as the real answer: still shown in the
                # collapsible trace as it streamed, but now also persisted
                # and displayed as the actual response.
                fallback = "".join(reasoning_chunks).strip()
                if fallback:
                    log_event(
                        logger,
                        logging.WARNING,
                        "chat.reasoning_used_as_fallback_answer",
                        request_id=request_id,
                        profile_key=profile.key,
                        reasoning_chars=len(fallback),
                    )
                    yield TextChunk(fallback, is_reasoning=False)
            return

        yield ToolCallEvent(tuple(pending_tool_calls))
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
                # Sequential on purpose: each tool call is awaited to completion
                # before the next one starts (and before the model gets to see any
                # results), even though tools themselves run async.
                result = await _execute_tool(call["name"], arguments)
            else:
                result = (
                    "Error: too many tool calls in a single turn "
                    f"(max {MAX_TOOL_CALLS_PER_ROUND}); this call was skipped."
                )
            yield ToolResultEvent(tool_call_id=call["id"], name=call["name"], result=result)
            conversation_messages.append(
                {"role": "tool", "tool_call_id": call["id"], "content": result}
            )


async def refresh_profile_status() -> None:
    """Check every profile's vLLM endpoint concurrently and update `profile_status` in place."""
    results = await asyncio.gather(
        *(check_model_available(profile) for profile in MODEL_PROFILES),
        return_exceptions=True,
    )
    for profile, result in zip(MODEL_PROFILES, results, strict=True):
        if isinstance(result, BaseException):
            logger.exception(
                "chat.profile_status.check_failed",
                exc_info=result,
                extra={"event": "chat.profile_status.check_failed", "model_profile": profile.key},
            )
            is_available = False
        else:
            is_available = result
        previous = profile_status[profile.key]
        profile_status[profile.key] = is_available
        if previous is not None and previous != is_available:
            log_event(
                logger,
                logging.INFO if is_available else logging.WARNING,
                "chat.profile_status.changed",
                model_profile=profile.key,
                available=is_available,
            )


async def poll_profile_status() -> None:
    """Continuously refresh `profile_status` in the background."""
    while True:
        await refresh_profile_status()
        await asyncio.sleep(PROFILE_STATUS_POLL_INTERVAL_SECONDS)

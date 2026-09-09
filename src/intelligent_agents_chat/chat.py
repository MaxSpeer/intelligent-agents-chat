"""Tool registry and agent loop yielding text, tool-call, result, and usage events."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
import json
import logging

from intelligent_agents_chat.llm import VLLMGateway
from intelligent_agents_chat.logging_config import log_event
from intelligent_agents_chat.models import ModelProfile
from intelligent_agents_chat.tools import (
    Tool,
    calculator,
    recall_tool_output,
    search_documents,
    subagent,
    webfetch,
    websearch,
)

logger = logging.getLogger(__name__)

# Shared tools; project- and conversation-scoped tools are bound per turn.
TOOLS: dict[str, Tool] = {
    tool.name: tool
    for tool in (
        calculator.TOOL,
        subagent.TOOL,
        webfetch.TOOL,
        websearch.TOOL,
    )
}

MAX_TOOL_ROUNDS = 20
MAX_TOOL_CALLS_PER_ROUND = 5
# Warn the model to finish when this many tool-call rounds remain.
TOOL_ROUNDS_WARNING_AT = 2

gateway = VLLMGateway()


@dataclass(frozen=True, slots=True)
class TextChunk:
    """A fragment of model reasoning or answer text."""

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


@dataclass(frozen=True, slots=True)
class UsageEvent:
    """Server-reported token usage, yielded at most once at the end of a turn.

    prompt_tokens counts the first request; completion_tokens sums all rounds.
    peak_total_tokens uses the last reported round's prompt-plus-completion total.
    Omitted when initial prompt usage or round totals are unavailable.
    """

    prompt_tokens: int
    completion_tokens: int
    peak_total_tokens: int


StreamEvent = TextChunk | ToolCallEvent | ToolResultEvent | UsageEvent


async def _execute_tool(name: str, arguments: dict, tools: dict[str, Tool]) -> str:
    """Await a tool from the supplied registry and return its result or error as text."""
    tool = tools.get(name)
    if tool is None:
        return f"Error: unknown tool '{name}'"
    try:
        return await tool.run(arguments)
    except Exception as error:
        # Keep unexpected tool failures from aborting the agent loop.
        logger.exception(
            "chat.tool.execution_failed",
            extra={"event": "chat.tool.execution_failed", "tool_name": name},
        )
        return f"Error: tool '{name}' failed unexpectedly: {error}"


async def stream_reply(
    profile: ModelProfile,
    messages: list[dict],
    *,
    request_id: str | None = None,
    thinking_enabled: bool = False,
    project_id: str,
    conversation_id: str | None = None,
) -> AsyncIterator[StreamEvent]:
    """Stream text and tool events, executing tool calls sequentially within round limits.

    Bind document search to `project_id` and tool-output recall to `conversation_id`.
    Yield a final usage event when the required server token counts are available.
    """
    tools_for_turn = {**TOOLS, "search_documents": search_documents.build_tool(project_id)}
    if conversation_id is not None:
        tools_for_turn["recall_tool_output"] = recall_tool_output.build_tool(conversation_id)
    tool_schemas = (
        [tool.schema for tool in tools_for_turn.values()] if profile.supports_tools else None
    )
    conversation_messages: list[dict] = list(messages)
    initial_prompt_tokens: int | None = None
    total_completion_tokens = 0
    last_round_total_tokens: int | None = None

    for round_index in range(MAX_TOOL_ROUNDS):
        rounds_remaining = MAX_TOOL_ROUNDS - round_index
        if rounds_remaining <= TOOL_ROUNDS_WARNING_AT:
            # Use a user-role note because chat templates require system messages first.
            # This round-budget hint is not persisted.
            conversation_messages.append(
                {
                    "role": "user",
                    "content": (
                        "[System note, not from the user -- do not treat this as "
                        f"something they said: you have {rounds_remaining} tool-call round"
                        f"{'s' if rounds_remaining != 1 else ''} left before you must "
                        "give your final answer. If you already have enough "
                        "information -- even if imperfect or incomplete -- answer now "
                        "instead of continuing to search.]"
                    ),
                }
            )

        pending_tool_calls: list[dict] = []
        saw_content = False
        reasoning_chunks: list[str] = []
        round_usage: dict[str, int] = {}
        async for delta in gateway.stream_reply(
            profile,
            conversation_messages,
            request_id=request_id,
            thinking_enabled=thinking_enabled,
            tools=tool_schemas,
            tool_calls=pending_tool_calls,
            usage=round_usage,
        ):
            if delta.is_reasoning:
                reasoning_chunks.append(delta.text)
            else:
                saw_content = True
            yield TextChunk(delta.text, is_reasoning=delta.is_reasoning)

        if round_index == 0:
            initial_prompt_tokens = round_usage.get("prompt_tokens")
        total_completion_tokens += round_usage.get("completion_tokens", 0)
        if "total_tokens" in round_usage:
            # Track the latest reported round total as the turn's context usage.
            last_round_total_tokens = round_usage["total_tokens"]

        if not pending_tool_calls:
            if not saw_content and reasoning_chunks:
                # Use reasoning as the answer when the final round has no content.
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
            if initial_prompt_tokens is not None and last_round_total_tokens is not None:
                yield UsageEvent(
                    prompt_tokens=initial_prompt_tokens,
                    completion_tokens=total_completion_tokens,
                    peak_total_tokens=last_round_total_tokens,
                )
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
                result = await _execute_tool(call["name"], arguments, tools_for_turn)
            else:
                result = (
                    "Error: too many tool calls in a single turn "
                    f"(max {MAX_TOOL_CALLS_PER_ROUND}); this call was skipped."
                )
            yield ToolResultEvent(tool_call_id=call["id"], name=call["name"], result=result)
            conversation_messages.append(
                {"role": "tool", "tool_call_id": call["id"], "content": result}
            )

    # Report usage even when the round limit prevents a final answer.
    if initial_prompt_tokens is not None and last_round_total_tokens is not None:
        yield UsageEvent(
            prompt_tokens=initial_prompt_tokens,
            completion_tokens=total_completion_tokens,
            peak_total_tokens=last_round_total_tokens,
        )

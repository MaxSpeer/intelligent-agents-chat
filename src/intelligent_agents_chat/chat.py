"""Chat backend: the tool registry and the agent loop.

`stream_reply` yields structured `StreamEvent`s (text, tool calls, tool results) so
callers can both render and persist each one correctly -- see app.py's send_message
for how they get turned into chat_message widgets and `messages` table rows. It
consumes whatever context.py's prepare_conversation_context already assembled; this
module doesn't need to import anything from there to do that.

Project-document retrieval (search_documents) is a tool like any other here,
not a separate pipeline -- the model decides for itself whether/when to call
it, inside the same round budget as every other tool. It's the one tool that
needs a project_id bound at request time rather than left to the model (see
tools/search_documents.py), which is why stream_reply takes project_id and
builds one extra, request-scoped tool alongside the static TOOLS registry.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
import json
import logging

from intelligent_agents_chat.llm import VLLMGateway, check_model_available
from intelligent_agents_chat.logging_config import log_event
from intelligent_agents_chat.models import MODEL_PROFILES, ModelProfile
from intelligent_agents_chat.tools import (
    Tool,
    calculator,
    search_documents,
    subagent,
    webfetch,
    websearch,
)

logger = logging.getLogger(__name__)

# Model Reachability Status
PROFILE_STATUS_POLL_INTERVAL_SECONDS = 15.0
profile_status: dict[str, bool | None] = {profile.key: None for profile in MODEL_PROFILES}

# Every tool available regardless of which conversation is asking, self-
# registered by its module. Add a new tool by writing a `tools/<name>.py`
# that exports a `Tool` (see tools/calculator.py), then listing it here --
# nothing else in this file needs to change. search_documents is deliberately
# NOT here: it needs a project_id bound per request (see stream_reply).
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
# Once this few tool call rounds are left warn the model to wrap up
TOOL_ROUNDS_WARNING_AT = 2

gateway = VLLMGateway()
# Tracked here (not in context.py) -- purely a UI concurrency guard ("this
# chat is already generating in another tab"), unrelated to context assembly.
active_generations: set[str] = set()


#
# Main Agent Loop: Generating - Toolcall - Tool Result - Generating
#


@dataclass(frozen=True, slots=True)
class TextChunk:
    """A fragment of visible model text. Can be reasoning or the part of the final answer"""

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
    """Real token usage for the whole turn, straight from the model server --
    not the chars/3 estimate context.py uses for budgeting. Yielded once, at
    most, right before the turn ends -- omitted entirely if the server never
    reported usage.

    prompt_tokens: from the turn's first (initial) request only -- the one
    directly comparable to ContextPlan.estimated_input_tokens. Every later
    round's prompt also includes everything from earlier rounds (their
    replies get appended before the next request), so "real input tokens for
    the whole turn" isn't a meaningful single number beyond round 0.

    completion_tokens: summed across every request the turn made (the
    initial one plus any tool-call rounds, each of which is its own separate
    completion with its own full output budget -- see stream_reply). This is
    "how much the model generated in total this turn", not a context-window
    quantity by itself.

    peak_total_tokens: the largest prompt+completion total any single request
    this turn reached -- the most the model ever had to hold in context at
    once. Since the conversation only ever grows across rounds, that's always
    the *last* round's total, never an earlier one. Compare against the
    profile's context_window_tokens to see how close a turn came to actually
    running out of context mid-turn (nothing currently stops that -- see
    ContextAssembler, which only ever budgets the turn's first request).
    """

    prompt_tokens: int
    completion_tokens: int
    peak_total_tokens: int


StreamEvent = TextChunk | ToolCallEvent | ToolResultEvent | UsageEvent


async def _execute_tool(name: str, arguments: dict, tools: dict[str, Tool]) -> str:
    """Await one tool's `run` to completion before returning.

    Tools run asynchronously so none of them block
    the event loop that serves every other connected browser tab. This is
    still a sequential await, though - the calling agent loop (below)
    stops and waits for this to finish before it does anything else; it does
    not continue on in parallel while a tool (e.g. a sub-agent) is running.

    `tools` is the round's actual tool set (TOOLS plus the request-scoped
    search_documents, see stream_reply) -- passed in rather than read off the
    module-level TOOLS directly, since that no longer has everything a given
    turn can call.
    """
    tool = tools.get(name)
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


async def stream_reply(
    profile: ModelProfile,
    messages: list[dict],
    *,
    request_id: str | None = None,
    thinking_enabled: bool = False,
    project_id: str,
) -> AsyncIterator[StreamEvent]:
    """Stream the assistant's reply, executing any tool calls the model makes.

    Yields `TextChunk` for visible model text including reasoning,
    `ToolCallEvent` once per round when the model requests tool calls,
    `ToolResultEvent` once per executed call, and a final `UsageEvent` (see its
    docstring) once the turn's real answer is ready.

    project_id scopes the one request-specific tool, search_documents (see
    its module docstring) -- built fresh here, alongside the static TOOLS,
    rather than the model ever supplying which project to search itself.
    """
    tools_for_turn = {**TOOLS, "search_documents": search_documents.build_tool(project_id)}
    tool_schemas = (
        [tool.schema for tool in tools_for_turn.values()] if profile.supports_tools else None
    )
    conversation_messages: list[dict] = list(messages)
    # See UsageEvent's docstring for what each of these means and why.
    initial_prompt_tokens: int | None = None
    total_completion_tokens = 0
    last_round_total_tokens: int | None = None

    for round_index in range(MAX_TOOL_ROUNDS):
        rounds_remaining = MAX_TOOL_ROUNDS - round_index
        if rounds_remaining <= TOOL_ROUNDS_WARNING_AT:
            # A runtime hint for the model to wrap up instead of continuing to
            # retry. Not persisted in the DB, just a nudge for this one turn.
            # role: "user", not "system" -- this is appended mid-conversation,
            # and at least one vLLM chat template in use here rejects any
            # system message that isn't the very first one in the request
            # ("System message must be at the beginning"). The bracketed
            # disclaimer keeps it from reading as something the user said
            # (see context.py's not_from_user_note, which this mirrors --
            # small enough not to be worth importing across an otherwise
            # decoupled pair of modules).
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
            # Overwritten every round on purpose -- conversation_messages only
            # ever grows, so the last round's total is always the peak (see
            # UsageEvent's docstring).
            last_round_total_tokens = round_usage["total_tokens"]

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

        # Now the actual tool execution
        for index, call in enumerate(pending_tool_calls):
            if index < MAX_TOOL_CALLS_PER_ROUND:
                try:
                    arguments = json.loads(call["arguments"]) if call["arguments"] else {}
                except json.JSONDecodeError:
                    arguments = {}
                # Each tool call is awaited to completion
                # before the next one starts (and before the model gets to see any
                # results), even though tools themselves run async.
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

    # MAX_TOOL_ROUNDS was exhausted without the model ever giving a final
    # answer (see the round-warning nudge above, which is meant to prevent
    # this) -- still report what was actually spent.
    if initial_prompt_tokens is not None and last_round_total_tokens is not None:
        yield UsageEvent(
            prompt_tokens=initial_prompt_tokens,
            completion_tokens=total_completion_tokens,
            peak_total_tokens=last_round_total_tokens,
        )


#
# Model Reachability Polling: Check every profile's vLLM endpoint concurrently
#


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

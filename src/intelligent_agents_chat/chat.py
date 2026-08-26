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
import json
import logging

from intelligent_agents_chat.context import ContextAssembler, ContextPlan
from intelligent_agents_chat.database import ChatRepository, Conversation, Message
from intelligent_agents_chat.llm import (
    MAX_TOKENS,
    THINKING_MAX_TOKENS,
    VLLMGateway,
    check_model_available,
    system_prompt_for_today,
)
from intelligent_agents_chat.logging_config import configure_logging, log_event
from intelligent_agents_chat.memory import ProjectMemoryStore
from intelligent_agents_chat.models import MODEL_PROFILES, ModelProfile
from intelligent_agents_chat.retrieval import ContextCandidate, RetrievalQuery
from intelligent_agents_chat.tools import Tool, calculator, subagent, webfetch, websearch

# Model Reachability Status
PROFILE_STATUS_POLL_INTERVAL_SECONDS = 15.0
profile_status: dict[str, bool | None] = {profile.key: None for profile in MODEL_PROFILES}

# Every available tool, self-registered by its module. Add a new tool by
# writing a `tools/<name>.py` that exports a `Tool` (see tools/calculator.py),
# then listing it here -- nothing else in this file needs to change.
TOOLS: dict[str, Tool] = {
    tool.name: tool
    for tool in (
        calculator.TOOL,
        subagent.TOOL,
        webfetch.TOOL,
        websearch.TOOL,
    )
}

MAX_TOOL_ROUNDS = 10
MAX_TOOL_CALLS_PER_ROUND = 5
# Once this few tool call rounds are left warn the model to wrap up
TOOL_ROUNDS_WARNING_AT = 2


configure_logging()
logger = logging.getLogger(__name__)

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
memory_store = ProjectMemoryStore(repository.database_path)
context_assembler = ContextAssembler(system_prompt_for_today)
active_generations: set[str] = set()

try:
    rebuilt_entry_count = sum(
        memory_store.rebuild_project(project.id) for project in repository.list_projects()
    )
except Exception:
    logger.exception(
        "application.memory_backfill_failed",
        extra={
            "event": "application.memory_backfill_failed",
            "database_path": str(repository.database_path),
        },
    )
else:
    log_event(
        logger,
        logging.INFO,
        "application.memory_backfill_completed",
        entry_count=rebuilt_entry_count,
    )

log_event(logger, logging.INFO, "application.initialized")



#
# Build the Context for a Completion Request
#

def prepare_conversation_context(
    conversation: Conversation,
    profile: ModelProfile,
    messages: list[Message],
    query_text: str,
) -> ContextPlan:
    """Collect optional sources and assemble a context plan for the model 
    to consume, including the system prompt, memories, message history, 
    and the latest user message."""
    candidates = (
        retrieve_project_memory(
            project_id=conversation.project_id,
            conversation_id=conversation.id,
            query_text=query_text,
        )
        if conversation.memory_enabled
        else []
    )
    output_reserve_tokens = THINKING_MAX_TOKENS if conversation.thinking_enabled else MAX_TOKENS

    # Context is assembled from retrieved memories and message history
    plan = context_assembler.assemble(
        completion_messages(messages),
        candidates,
        context_window_tokens=profile.context_window_tokens,
        output_reserve_tokens=output_reserve_tokens,
    )
    log_event(
        logger,
        logging.INFO,
        "chat.context.prepared",
        model_profile=profile.key,
        context_window_tokens=profile.context_window_tokens,
        output_reserve_tokens=output_reserve_tokens,
        input_budget_tokens=plan.input_budget_tokens,
        estimated_input_tokens=plan.estimated_input_tokens,
        history_message_count=len(messages),
        omitted_history_messages=plan.omitted_history_messages,
        retrieval_candidate_count=len(candidates),
        included_source_count=len(plan.included_sources),
        excluded_source_count=len(plan.excluded_sources),
        excluded_reasons=sorted({source.reason for source in plan.excluded_sources}),
    )
    return plan

#
# Build the message history for Context Assembly
#

def completion_messages(messages: list[Message]) -> list[dict]:
    """Convert stored messages into the OpenAI-style shape, without a system
    message -- that's added by ContextAssembler.assemble() (see
    prepare_conversation_context below), the single place a system message
    ever gets constructed.

    By the time this runs (see prepare_conversation_context), every message
    here belongs to an already-closed turn except the newest one (the
    just-asked question, with nothing after it yet) -- so a closed turn that
    made tool calls gets collapsed to just its user message plus one
    assistant message: a compact
    trace line per tool call (name, arguments, ok/error -- never the tool's
    full result) followed by the turn's final answer. The model only ever
    needed the full tool output to produce that answer, which now fully
    captures it -- the same reasoning as why `reasoning` is already dropped
    here, just applied to tool calls/results instead. This also means a turn
    that took several tool-call rounds collapses to as few messages as a
    plain one, which matters for ContextAssembler's history budget: it
    reasons in messages, so without this, one tool-heavy turn could fill (or
    overflow) the entire window meant to hold several turns of history. A
    turn without any tool activity converts unchanged.
    """
    result: list[dict] = []
    for turn in _group_into_turns(messages):
        result.extend(_completion_messages_for_turn(turn))
    return result


def _group_into_turns(messages: list[Message]) -> list[list[Message]]:
    """Split into turns at each user message -- the same boundary as
    memory.py's _conversation_chunks: a turn is a user message plus
    everything that followed it, up to (not including) the next one.
    """
    turns: list[list[Message]] = []
    current: list[Message] = []
    for message in messages:
        if message.role == "user" and current:
            turns.append(current)
            current = []
        current.append(message)
    if current:
        turns.append(current)
    return turns


def _completion_messages_for_turn(turn: list[Message]) -> list[dict]:
    user_message = next((message for message in turn if message.role == "user"), None)
    has_tool_activity = any(
        message.role == "tool" or (message.role == "assistant" and message.tool_calls)
        for message in turn
    )
    if user_message is None or not has_tool_activity:
        return [{"role": message.role, "content": message.content} for message in turn]

    results_by_call_id = {
        message.tool_call_id: message.content for message in turn if message.role == "tool"
    }
    trace_lines = []
    for message in turn:
        if message.role != "assistant" or not message.tool_calls:
            continue
        for call in message.tool_calls:
            result = results_by_call_id.get(call["id"])
            if result is None:
                outcome = "never returned a result (generation was stopped)"
            elif result.startswith("Error:"):
                outcome = result
            else:
                outcome = "ok"
            trace_lines.append(f"[tool: {call['name']}({call['arguments']}) -> {outcome}]")
    final_answer = next(
        (
            message.content
            for message in reversed(turn)
            if message.role == "assistant" and not message.tool_calls and message.content.strip()
        ),
        None,
    )
    content = "\n".join(trace_lines + ([final_answer] if final_answer else []))
    return [
        {"role": "user", "content": user_message.content},
        {"role": "assistant", "content": content},
    ]

#
# Retrieve (cross chat) Project Memories for Context Assembly
#

def rebuild_conversation_memory(conversation_id: str) -> int:
    """Refresh the rebuildable memory index for one changed conversation."""
    return memory_store.rebuild_conversation(conversation_id)


def retrieve_project_memory(
    *,
    project_id: str,
    conversation_id: str,
    query_text: str,
) -> list[ContextCandidate]:
    """Retrieve relevant turns from other chats in the same project."""
    return memory_store.retrieve(
        RetrievalQuery(
            project_id=project_id,
            text=query_text,
            exclude_conversation_id=conversation_id,
        )
    )



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
) -> AsyncIterator[StreamEvent]:
    """Stream the assistant's reply, executing any tool calls the model makes.

    Yields `TextChunk` for visible model text including reasoning,
    `ToolCallEvent` once per round when the model requests tool calls,
    `ToolResultEvent` once per executed call, and a final `UsageEvent` (see its
    docstring) once the turn's real answer is ready.
    """
    tools = [tool.schema for tool in TOOLS.values()] if profile.supports_tools else None
    conversation_messages: list[dict] = list(messages)
    # See UsageEvent's docstring for what each of these means and why.
    initial_prompt_tokens: int | None = None
    total_completion_tokens = 0
    last_round_total_tokens: int | None = None

    for round_index in range(MAX_TOOL_ROUNDS):
        rounds_remaining = MAX_TOOL_ROUNDS - round_index
        if rounds_remaining <= TOOL_ROUNDS_WARNING_AT:
            # A runtime hint for the model to wrap up instead of continuing to retry
            # Not persisted in the DB, just a system message for this one turn
            conversation_messages.append(
                {
                    "role": "system",
                    "content": (
                        f"You have {rounds_remaining} tool-call round"
                        f"{'s' if rounds_remaining != 1 else ''} left before you must "
                        "give your final answer. If you already have enough "
                        "information -- even if imperfect or incomplete -- answer now "
                        "instead of continuing to search."
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
            tools=tools,
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



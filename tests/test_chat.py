"""Tests for reconstructing OpenAI-shape history from stored messages."""

from datetime import datetime, timezone
from types import SimpleNamespace
import unittest

from unittest import mock

from intelligent_agents_chat import chat
from intelligent_agents_chat.chat import (
    SYSTEM_PROMPT,
    TOOLS,
    TextChunk,
    _execute_tool,
    completion_messages,
    format_reasoning_entry,
    format_tool_call_entry,
    format_tool_result_entry,
    stream_reply,
)
from intelligent_agents_chat.database import Message
from intelligent_agents_chat.llm import ContentDelta
from intelligent_agents_chat.models import ModelProfile
from intelligent_agents_chat.tools import Tool


def _message(
    role: str,
    content: str,
    *,
    tool_calls: tuple[dict, ...] | None = None,
    tool_call_id: str | None = None,
    reasoning: str | None = None,
) -> Message:
    return Message(
        id=1,
        conversation_id="conversation",
        role=role,
        content=content,
        model_profile=None,
        created_at=datetime.now(timezone.utc),
        tool_calls=tool_calls,
        tool_call_id=tool_call_id,
        reasoning=reasoning,
    )


class CompletionMessagesTests(unittest.TestCase):
    def test_plain_messages_pass_through_with_the_system_prompt_prepended(self) -> None:
        messages = completion_messages([_message("user", "Hello")])

        # The system message is SYSTEM_PROMPT with today's real date spliced
        # in fresh on every call (see system_prompt_for_today) -- not the
        # bare constant, so this checks containment, not exact equality.
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn(SYSTEM_PROMPT, messages[0]["content"])
        self.assertIn("Today's date is", messages[0]["content"])
        self.assertEqual(messages[1], {"role": "user", "content": "Hello"})

    def test_assistant_tool_calls_are_reconstructed_in_the_real_api_shape(self) -> None:
        stored_call = {"id": "call_1", "name": "calculator", "arguments": '{"expression": "1+1"}'}
        messages = completion_messages([_message("assistant", "", tool_calls=(stored_call,))])

        self.assertEqual(
            messages[1],
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "calculator",
                            "arguments": '{"expression": "1+1"}',
                        },
                    }
                ],
            },
        )

    def test_tool_messages_are_reconstructed_with_their_tool_call_id(self) -> None:
        messages = completion_messages([_message("tool", "2", tool_call_id="call_1")])

        self.assertEqual(
            messages[1],
            {"role": "tool", "tool_call_id": "call_1", "content": "2"},
        )

    def test_assistant_content_survives_alongside_tool_calls(self) -> None:
        stored_call = {"id": "call_1", "name": "calculator", "arguments": "{}"}
        messages = completion_messages(
            [_message("assistant", "Let me check.", tool_calls=(stored_call,))]
        )

        self.assertEqual(messages[1]["content"], "Let me check.")

    def test_reasoning_is_never_sent_back_to_the_model(self) -> None:
        messages = completion_messages(
            [_message("assistant", "The answer is 2.", reasoning="1 + 1 = 2, obviously.")]
        )

        self.assertEqual(messages[1], {"role": "assistant", "content": "The answer is 2."})


class FormatEntryTests(unittest.TestCase):
    """These are used both live (while streaming) and on replay from the DB --
    see app.py's send_message and render_assistant_turn.
    """

    def test_format_reasoning_entry_includes_the_full_text(self) -> None:
        self.assertIn("because 1+1=2", format_reasoning_entry("because 1+1=2"))

    def test_format_tool_call_entry_includes_name_and_arguments(self) -> None:
        call = {"id": "call_1", "name": "calculator", "arguments": '{"expression": "1+1"}'}

        entry = format_tool_call_entry(call)

        self.assertIn("calculator", entry)
        self.assertIn('{"expression": "1+1"}', entry)

    def test_format_tool_result_entry_includes_name_and_result(self) -> None:
        entry = format_tool_result_entry("calculator", "2")

        self.assertIn("calculator", entry)
        self.assertIn("2", entry)


class ToolRegistryTests(unittest.IsolatedAsyncioTestCase):
    """`TOOLS` is assembled purely from each tool module's exported `Tool` --
    these guard the registry/dispatch wiring itself, not any one tool's logic
    (calculator's own behavior is covered by test_calculator.py, the
    sub-agent's by test_subagent.py).
    """

    def test_every_registered_tool_is_keyed_by_its_own_name(self) -> None:
        for name, tool in TOOLS.items():
            self.assertEqual(name, tool.name)
            self.assertEqual(tool.schema["function"]["name"], tool.name)

    async def test_dispatch_reaches_the_real_tool(self) -> None:
        self.assertEqual(await _execute_tool("calculator", {"expression": "1+1"}), "2")

    async def test_unknown_tool_name_is_a_reported_error_not_a_crash(self) -> None:
        result = await _execute_tool("no-such-tool", {})
        self.assertEqual(result, "Error: unknown tool 'no-such-tool'")

    async def test_a_tool_that_raises_is_caught_not_left_to_crash_the_agent_loop(self) -> None:
        broken_tool = Tool(
            name="broken",
            schema={"type": "function", "function": {"name": "broken"}},
            run=mock.AsyncMock(side_effect=RuntimeError("boom")),
        )
        with mock.patch.dict(chat.TOOLS, {"broken": broken_tool}):
            result = await _execute_tool("broken", {})

        self.assertTrue(result.startswith("Error:"))
        self.assertIn("boom", result)


def _fake_gateway(*delta_lists):
    """A fake `gateway` yielding one canned list of `ContentDelta`s per round
    -- each call to `stream_reply` consumes the next list."""
    rounds = iter(delta_lists)

    async def stream_reply(profile, messages, **kwargs):
        for delta in next(rounds):
            yield delta

    return SimpleNamespace(stream_reply=stream_reply)


class StreamReplyReasoningFallbackTests(unittest.IsolatedAsyncioTestCase):
    """When a final round (no more tool calls) produces only reasoning and no
    real `content` -- typically because vLLM's reasoning parser never saw a
    closing think-tag -- `stream_reply` promotes that reasoning text to a real
    answer instead of leaving the caller with nothing. See chat.py.
    """

    PROFILE = ModelProfile(key="test", label="Test", base_url="http://x/v1", model="test-model")

    async def test_reasoning_only_round_is_used_as_the_final_answer(self) -> None:
        gateway = _fake_gateway([ContentDelta("This is actually the answer.", is_reasoning=True)])
        with mock.patch.object(chat, "gateway", gateway):
            events = [event async for event in stream_reply(self.PROFILE, [])]

        final_answer = "".join(
            e.text for e in events if isinstance(e, TextChunk) and not e.is_reasoning
        )
        self.assertEqual(final_answer, "This is actually the answer.")

    async def test_a_round_with_real_content_is_not_duplicated(self) -> None:
        gateway = _fake_gateway(
            [ContentDelta("thinking...", is_reasoning=True), ContentDelta("Real answer.")]
        )
        with mock.patch.object(chat, "gateway", gateway):
            events = [event async for event in stream_reply(self.PROFILE, [])]

        final_answer = "".join(
            e.text for e in events if isinstance(e, TextChunk) and not e.is_reasoning
        )
        self.assertEqual(final_answer, "Real answer.")

    async def test_the_fallback_does_not_apply_to_a_tool_calling_round(self) -> None:
        profile = ModelProfile(
            key="tools",
            label="Tools",
            base_url="http://x/v1",
            model="test-model",
            supports_tools=True,
        )
        # Round 1: reasoning only, then a tool call -- no content is expected
        # here regardless, so no fallback should be synthesized before the
        # ToolCallEvent. Round 2: a normal, real-content answer.
        gateway_calls: list[list[ContentDelta]] = [
            [ContentDelta("deciding to call the tool...", is_reasoning=True)],
            [ContentDelta("2")],
        ]
        rounds = iter(gateway_calls)
        tool_calls_by_round = iter(
            [[{"id": "call_1", "name": "calculator", "arguments": '{"expression": "1+1"}'}], []]
        )

        async def fake_stream_reply(profile, messages, *, tool_calls=None, **kwargs):
            tool_calls.extend(next(tool_calls_by_round))
            for delta in next(rounds):
                yield delta

        with mock.patch.object(chat, "gateway", SimpleNamespace(stream_reply=fake_stream_reply)):
            events = [event async for event in stream_reply(profile, [])]

        text_events_before_tool_call = []
        for event in events:
            if isinstance(event, chat.ToolCallEvent):
                break
            text_events_before_tool_call.append(event)
        self.assertFalse(any(not e.is_reasoning for e in text_events_before_tool_call))


class ToolRoundBudgetTests(unittest.IsolatedAsyncioTestCase):
    """Real case this addresses: with generous round budgets, a model that
    hasn't found a fully satisfying answer can keep retrying (e.g.
    re-searching the same page with ever more specific terms) all the way to
    MAX_TOOL_ROUNDS without ever concluding. stream_reply nudges it to wrap
    up once few rounds remain.
    """

    async def test_warning_appears_only_once_few_rounds_remain(self) -> None:
        profile = ModelProfile(
            key="tools",
            label="Tools",
            base_url="http://x/v1",
            model="test-model",
            supports_tools=True,
        )
        # A model that never stops calling tools on its own -- forces every
        # round up to the (patched, small) MAX_TOOL_ROUNDS to run.
        calls_seen: list[list[dict]] = []

        async def fake_stream_reply(profile, messages, *, tool_calls=None, **kwargs):
            calls_seen.append(list(messages))
            tool_calls.append(
                {"id": f"call_{len(calls_seen)}", "name": "calculator", "arguments": "{}"}
            )
            yield ContentDelta("thinking", is_reasoning=True)

        def has_warning(messages: list[dict]) -> bool:
            return any(
                m.get("role") == "system" and "tool-call round" in (m.get("content") or "")
                for m in messages
            )

        with (
            mock.patch.object(chat, "gateway", SimpleNamespace(stream_reply=fake_stream_reply)),
            mock.patch.object(chat, "MAX_TOOL_ROUNDS", 3),
        ):
            [event async for event in stream_reply(profile, [{"role": "user", "content": "hi"}])]

        self.assertEqual(len(calls_seen), 3)
        self.assertFalse(has_warning(calls_seen[0]))  # 3 rounds left, > TOOL_ROUNDS_WARNING_AT
        self.assertTrue(has_warning(calls_seen[1]))  # 2 left
        self.assertTrue(has_warning(calls_seen[2]))  # 1 left


if __name__ == "__main__":
    unittest.main()

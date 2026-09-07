"""Tests for the tool registry and the streaming agent loop."""

from types import SimpleNamespace
import unittest

from unittest import mock

from intelligent_agents_chat import chat
from intelligent_agents_chat.chat import (
    TOOLS,
    TextChunk,
    UsageEvent,
    _execute_tool,
    format_reasoning_entry,
    format_tool_call_entry,
    format_tool_result_entry,
    stream_reply,
)
from intelligent_agents_chat.llm import ContentDelta
from intelligent_agents_chat.models import ModelProfile
from intelligent_agents_chat.tools import Tool


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
        result = await _execute_tool("calculator", {"expression": "1+1"}, TOOLS)
        self.assertEqual(result, "2")

    async def test_unknown_tool_name_is_a_reported_error_not_a_crash(self) -> None:
        result = await _execute_tool("no-such-tool", {}, TOOLS)
        self.assertEqual(result, "Error: unknown tool 'no-such-tool'")

    async def test_a_tool_that_raises_is_caught_not_left_to_crash_the_agent_loop(self) -> None:
        broken_tool = Tool(
            name="broken",
            schema={"type": "function", "function": {"name": "broken"}},
            run=mock.AsyncMock(side_effect=RuntimeError("boom")),
        )
        result = await _execute_tool("broken", {}, {"broken": broken_tool})

        self.assertTrue(result.startswith("Error:"))
        self.assertIn("boom", result)

    async def test_recall_tool_output_is_offered_only_when_a_conversation_id_is_given(
        self,
    ) -> None:
        """recall_tool_output needs a conversation to scope itself to (see
        stream_reply) -- built fresh per turn, not part of the static TOOLS
        registry above, so it must only show up in the schema the model
        actually sees once a real conversation_id is passed in.
        """
        profile = ModelProfile(
            key="tools", label="Tools", base_url="http://x/v1", model="test-model",
            supports_tools=True,
        )
        offered_tool_names: list[list[str]] = []

        async def fake_stream_reply(profile, messages, *, tools=None, tool_calls=None, **kwargs):
            offered_tool_names.append([tool["function"]["name"] for tool in (tools or [])])
            yield ContentDelta("answer")

        with mock.patch.object(chat, "gateway", SimpleNamespace(stream_reply=fake_stream_reply)):
            [event async for event in stream_reply(profile, [])]
            [event async for event in stream_reply(profile, [], conversation_id="conversation-1")]

        self.assertNotIn("recall_tool_output", offered_tool_names[0])
        self.assertIn("recall_tool_output", offered_tool_names[1])


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


class UsageEventTests(unittest.IsolatedAsyncioTestCase):
    """UsageEvent reports the model server's real usage for the whole turn:
    prompt_tokens from the initial request only (later rounds' prompts grow
    with tool activity, which isn't what ContextAssembler budgeted for), but
    completion_tokens summed across every round -- each tool-call round is
    its own separate completion request with its own full output budget (see
    llm.py's stream_reply and chat.py's UsageEvent docstring).
    """

    PROFILE = ModelProfile(
        key="tools",
        label="Tools",
        base_url="http://x/v1",
        model="test-model",
        supports_tools=True,
    )

    async def test_prompt_from_round_one_completion_summed_peak_from_the_last_round(
        self,
    ) -> None:
        call = {"id": "call_1", "name": "calculator", "arguments": "{}"}
        rounds = iter([[ContentDelta("")], [ContentDelta("2")]])
        tool_calls_by_round = iter([[call], []])
        usage_by_round = iter(
            [
                {"prompt_tokens": 100, "completion_tokens": 5, "total_tokens": 105},
                {"prompt_tokens": 150, "completion_tokens": 3, "total_tokens": 153},
            ]
        )

        async def fake_stream_reply(profile, messages, *, tool_calls=None, usage=None, **kwargs):
            tool_calls.extend(next(tool_calls_by_round))
            if usage is not None:
                usage.update(next(usage_by_round))
            for delta in next(rounds):
                yield delta

        with mock.patch.object(chat, "gateway", SimpleNamespace(stream_reply=fake_stream_reply)):
            events = [event async for event in stream_reply(self.PROFILE, [])]

        usage_events = [event for event in events if isinstance(event, UsageEvent)]
        self.assertEqual(len(usage_events), 1)
        self.assertEqual(usage_events[0].prompt_tokens, 100)
        self.assertEqual(usage_events[0].completion_tokens, 8)
        # Round 2's total (153), not round 1's (105) -- the conversation only
        # ever grows, so the last round always holds the peak.
        self.assertEqual(usage_events[0].peak_total_tokens, 153)

    async def test_no_usage_event_when_the_server_never_reports_usage(self) -> None:
        gateway = _fake_gateway([ContentDelta("Hello")])
        with mock.patch.object(chat, "gateway", gateway):
            events = [event async for event in stream_reply(self.PROFILE, [])]

        self.assertFalse(any(isinstance(event, UsageEvent) for event in events))


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
            # role: "user", not "system" -- see stream_reply's comment on why.
            return any(
                m.get("role") == "user" and "tool-call round" in (m.get("content") or "")
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

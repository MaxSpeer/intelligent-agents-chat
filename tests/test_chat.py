"""Tests for reconstructing OpenAI-shape history from stored messages."""

from datetime import datetime, timezone
from types import SimpleNamespace
import unittest

from unittest import mock

from intelligent_agents_chat import chat
from intelligent_agents_chat.chat import (
    TOOLS,
    TextChunk,
    _execute_tool,
    completion_messages,
    format_reasoning_entry,
    format_tool_call_entry,
    format_tool_result_entry,
    prepare_conversation_context,
    stream_reply,
)
from intelligent_agents_chat.database import Conversation, Message
from intelligent_agents_chat.llm import ContentDelta, SYSTEM_PROMPT
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
    """completion_messages only reshapes stored Messages into the OpenAI-style
    dicts -- it never adds a system message itself. That's ContextAssembler's
    job (see ContextAssemblerTests in test_context.py and
    ToolRoundBudgetTests below for the system message's actual content).

    It groups messages into turns (a user message plus everything up to the
    next one -- see _group_into_turns) and collapses any turn that made tool
    calls into just its user message plus one assistant message: a trace
    line per tool call, no full tool output, plus the turn's final answer.
    """

    def test_a_turn_without_tool_activity_passes_through_unchanged(self) -> None:
        messages = completion_messages(
            [_message("user", "What is 1+1?"), _message("assistant", "It's 2.")]
        )

        self.assertEqual(
            messages,
            [
                {"role": "user", "content": "What is 1+1?"},
                {"role": "assistant", "content": "It's 2."},
            ],
        )

    def test_a_lone_new_user_message_passes_through_unchanged(self) -> None:
        """The turn a request is being prepared for -- no answer yet."""
        messages = completion_messages([_message("user", "Hello")])

        self.assertEqual(messages, [{"role": "user", "content": "Hello"}])

    def test_reasoning_is_never_sent_back_to_the_model(self) -> None:
        messages = completion_messages(
            [
                _message("user", "What is 1+1?"),
                _message("assistant", "The answer is 2.", reasoning="1 + 1 = 2, obviously."),
            ]
        )

        self.assertEqual(messages[1], {"role": "assistant", "content": "The answer is 2."})

    def test_a_turns_tool_call_rounds_collapse_to_one_trace_line_each_plus_the_answer(
        self,
    ) -> None:
        call_1 = {"id": "call_1", "name": "calculator", "arguments": '{"expression": "1+1"}'}
        call_2 = {"id": "call_2", "name": "calculator", "arguments": '{"expression": "2+2"}'}
        messages = completion_messages(
            [
                _message("user", "What is 1+1 and 2+2?"),
                _message("assistant", "Let me compute that.", tool_calls=(call_1,)),
                _message("tool", "2", tool_call_id="call_1"),
                _message("assistant", "", tool_calls=(call_2,)),
                _message("tool", "4", tool_call_id="call_2"),
                _message("assistant", "1+1 is 2 and 2+2 is 4."),
            ]
        )

        self.assertEqual(
            messages,
            [
                {"role": "user", "content": "What is 1+1 and 2+2?"},
                {
                    "role": "assistant",
                    "content": (
                        '[tool: calculator({"expression": "1+1"}) -> ok]\n'
                        '[tool: calculator({"expression": "2+2"}) -> ok]\n'
                        "1+1 is 2 and 2+2 is 4."
                    ),
                },
            ],
        )
        # The full tool results ("2", "4") and the narration before the first
        # call ("Let me compute that.") are gone, not just hidden.
        self.assertNotIn("Let me compute that.", messages[1]["content"])

    def test_a_tool_error_result_is_reported_in_the_trace_line(self) -> None:
        call = {"id": "call_1", "name": "fetch_page", "arguments": '{"url": "not-a-url"}'}
        messages = completion_messages(
            [
                _message("user", "Fetch that page."),
                _message("assistant", "", tool_calls=(call,)),
                _message("tool", "Error: 'url' must start with http:// or https://.", tool_call_id="call_1"),
                _message("assistant", "I couldn't fetch that -- the URL looks invalid."),
            ]
        )

        self.assertIn(
            "Error: 'url' must start with http:// or https://.", messages[1]["content"]
        )

    def test_a_tool_call_with_no_result_is_reported_as_interrupted(self) -> None:
        """The turn was stopped between the tool call and its result being
        saved -- e.g. the user hit stop mid-round. Distinct from an empty/ok
        result so it isn't mistaken for success.
        """
        call = {"id": "call_1", "name": "fetch_page", "arguments": '{"url": "https://x"}'}
        messages = completion_messages(
            [_message("user", "Fetch that page."), _message("assistant", "", tool_calls=(call,))]
        )

        self.assertIn("never returned a result", messages[1]["content"])

    def test_several_turns_in_one_history_are_each_grouped_independently(self) -> None:
        call = {"id": "call_1", "name": "calculator", "arguments": "{}"}
        messages = completion_messages(
            [
                _message("user", "First question."),
                _message("assistant", "First answer."),
                _message("user", "Second question."),
                _message("assistant", "", tool_calls=(call,)),
                _message("tool", "42", tool_call_id="call_1"),
                _message("assistant", "Second answer."),
                _message("user", "Third question, not answered yet."),
            ]
        )

        self.assertEqual(len(messages), 5)  # 2 + 2 + 1, not 2 + 3 + 1
        self.assertEqual(messages[0], {"role": "user", "content": "First question."})
        self.assertEqual(messages[1], {"role": "assistant", "content": "First answer."})
        self.assertEqual(messages[2], {"role": "user", "content": "Second question."})
        self.assertIn("ok", messages[3]["content"])
        self.assertIn("Second answer.", messages[3]["content"])
        self.assertEqual(
            messages[4], {"role": "user", "content": "Third question, not answered yet."}
        )


class PrepareConversationContextWiringTests(unittest.TestCase):
    """Exercises the actual module-level chat.context_assembler, not a
    freshly-constructed ContextAssembler -- this is what would have caught the
    regression where context_assembler got built once at import time from the
    static SYSTEM_PROMPT instead of system_prompt_for_today, silently freezing
    the date the process happened to start on.
    """

    def test_the_real_module_wiring_produces_a_system_message_with_todays_date(self) -> None:
        profile = ModelProfile(
            key="tools",
            label="Tools",
            base_url="http://x/v1",
            model="test-model",
        )
        conversation = Conversation(
            id="conversation",
            project_id="project",
            title="Test",
            model_profile=profile.key,
            thinking_enabled=False,
            memory_enabled=False,  # skips retrieval -- no DB needed for this test
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

        plan = prepare_conversation_context(
            conversation,
            profile,
            [_message("user", "Hello")],
            "Hello",
        )

        system_message = plan.messages[0]
        self.assertEqual(system_message["role"], "system")
        self.assertIn(SYSTEM_PROMPT, system_message["content"])
        self.assertIn("Today's date is", system_message["content"])


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

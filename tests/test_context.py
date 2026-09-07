"""Token-aware context assembly tests, plus the message-history/compaction
pipeline that feeds it (shared by memory and future RAG).
"""

from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from unittest import mock

from intelligent_agents_chat import context
from intelligent_agents_chat.context import (
    MEMORY_GUARD,
    ContextAssembler,
    ContextOverflowError,
    _compact_conversation_history,
    _history_messages,
    completion_messages,
    prepare_conversation_context,
)
from intelligent_agents_chat.database import ChatRepository, Conversation, Message
from intelligent_agents_chat.llm import SYSTEM_PROMPT
from intelligent_agents_chat.models import ModelProfile
from intelligent_agents_chat.retrieval import ContextCandidate


def candidate(source_id: str = "1", *, text: str = "The decision was SQLite.") -> ContextCandidate:
    return ContextCandidate(
        source_kind="project_memory",
        source_id=source_id,
        project_id="project-a",
        source_conversation_id="source-chat",
        text=text,
        title="Architecture",
        locator="messages 1-2",
        score=0.5,
    )


def _message(
    role: str,
    content: str,
    *,
    message_id: int = 1,
    tool_calls: tuple[dict, ...] | None = None,
    tool_call_id: str | None = None,
    reasoning: str | None = None,
) -> Message:
    return Message(
        id=message_id,
        conversation_id="conversation",
        role=role,
        content=content,
        model_profile=None,
        created_at=datetime.now(timezone.utc),
        tool_calls=tool_calls,
        tool_call_id=tool_call_id,
        reasoning=reasoning,
    )


class ContextAssemblerTests(unittest.TestCase):
    def test_includes_memory_with_guard_and_auditable_source(self) -> None:
        plan = ContextAssembler("Base system prompt", memory_budget_tokens=500).assemble(
            [
                {"role": "user", "content": "Earlier question"},
                {"role": "assistant", "content": "Earlier answer"},
                {"role": "user", "content": "What database did we choose?"},
            ],
            [candidate()],
            context_window_tokens=2_000,
            output_reserve_tokens=200,
        )

        self.assertEqual(len(plan.included_sources), 1)
        self.assertEqual(plan.included_sources[0].rank, 1)
        self.assertIn(MEMORY_GUARD, plan.messages[0]["content"])
        self.assertEqual(plan.messages[1]["role"], "user")
        self.assertIn("<project-memory>", plan.messages[1]["content"])
        self.assertIn("[project_memory:1]", plan.messages[1]["content"])
        self.assertEqual(plan.messages[-1]["content"], "What database did we choose?")
        self.assertLessEqual(plan.estimated_input_tokens, plan.input_budget_tokens)

    def test_excludes_memory_when_its_complete_wrapper_exceeds_budget(self) -> None:
        plan = ContextAssembler("System", memory_budget_tokens=20).assemble(
            [{"role": "user", "content": "Question"}],
            [candidate(text="large " * 100)],
            context_window_tokens=500,
            output_reserve_tokens=100,
        )

        self.assertEqual(plan.included_sources, ())
        self.assertEqual(len(plan.excluded_sources), 1)
        self.assertEqual(plan.excluded_sources[0].reason, "memory_budget_exceeded")
        self.assertNotIn(MEMORY_GUARD, plan.messages[0]["content"])

    def test_history_is_filled_newest_first_and_the_oldest_is_cut_when_it_does_not_fit(
        self,
    ) -> None:
        plan = ContextAssembler("", memory_budget_tokens=0).assemble(
            [
                {"role": "user", "content": "old " * 100},
                {"role": "assistant", "content": "old reply " * 100},
                {"role": "user", "content": "recent question"},
                {"role": "assistant", "content": "recent answer"},
                {"role": "user", "content": "latest question"},
            ],
            [],
            context_window_tokens=90,
            output_reserve_tokens=30,
        )

        self.assertEqual(
            [message["content"] for message in plan.messages],
            ["recent question", "recent answer", "latest question"],
        )
        self.assertEqual(plan.omitted_history_messages, 2)
        self.assertLessEqual(plan.estimated_input_tokens, plan.input_budget_tokens)

    def test_memory_has_priority_over_history_even_the_most_recent_message(self) -> None:
        """Memory gets first claim on the leftover budget -- not just on
        whatever history's newest-first fill didn't already spend. Without
        this, a long conversation could fill the whole budget on its own
        recent history and starve memory, even though memory is often the
        only way to recall something from a *different* chat.
        """
        plan = ContextAssembler("", memory_budget_tokens=200).assemble(
            [
                {"role": "user", "content": "recent question"},
                {"role": "assistant", "content": "recent answer"},
                {"role": "user", "content": "latest question"},
            ],
            [candidate()],
            context_window_tokens=250,
            output_reserve_tokens=100,
        )

        self.assertEqual(len(plan.included_sources), 1)
        all_content = "\n".join(message["content"] for message in plan.messages)
        self.assertNotIn("recent question", all_content)
        self.assertNotIn("recent answer", all_content)
        self.assertEqual(plan.messages[-1], {"role": "user", "content": "latest question"})
        self.assertEqual(plan.omitted_history_messages, 2)

    def test_raises_when_mandatory_input_cannot_fit(self) -> None:
        with self.assertRaisesRegex(ContextOverflowError, "latest user message"):
            ContextAssembler("System").assemble(
                [{"role": "user", "content": "x" * 300}],
                [],
                context_window_tokens=100,
                output_reserve_tokens=50,
            )


class CompletionMessagesTests(unittest.TestCase):
    """completion_messages only reshapes stored Messages into the OpenAI-style
    dicts -- it never adds a system message itself. That's ContextAssembler's
    job (see ContextAssemblerTests above and ToolRoundBudgetTests in
    test_chat.py for the system message's actual content).

    It groups messages into turns (a user message plus everything up to the
    next one -- see _group_into_turns) and collapses any turn that made tool
    calls into its user message, a bracketed trace note (one line per tool
    call, no full tool output), and its final answer as a plain assistant
    message -- kept out of the assistant message on purpose, see
    completion_messages' own docstring for why.
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
        # A trailing turn, so the tool-calling turn isn't the latest one --
        # only a closed turn collapses (see
        # test_the_latest_turn_is_never_collapsed_even_with_tool_activity).
        messages = completion_messages(
            [
                _message("user", "What is 1+1 and 2+2?"),
                _message("assistant", "Let me compute that.", tool_calls=(call_1,)),
                _message("tool", "2", message_id=10, tool_call_id="call_1"),
                _message("assistant", "", tool_calls=(call_2,)),
                _message("tool", "4", message_id=11, tool_call_id="call_2"),
                _message("assistant", "1+1 is 2 and 2+2 is 4."),
                _message("user", "Thanks, anything else?"),
            ]
        )

        self.assertEqual(len(messages), 4)
        self.assertEqual(messages[0], {"role": "user", "content": "What is 1+1 and 2+2?"})
        # role: "user", not "system" -- see completion_messages' docstring
        # (vLLM rejects a system message that isn't first, and this note is
        # framed to not read as something the user said).
        self.assertEqual(messages[1]["role"], "user")
        self.assertIn("not from the user", messages[1]["content"])
        # Tagged with each result's own stored message id -- see
        # recall_tool_output, which lets the model ask for one back in full.
        self.assertIn(
            '[tool: calculator({"expression": "1+1"}) -> ok (id: 10)]', messages[1]["content"]
        )
        self.assertIn(
            '[tool: calculator({"expression": "2+2"}) -> ok (id: 11)]', messages[1]["content"]
        )
        self.assertEqual(messages[2], {"role": "assistant", "content": "1+1 is 2 and 2+2 is 4."})
        self.assertEqual(messages[3], {"role": "user", "content": "Thanks, anything else?"})
        # The full tool results ("2", "4") and the narration before the first
        # call ("Let me compute that.") are gone, not just hidden. And the
        # trace is its own message, not mixed into the assistant message --
        # see completion_messages' docstring for why that matters (a
        # chat-tuned model reads role="assistant" content as its own voice,
        # and would otherwise start imitating the trace notation).
        self.assertNotIn("Let me compute that.", "\n".join(m["content"] for m in messages))
        self.assertNotIn("[tool:", messages[2]["content"])

    def test_a_tool_error_result_is_reported_in_the_trace_line(self) -> None:
        call = {"id": "call_1", "name": "fetch_page", "arguments": '{"url": "not-a-url"}'}
        messages = completion_messages(
            [
                _message("user", "Fetch that page."),
                _message("assistant", "", tool_calls=(call,)),
                _message("tool", "Error: 'url' must start with http:// or https://.", tool_call_id="call_1"),
                _message("assistant", "I couldn't fetch that -- the URL looks invalid."),
                _message("user", "Try a different URL then."),
            ]
        )

        self.assertEqual(messages[1]["role"], "user")
        self.assertIn(
            "Error: 'url' must start with http:// or https://.", messages[1]["content"]
        )

    def test_a_tool_call_with_no_result_is_reported_as_interrupted(self) -> None:
        """The turn was stopped between the tool call and its result being
        saved -- e.g. the user hit stop mid-round. Distinct from an empty/ok
        result so it isn't mistaken for success. No final answer exists yet
        either, so there's no assistant message at all -- just the question
        and the trace note. A trailing turn closes it, otherwise (being the
        latest) it wouldn't be collapsed at all -- see
        test_the_latest_turn_is_never_collapsed_even_with_tool_activity.
        """
        call = {"id": "call_1", "name": "fetch_page", "arguments": '{"url": "https://x"}'}
        messages = completion_messages(
            [
                _message("user", "Fetch that page."),
                _message("assistant", "", tool_calls=(call,)),
                _message("user", "Never mind, try this instead."),
            ]
        )

        self.assertEqual(len(messages), 3)
        self.assertEqual(messages[1]["role"], "user")
        self.assertIn("never returned a result", messages[1]["content"])
        self.assertEqual(messages[2], {"role": "user", "content": "Never mind, try this instead."})

    def test_the_latest_turn_is_never_collapsed_even_with_tool_activity(self) -> None:
        """Only a turn closed by a *later* user message gets collapsed --
        the latest turn never does, even if it already made tool calls. This
        is what lets app.py retry a generation after a context-overflow
        (ContextLengthExceededError) without losing that turn's own tool
        results: no new user message was added, so this turn is still "the
        latest" and the model sees it in full on the retry.
        """
        call = {"id": "call_1", "name": "calculator", "arguments": '{"expression": "1+1"}'}
        messages = completion_messages(
            [
                _message("user", "What is 1+1?"),
                _message("assistant", "Let me check.", tool_calls=(call,)),
                _message("tool", "2", tool_call_id="call_1"),
            ]
        )

        self.assertEqual(
            messages,
            [
                {"role": "user", "content": "What is 1+1?"},
                {
                    "role": "assistant",
                    "content": "Let me check.",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "calculator", "arguments": '{"expression": "1+1"}'},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "2"},
            ],
        )

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

        self.assertEqual(len(messages), 6)  # 2 + 3 + 1, not 2 + 2 + 1
        self.assertEqual(messages[0], {"role": "user", "content": "First question."})
        self.assertEqual(messages[1], {"role": "assistant", "content": "First answer."})
        self.assertEqual(messages[2], {"role": "user", "content": "Second question."})
        self.assertEqual(messages[3]["role"], "user")
        self.assertIn("ok", messages[3]["content"])
        self.assertEqual(messages[4], {"role": "assistant", "content": "Second answer."})
        self.assertEqual(
            messages[5], {"role": "user", "content": "Third question, not answered yet."}
        )


class PrepareConversationContextWiringTests(unittest.IsolatedAsyncioTestCase):
    """Exercises the actual module-level context.context_assembler, not a
    freshly-constructed ContextAssembler -- this is what would have caught the
    regression where context_assembler got built once at import time from the
    static SYSTEM_PROMPT instead of system_prompt_for_today, silently freezing
    the date the process happened to start on.
    """

    async def test_the_real_module_wiring_produces_a_system_message_with_todays_date(
        self,
    ) -> None:
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
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

        # A single message can never be truncated, so this never touches
        # compaction (no subagent call, no DB write) -- see the dedicated
        # ConversationCompactionTests below for that.
        plan = await prepare_conversation_context(
            conversation,
            profile,
            [_message("user", "Hello")],
            "Hello",
            thinking_enabled=False,
            memory_enabled=False,  # skips retrieval -- no DB needed for this test
        )

        system_message = plan.messages[0]
        self.assertEqual(system_message["role"], "system")
        self.assertIn(SYSTEM_PROMPT, system_message["content"])
        self.assertIn("Today's date is", system_message["content"])


class ConversationCompactionTests(unittest.IsolatedAsyncioTestCase):
    """Context compaction: summarize turns older than
    context.COMPACTION_KEEP_RECENT_TURNS instead of silently cutting them once
    history no longer fits the input budget (see prepare_conversation_context).
    COMPACTION_KEEP_RECENT_TURNS is patched down to 2 so tests don't need a
    dozen turns to exercise the boundary.
    """

    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "chats.sqlite3"
        self.repository = ChatRepository(self.database_path)
        self.repository.initialize()
        self.conversation = self.repository.create_conversation()
        self.repository_patch = mock.patch.object(context, "repository", self.repository)
        self.repository_patch.start()
        self.keep_recent_patch = mock.patch.object(context, "COMPACTION_KEEP_RECENT_TURNS", 2)
        self.keep_recent_patch.start()
        self.addCleanup(self.keep_recent_patch.stop)
        self.addCleanup(self.repository_patch.stop)
        self.addCleanup(self.temporary_directory.cleanup)

    def _add_turn(self, user_text: str, assistant_text: str) -> None:
        self.repository.add_message(self.conversation.id, "user", user_text)
        self.repository.add_message(
            self.conversation.id, "assistant", assistant_text, model_profile="base"
        )

    def _fake_subagent(self, summary_text: str = "a summary"):
        calls: list[str] = []

        async def run(arguments: dict) -> str:
            calls.append(arguments["task"])
            return summary_text

        return SimpleNamespace(run=run), calls

    async def test_noop_when_there_are_not_enough_turns_yet(self) -> None:
        self._add_turn("Q1", "A1")
        self._add_turn("Q2", "A2")  # == COMPACTION_KEEP_RECENT_TURNS -- nothing to compact
        messages = self.repository.list_messages(self.conversation.id)
        fake_subagent, calls = self._fake_subagent()

        with mock.patch.object(context, "subagent", fake_subagent):
            compacted = await _compact_conversation_history(self.conversation.id, messages)

        self.assertFalse(compacted)
        self.assertEqual(calls, [])
        self.assertIsNone(self.repository.get_conversation_compaction(self.conversation.id))

    async def test_compacts_everything_except_the_most_recent_turns(self) -> None:
        self._add_turn("Q1", "A1")
        self._add_turn("Q2", "A2")
        self._add_turn("Q3", "A3")
        self._add_turn("Q4", "A4")
        messages = self.repository.list_messages(self.conversation.id)
        fake_subagent, calls = self._fake_subagent("Q1/A1 and Q2/A2 happened.")

        with mock.patch.object(context, "subagent", fake_subagent):
            compacted = await _compact_conversation_history(self.conversation.id, messages)

        self.assertTrue(compacted)
        self.assertEqual(len(calls), 1)
        self.assertIn("Q1", calls[0])
        self.assertIn("A1", calls[0])
        self.assertIn("Q2", calls[0])
        self.assertIn("A2", calls[0])
        # The kept-recent turns must never leak into the summarization task.
        self.assertNotIn("Q3", calls[0])
        self.assertNotIn("Q4", calls[0])

        record = self.repository.get_conversation_compaction(self.conversation.id)
        assert record is not None
        self.assertEqual(record.summary, "Q1/A1 and Q2/A2 happened.")
        # Boundary is the last message of the last compacted turn (Q2/A2's reply).
        self.assertEqual(messages[3].content, "A2")
        self.assertEqual(record.compacted_through_message_id, messages[3].id)

    async def test_extends_an_existing_compaction_with_only_the_new_delta(self) -> None:
        self._add_turn("Q1", "A1")
        self._add_turn("Q2", "A2")
        self._add_turn("Q3", "A3")
        self._add_turn("Q4", "A4")
        messages = self.repository.list_messages(self.conversation.id)
        first_subagent, _ = self._fake_subagent("Summary of Q1/Q2.")
        with mock.patch.object(context, "subagent", first_subagent):
            await _compact_conversation_history(self.conversation.id, messages)

        # Two more turns age past the keep-recent window.
        self._add_turn("Q5", "A5")
        self._add_turn("Q6", "A6")
        messages = self.repository.list_messages(self.conversation.id)
        second_subagent, second_calls = self._fake_subagent("Summary of Q1 through Q4.")

        with mock.patch.object(context, "subagent", second_subagent):
            compacted = await _compact_conversation_history(self.conversation.id, messages)

        self.assertTrue(compacted)
        self.assertEqual(len(second_calls), 1)
        task = second_calls[0]
        # Extends the previous summary -- doesn't re-read Q1/Q2's raw text.
        self.assertIn("Summary of Q1/Q2.", task)
        self.assertIn("Q3", task)
        self.assertIn("Q4", task)
        self.assertNotIn("A1", task)

        record = self.repository.get_conversation_compaction(self.conversation.id)
        assert record is not None
        self.assertEqual(record.summary, "Summary of Q1 through Q4.")

    async def test_returns_false_when_nothing_new_since_the_last_compaction(self) -> None:
        self._add_turn("Q1", "A1")
        self._add_turn("Q2", "A2")
        self._add_turn("Q3", "A3")
        self._add_turn("Q4", "A4")
        messages = self.repository.list_messages(self.conversation.id)
        fake_subagent, calls = self._fake_subagent()
        with mock.patch.object(context, "subagent", fake_subagent):
            await _compact_conversation_history(self.conversation.id, messages)

        with mock.patch.object(context, "subagent", fake_subagent):
            compacted_again = await _compact_conversation_history(self.conversation.id, messages)

        self.assertFalse(compacted_again)
        self.assertEqual(len(calls), 1)  # only the first call happened

    async def test_history_messages_splices_in_the_summary_and_drops_covered_messages(
        self,
    ) -> None:
        self._add_turn("Q1", "A1")
        self._add_turn("Q2", "A2")
        self._add_turn("Q3", "A3")
        self._add_turn("Q4", "A4")
        messages = self.repository.list_messages(self.conversation.id)
        fake_subagent, _ = self._fake_subagent("Q1 and Q2 happened.")
        with mock.patch.object(context, "subagent", fake_subagent):
            await _compact_conversation_history(self.conversation.id, messages)

        history = _history_messages(self.conversation.id, messages)

        # role: "user", not "system" -- see _history_messages' comment (vLLM
        # rejects a system message that isn't first).
        self.assertEqual(history[0]["role"], "user")
        self.assertIn("Q1 and Q2 happened.", history[0]["content"])
        remaining_text = "\n".join(message["content"] for message in history[1:])
        self.assertIn("Q3", remaining_text)
        self.assertIn("Q4", remaining_text)
        self.assertNotIn("Q1", remaining_text)
        self.assertNotIn("Q2", remaining_text)

    async def test_history_messages_without_any_compaction_is_unchanged(self) -> None:
        self._add_turn("Q1", "A1")
        messages = self.repository.list_messages(self.conversation.id)

        self.assertEqual(
            _history_messages(self.conversation.id, messages), completion_messages(messages)
        )


class PrepareConversationContextCompactionTests(unittest.IsolatedAsyncioTestCase):
    """End-to-end: prepare_conversation_context triggers compaction exactly
    when the first assembly attempt has to cut older history, then uses the
    compacted result on a single retry -- never when everything already fits.
    """

    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "chats.sqlite3"
        self.repository = ChatRepository(self.database_path)
        self.repository.initialize()
        self.conversation = self.repository.create_conversation()
        self.repository_patch = mock.patch.object(context, "repository", self.repository)
        self.repository_patch.start()
        self.keep_recent_patch = mock.patch.object(context, "COMPACTION_KEEP_RECENT_TURNS", 1)
        self.keep_recent_patch.start()
        self.addCleanup(self.keep_recent_patch.stop)
        self.addCleanup(self.repository_patch.stop)
        self.addCleanup(self.temporary_directory.cleanup)

    def _add_turn(self, user_text: str, assistant_text: str) -> None:
        self.repository.add_message(self.conversation.id, "user", user_text)
        self.repository.add_message(
            self.conversation.id, "assistant", assistant_text, model_profile="base"
        )

    async def test_compacts_when_history_would_otherwise_be_cut(self) -> None:
        for index in range(5):
            self._add_turn(f"Question {index} details " * 30, f"Answer {index} details " * 30)
        self.repository.add_message(self.conversation.id, "user", "Latest question")
        messages = self.repository.list_messages(self.conversation.id)
        conversation = self.repository.get_conversation(self.conversation.id)
        assert conversation is not None
        profile = ModelProfile(
            key="tiny",
            label="Tiny",
            base_url="http://x/v1",
            model="test-model",
            # Comfortably above MAX_TOKENS (the output reserve, or assemble()
            # raises ContextOverflowError outright) but too small to hold all
            # five padded turns -- forces the omitted_history_messages > 0
            # path without also tripping the mandatory-input check.
            context_window_tokens=2_000,
        )

        async def fake_run(arguments: dict) -> str:
            return "Compact summary of the earlier turns."

        compacting_calls = 0

        def on_compacting() -> None:
            nonlocal compacting_calls
            compacting_calls += 1

        with mock.patch.object(context, "subagent", SimpleNamespace(run=fake_run)):
            plan = await prepare_conversation_context(
                conversation,
                profile,
                messages,
                "Latest question",
                thinking_enabled=False,
                memory_enabled=False,
                on_compacting=on_compacting,
            )

        self.assertIsNotNone(
            self.repository.get_conversation_compaction(self.conversation.id)
        )
        self.assertTrue(
            any(
                "Compact summary of the earlier turns." in message["content"]
                for message in plan.messages
            )
        )
        # The UI hook fires exactly once, synchronously right before the
        # sub-agent call -- not once per assemble() attempt.
        self.assertEqual(compacting_calls, 1)

    async def test_does_not_compact_when_everything_fits(self) -> None:
        self._add_turn("Q1", "A1")
        self.repository.add_message(self.conversation.id, "user", "Latest question")
        messages = self.repository.list_messages(self.conversation.id)
        conversation = self.repository.get_conversation(self.conversation.id)
        assert conversation is not None
        profile = ModelProfile(
            key="big", label="Big", base_url="http://x/v1", model="test-model"
        )  # default 32,768-token window -- plenty of room for this tiny history
        calls: list[str] = []

        async def fake_run(arguments: dict) -> str:
            calls.append(arguments["task"])
            return "unused"

        compacting_calls = 0

        def on_compacting() -> None:
            nonlocal compacting_calls
            compacting_calls += 1

        with mock.patch.object(context, "subagent", SimpleNamespace(run=fake_run)):
            await prepare_conversation_context(
                conversation,
                profile,
                messages,
                "Latest question",
                thinking_enabled=False,
                memory_enabled=False,
                on_compacting=on_compacting,
            )

        self.assertEqual(calls, [])
        self.assertIsNone(self.repository.get_conversation_compaction(self.conversation.id))
        self.assertEqual(compacting_calls, 0)


if __name__ == "__main__":
    unittest.main()

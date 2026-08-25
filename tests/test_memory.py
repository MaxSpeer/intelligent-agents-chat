"""Project-memory retrieval, isolation, and lifecycle tests."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from intelligent_agents_chat.database import ChatRepository
from intelligent_agents_chat.memory import ProjectMemoryStore, _fts_query
from intelligent_agents_chat.retrieval import RetrievalQuery


class ProjectMemoryStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "chats.sqlite3"
        self.repository = ChatRepository(self.database_path)
        self.repository.initialize()
        self.memory = ProjectMemoryStore(self.database_path)
        self.research = self.repository.create_project("Research")
        self.private = self.repository.create_project("Private")

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _add_turn(self, conversation_id: str, user: str, assistant: str) -> None:
        self.repository.add_message(conversation_id, "user", user)
        self.repository.add_message(
            conversation_id,
            "assistant",
            assistant,
            model_profile="base",
        )

    def test_retrieval_uses_other_chats_in_the_same_project_only(self) -> None:
        source = self.repository.create_conversation(
            "base", project_id=self.research.id, title="Orchid notes"
        )
        target = self.repository.create_conversation(
            "base", project_id=self.research.id, title="Current question"
        )
        other_project = self.repository.create_conversation(
            "base", project_id=self.private.id, title="Private orchids"
        )
        self._add_turn(source.id, "The orchid launch code is amber.", "Noted for the project.")
        self._add_turn(
            other_project.id,
            "The private orchid launch code is violet.",
            "Keep it private.",
        )
        self.memory.rebuild_project(self.research.id)
        self.memory.rebuild_project(self.private.id)

        results = self.memory.retrieve(
            RetrievalQuery(
                project_id=self.research.id,
                text="What is the orchid launch code?",
                exclude_conversation_id=target.id,
            )
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].source_conversation_id, source.id)
        self.assertIn("amber", results[0].text)
        self.assertNotIn("violet", results[0].text)

        current_chat_results = self.memory.retrieve(
            RetrievalQuery(
                project_id=self.research.id,
                text="orchid",
                exclude_conversation_id=source.id,
            )
        )
        self.assertEqual(current_chat_results, [])

    def test_rebuild_is_idempotent_and_preserves_disabled_entries(self) -> None:
        source = self.repository.create_conversation(
            "base", project_id=self.research.id, title="Decisions"
        )
        self._add_turn(source.id, "Choose SQLite for memory.", "Decision recorded.")

        self.assertEqual(self.memory.rebuild_conversation(source.id), 1)
        first = self.memory.list_entries(self.research.id)
        self.assertEqual(len(first), 1)
        self.assertTrue(self.memory.set_enabled(self.research.id, first[0].id, False))

        self.assertEqual(self.memory.rebuild_conversation(source.id), 1)
        second = self.memory.list_entries(self.research.id)

        self.assertEqual([entry.id for entry in second], [first[0].id])
        self.assertFalse(second[0].enabled)
        self.assertEqual(
            self.memory.retrieve(RetrievalQuery(project_id=self.research.id, text="SQLite memory")),
            [],
        )

        self.assertTrue(self.memory.set_enabled(self.research.id, first[0].id, True))
        self.assertEqual(
            len(
                self.memory.retrieve(
                    RetrievalQuery(project_id=self.research.id, text="SQLite memory")
                )
            ),
            1,
        )

    def test_rebuilding_an_unchanged_turn_does_not_bump_its_updated_at(self) -> None:
        """rebuild_conversation reprocesses the whole conversation every call
        (see the module docstring), so most turns are unchanged on any given
        rebuild. Those must be true no-ops -- otherwise updated_at (used to
        order both list_entries and retrieve's tie-break) would mean "this
        conversation had any activity recently" instead of "this chunk's
        content actually changed", and every rebuild would needlessly delete
        and reinsert unchanged rows in the FTS index.
        """
        source = self.repository.create_conversation(
            "base", project_id=self.research.id, title="Decisions"
        )
        self._add_turn(source.id, "Choose SQLite for memory.", "Decision recorded.")
        self.memory.rebuild_conversation(source.id)
        first_updated_at = self.memory.list_entries(self.research.id)[0].updated_at

        # A second, unrelated turn triggers another full rebuild of the same
        # conversation -- the first turn's chunk is unchanged by it.
        self._add_turn(source.id, "Also use FTS5 for search.", "Also recorded.")
        self.memory.rebuild_conversation(source.id)
        entries = {entry.content: entry for entry in self.memory.list_entries(self.research.id)}

        self.assertEqual(len(entries), 2)
        self.assertEqual(
            entries["Conversation: Decisions\nUser: Choose SQLite for memory.\n"
            "Assistant: Decision recorded."].updated_at,
            first_updated_at,
        )

    def test_deleting_source_conversation_removes_derived_memory(self) -> None:
        source = self.repository.create_conversation(
            "base", project_id=self.research.id, title="Temporary"
        )
        self._add_turn(source.id, "Remember the zephyr protocol.", "Remembered.")
        self.memory.rebuild_conversation(source.id)
        self.assertEqual(
            len(self.memory.retrieve(RetrievalQuery(self.research.id, "zephyr protocol"))),
            1,
        )

        self.assertTrue(self.repository.delete_conversation(source.id))

        self.assertEqual(self.memory.list_entries(self.research.id), [])
        self.assertEqual(
            self.memory.retrieve(RetrievalQuery(self.research.id, "zephyr protocol")),
            [],
        )

    def test_results_are_ranked_and_punctuation_cannot_change_fts_syntax(self) -> None:
        strongest = self.repository.create_conversation(
            "base", project_id=self.research.id, title="Quokka details"
        )
        weaker = self.repository.create_conversation(
            "base", project_id=self.research.id, title="Other notes"
        )
        self._add_turn(
            strongest.id,
            "Quokka quokka quokka habitat protocol",
            "The quokka protocol concerns habitat.",
        )
        self._add_turn(weaker.id, "One quokka mention.", "No further detail.")
        self.memory.rebuild_project(self.research.id)

        results = self.memory.retrieve(
            RetrievalQuery(
                project_id=self.research.id,
                text='quokka " OR * habitat:',
            )
        )

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].source_conversation_id, strongest.id)
        self.assertGreaterEqual(results[0].score, results[1].score)

    def test_a_decimal_number_in_the_query_still_finds_its_matching_entry(self) -> None:
        source = self.repository.create_conversation(
            "base", project_id=self.research.id, title="Berlin population"
        )
        self._add_turn(
            source.id,
            "What was Berlin's population in 2023?",
            "Berlin's population in 2023 was 3.878 million.",
        )
        self.memory.rebuild_conversation(source.id)

        results = self.memory.retrieve(
            RetrievalQuery(project_id=self.research.id, text="the 3.878 million figure")
        )

        self.assertEqual(len(results), 1)
        self.assertIn("3.878", results[0].text)


class FtsQueryTokenizationTests(unittest.TestCase):
    def test_decimal_numbers_are_kept_intact_not_split_on_the_dot(self) -> None:
        query = _fts_query("What was the 3.878 million figure for Berlin in 2023?")

        self.assertIn('"3.878"', query)
        self.assertNotIn('"3"', query)
        self.assertNotIn('"878"', query)


class ToolCallingTurnChunkingTests(unittest.TestCase):
    """A turn with tool-call rounds is still one message.role='user' row
    followed by several message.role='assistant' rows (tool result rows are
    excluded by rebuild_conversation's own query, so they never reach
    chunking) -- these tests build that shape directly through the
    repository, the same way app.py's flush_pending/ToolResultEvent handling
    does.
    """

    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "chats.sqlite3"
        self.repository = ChatRepository(self.database_path)
        self.repository.initialize()
        self.memory = ProjectMemoryStore(self.database_path)
        self.project = self.repository.create_project("Research")
        self.conversation = self.repository.create_conversation(
            "base", project_id=self.project.id, title="Weather lookup"
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_tool_call_rounds_do_not_fragment_the_turn_into_several_entries(self) -> None:
        user_message = self.repository.add_message(
            self.conversation.id, "user", "What's the weather in Kyoto right now?"
        )
        self.repository.add_message(
            self.conversation.id,
            "assistant",
            "Let me check that for you.",
            model_profile="base",
            tool_calls=[{"id": "call_1", "name": "websearch", "arguments": '{"query": "..."}'}],
        )
        self.repository.add_message(self.conversation.id, "tool", "22C, clear", tool_call_id="call_1")
        final_answer = self.repository.add_message(
            self.conversation.id,
            "assistant",
            "It's 22C and clear in Kyoto right now.",
            model_profile="base",
        )

        entry_count = self.memory.rebuild_conversation(self.conversation.id)
        entries = self.memory.list_entries(self.project.id)

        self.assertEqual(entry_count, 1)
        self.assertEqual(len(entries), 1)
        entry = entries[0]
        self.assertEqual(entry.source_message_start_id, user_message.id)
        self.assertEqual(entry.source_message_end_id, final_answer.id)
        self.assertIn("What's the weather in Kyoto right now?", entry.content)
        self.assertIn("It's 22C and clear in Kyoto right now.", entry.content)
        # The narration before the tool call is trace, not the answer -- the
        # UI itself never shows it as the answer bubble, so memory shouldn't
        # either.
        self.assertNotIn("Let me check that for you.", entry.content)

    def test_narration_before_a_tool_call_is_not_indexed_or_retrievable(self) -> None:
        self.repository.add_message(
            self.conversation.id, "user", "What's the weather in Kyoto right now?"
        )
        self.repository.add_message(
            self.conversation.id,
            "assistant",
            "Let me consult a falconer about this.",
            model_profile="base",
            tool_calls=[{"id": "call_1", "name": "websearch", "arguments": '{"query": "..."}'}],
        )
        self.repository.add_message(self.conversation.id, "tool", "22C, clear", tool_call_id="call_1")
        self.repository.add_message(
            self.conversation.id,
            "assistant",
            "It's 22C and clear in Kyoto right now.",
            model_profile="base",
        )
        self.memory.rebuild_conversation(self.conversation.id)

        matches = self.memory.retrieve(
            RetrievalQuery(project_id=self.project.id, text="falconer")
        )

        self.assertEqual(matches, [])

    def test_a_turn_without_an_answer_yet_produces_no_memory_entry(self) -> None:
        self.repository.add_message(
            self.conversation.id, "user", "What's the weather in Kyoto right now?"
        )

        entry_count = self.memory.rebuild_conversation(self.conversation.id)

        self.assertEqual(entry_count, 0)
        self.assertEqual(self.memory.list_entries(self.project.id), [])

    def test_a_tool_only_round_with_no_narration_still_produces_no_entry_until_answered(
        self,
    ) -> None:
        self.repository.add_message(
            self.conversation.id, "user", "What's the weather in Kyoto right now?"
        )
        self.repository.add_message(
            self.conversation.id,
            "assistant",
            "",
            model_profile="base",
            tool_calls=[{"id": "call_1", "name": "websearch", "arguments": '{"query": "..."}'}],
        )
        self.repository.add_message(self.conversation.id, "tool", "22C, clear", tool_call_id="call_1")

        entry_count = self.memory.rebuild_conversation(self.conversation.id)
        self.assertEqual(entry_count, 0)

        final_answer = self.repository.add_message(
            self.conversation.id,
            "assistant",
            "It's 22C and clear in Kyoto right now.",
            model_profile="base",
        )
        entry_count = self.memory.rebuild_conversation(self.conversation.id)
        entries = self.memory.list_entries(self.project.id)

        self.assertEqual(entry_count, 1)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].source_message_end_id, final_answer.id)


if __name__ == "__main__":
    unittest.main()

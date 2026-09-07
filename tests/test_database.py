"""Focused persistence tests for the SQLite chat repository."""

from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest

from intelligent_agents_chat.database import (
    ContextRunInput,
    ContextSourceInput,
    DEFAULT_CONVERSATION_TITLE,
    DEFAULT_DATABASE_PATH,
    DEFAULT_PROJECT_ID,
    ChatRepository,
)


class ChatRepositoryTests(unittest.TestCase):
    def test_default_database_path_is_under_the_project_data_directory(self) -> None:
        self.assertEqual(ChatRepository().database_path, DEFAULT_DATABASE_PATH)
        self.assertEqual(DEFAULT_DATABASE_PATH.name, "chats.sqlite3")

    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "chats.sqlite3"
        self.repository = ChatRepository(self.database_path)
        self.repository.initialize()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_initialize_is_idempotent_and_creates_default_project(self) -> None:
        self.repository.initialize()

        project = self.repository.get_project()
        self.assertIsNotNone(project)
        assert project is not None
        self.assertEqual(project.id, DEFAULT_PROJECT_ID)
        self.assertEqual(project.name, "General")

        with sqlite3.connect(self.database_path) as connection:
            mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
        self.assertEqual(mode, "wal")

    def test_conversation_and_messages_survive_a_new_repository_instance(self) -> None:
        conversation = self.repository.create_conversation()
        self.repository.add_message(conversation.id, "user", "Hello")
        assistant = self.repository.add_message(
            conversation.id,
            "assistant",
            "Hello back",
            model_profile="base",
        )

        reopened = ChatRepository(self.database_path)
        reopened.initialize()
        loaded_conversation = reopened.get_conversation(conversation.id)
        messages = reopened.list_messages(conversation.id)

        self.assertIsNotNone(loaded_conversation)
        assert loaded_conversation is not None
        self.assertEqual(loaded_conversation.title, DEFAULT_CONVERSATION_TITLE)
        self.assertEqual([message.role for message in messages], ["user", "assistant"])
        self.assertEqual(messages[1].id, assistant.id)
        self.assertEqual(messages[1].model_profile, "base")

    def test_projects_encapsulate_their_conversations(self) -> None:
        research = self.repository.create_project("Research")
        teaching = self.repository.create_project("Teaching")
        research_chat = self.repository.create_conversation(
            project_id=research.id,
            title="Paper notes",
        )
        teaching_chat = self.repository.create_conversation(
            project_id=teaching.id,
            title="Exercise sheet",
        )

        projects = self.repository.list_projects()

        self.assertEqual(projects[0].id, DEFAULT_PROJECT_ID)
        self.assertEqual([project.name for project in projects[1:]], ["Research", "Teaching"])
        self.assertEqual(
            [conversation.id for conversation in self.repository.list_conversations(research.id)],
            [research_chat.id],
        )
        self.assertEqual(
            [conversation.id for conversation in self.repository.list_conversations(teaching.id)],
            [teaching_chat.id],
        )
        self.assertEqual(self.repository.list_conversations(DEFAULT_PROJECT_ID), [])

    def test_project_names_are_normalized_and_must_be_unique(self) -> None:
        project = self.repository.create_project("  Intelligent   Agents  ")

        self.assertEqual(project.name, "Intelligent Agents")
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.repository.create_project("intelligent agents")
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            self.repository.create_project("   ")

    def test_rename_is_persisted(self) -> None:
        conversation = self.repository.create_conversation()

        self.assertTrue(self.repository.rename_conversation(conversation.id, "Planning"))

        updated = self.repository.get_conversation(conversation.id)
        self.assertIsNotNone(updated)
        assert updated is not None
        self.assertEqual(updated.title, "Planning")

    def test_app_settings_are_absent_until_first_set(self) -> None:
        self.assertIsNone(self.repository.get_app_settings())

    def test_app_settings_are_a_single_row_shared_by_the_whole_app(self) -> None:
        """Not per-conversation -- one row, upserted in place (see
        set_app_settings), so switching or creating a chat can't reset it.
        """
        self.repository.set_app_settings(
            model_profile="base", thinking_enabled=False, memory_enabled=False
        )
        updated = self.repository.set_app_settings(
            model_profile="tuned", thinking_enabled=True, memory_enabled=True
        )

        self.assertEqual(updated.model_profile, "tuned")
        self.assertTrue(updated.thinking_enabled)
        self.assertTrue(updated.memory_enabled)

        reopened = ChatRepository(self.database_path).get_app_settings()
        self.assertIsNotNone(reopened)
        assert reopened is not None
        self.assertEqual(reopened.model_profile, "tuned")
        self.assertTrue(reopened.thinking_enabled)
        self.assertTrue(reopened.memory_enabled)

    def test_context_source_provenance_is_persisted_and_cascades_with_message(self) -> None:
        source = self.repository.create_conversation(title="Architecture")
        target = self.repository.create_conversation(title="Implementation")
        assistant = self.repository.add_message(
            target.id,
            "assistant",
            "Use SQLite FTS5.",
            model_profile="base",
        )
        self.repository.add_message_context_sources(
            assistant.id,
            [
                ContextSourceInput(
                    source_title=source.title,
                    source_locator="messages 1-2",
                    source_excerpt="The database decision was SQLite FTS5.",
                    rank=1,
                    token_estimate=20,
                )
            ],
        )

        loaded = self.repository.list_message_context_sources(assistant.id)

        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].source_title, "Architecture")
        self.assertEqual(loaded[0].source_locator, "messages 1-2")
        self.assertEqual(loaded[0].source_excerpt, "The database decision was SQLite FTS5.")
        self.assertEqual(loaded[0].rank, 1)

        self.assertTrue(self.repository.delete_conversation(target.id))
        self.assertEqual(self.repository.list_message_context_sources(assistant.id), [])

    def test_context_run_is_persisted_and_cascades_with_its_message(self) -> None:
        conversation = self.repository.create_conversation()
        assistant = self.repository.add_message(
            conversation.id,
            "assistant",
            "The answer is 2.",
            model_profile="base",
        )

        self.repository.add_message_context_run(
            assistant.id,
            ContextRunInput(
                context_window_tokens=32_768,
                input_budget_tokens=31_744,
                estimated_input_tokens=512,
                real_prompt_tokens=498,
                real_completion_tokens=12,
                real_peak_total_tokens=510,
            ),
        )

        loaded = self.repository.get_message_context_run(assistant.id)

        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertEqual(loaded.assistant_message_id, assistant.id)
        self.assertEqual(loaded.context_window_tokens, 32_768)
        self.assertEqual(loaded.input_budget_tokens, 31_744)
        self.assertEqual(loaded.estimated_input_tokens, 512)
        self.assertEqual(loaded.real_prompt_tokens, 498)
        self.assertEqual(loaded.real_completion_tokens, 12)
        self.assertEqual(loaded.real_peak_total_tokens, 510)

        self.assertTrue(self.repository.delete_conversation(conversation.id))
        self.assertIsNone(self.repository.get_message_context_run(assistant.id))

    def test_context_run_real_token_fields_are_optional(self) -> None:
        """The server not reporting usage (see llm.py's stream_options
        caveat) shouldn't block persisting the estimated numbers.
        """
        conversation = self.repository.create_conversation()
        assistant = self.repository.add_message(
            conversation.id, "assistant", "The answer is 2.", model_profile="base"
        )

        self.repository.add_message_context_run(
            assistant.id,
            ContextRunInput(
                context_window_tokens=32_768,
                input_budget_tokens=31_744,
                estimated_input_tokens=512,
                real_prompt_tokens=None,
                real_completion_tokens=None,
                real_peak_total_tokens=None,
            ),
        )

        loaded = self.repository.get_message_context_run(assistant.id)

        assert loaded is not None
        self.assertIsNone(loaded.real_prompt_tokens)
        self.assertIsNone(loaded.real_completion_tokens)
        self.assertIsNone(loaded.real_peak_total_tokens)

    def test_get_message_context_run_returns_none_when_absent(self) -> None:
        conversation = self.repository.create_conversation()
        assistant = self.repository.add_message(
            conversation.id, "assistant", "The answer is 2.", model_profile="base"
        )

        self.assertIsNone(self.repository.get_message_context_run(assistant.id))

    def test_conversation_compaction_is_persisted_and_extended_in_place(self) -> None:
        conversation = self.repository.create_conversation()
        message = self.repository.add_message(conversation.id, "user", "Hello")

        self.repository.set_conversation_compaction(
            conversation.id,
            compacted_through_message_id=message.id,
            summary="First summary.",
        )
        first = self.repository.get_conversation_compaction(conversation.id)
        assert first is not None
        self.assertEqual(first.conversation_id, conversation.id)
        self.assertEqual(first.compacted_through_message_id, message.id)
        self.assertEqual(first.summary, "First summary.")

        later_message = self.repository.add_message(conversation.id, "assistant", "Hi!")
        self.repository.set_conversation_compaction(
            conversation.id,
            compacted_through_message_id=later_message.id,
            summary="Extended summary.",
        )
        second = self.repository.get_conversation_compaction(conversation.id)

        assert second is not None
        # One row per conversation, updated in place -- not a second row.
        self.assertEqual(second.compacted_through_message_id, later_message.id)
        self.assertEqual(second.summary, "Extended summary.")
        self.assertEqual(second.created_at, first.created_at)
        self.assertGreaterEqual(second.updated_at, first.updated_at)

    def test_get_conversation_compaction_returns_none_when_absent(self) -> None:
        conversation = self.repository.create_conversation()

        self.assertIsNone(self.repository.get_conversation_compaction(conversation.id))

    def test_conversation_compaction_cascades_with_its_conversation(self) -> None:
        conversation = self.repository.create_conversation()
        message = self.repository.add_message(conversation.id, "user", "Hello")
        self.repository.set_conversation_compaction(
            conversation.id, compacted_through_message_id=message.id, summary="Summary."
        )

        self.assertTrue(self.repository.delete_conversation(conversation.id))

        self.assertIsNone(self.repository.get_conversation_compaction(conversation.id))

    def test_deleting_a_conversation_cascades_to_messages(self) -> None:
        conversation = self.repository.create_conversation()
        message = self.repository.add_message(conversation.id, "user", "Temporary")

        self.assertTrue(self.repository.delete_conversation(conversation.id))
        self.assertIsNone(self.repository.get_conversation(conversation.id))
        self.assertIsNone(self.repository.get_message(message.id))

    def test_conversations_are_sorted_by_latest_activity(self) -> None:
        older = self.repository.create_conversation(title="Older")
        newer = self.repository.create_conversation(title="Newer")
        self.repository.add_message(older.id, "user", "Touch the older chat")

        conversations = self.repository.list_conversations()

        self.assertEqual(conversations[0].id, older.id)
        self.assertEqual(conversations[1].id, newer.id)

    def test_assistant_message_can_carry_tool_calls_with_empty_content(self) -> None:
        conversation = self.repository.create_conversation()
        tool_calls = [{"id": "call_1", "name": "calculator", "arguments": '{"expression": "1+1"}'}]

        message = self.repository.add_message(
            conversation.id,
            "assistant",
            "",
            model_profile="base",
            tool_calls=tool_calls,
        )

        reloaded = self.repository.get_message(message.id)
        self.assertIsNotNone(reloaded)
        assert reloaded is not None
        self.assertEqual(reloaded.content, "")
        self.assertEqual(reloaded.tool_calls, tuple(tool_calls))

    def test_tool_message_stores_its_tool_call_id(self) -> None:
        conversation = self.repository.create_conversation()

        message = self.repository.add_message(
            conversation.id,
            "tool",
            "42",
            tool_call_id="call_1",
        )

        reloaded = self.repository.get_message(message.id)
        self.assertIsNotNone(reloaded)
        assert reloaded is not None
        self.assertEqual(reloaded.role, "tool")
        self.assertEqual(reloaded.tool_call_id, "call_1")
        self.assertIsNone(reloaded.tool_calls)

    def test_empty_content_without_tool_calls_is_still_rejected(self) -> None:
        conversation = self.repository.create_conversation()

        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            self.repository.add_message(conversation.id, "assistant", "   ")

    def test_messages_without_tool_data_round_trip_as_none(self) -> None:
        conversation = self.repository.create_conversation()
        message = self.repository.add_message(conversation.id, "user", "Hello")

        reloaded = self.repository.get_message(message.id)

        assert reloaded is not None
        self.assertIsNone(reloaded.tool_calls)
        self.assertIsNone(reloaded.tool_call_id)
        self.assertIsNone(reloaded.reasoning)

    def test_assistant_message_can_carry_reasoning_alongside_content(self) -> None:
        conversation = self.repository.create_conversation()

        message = self.repository.add_message(
            conversation.id,
            "assistant",
            "The answer is 2.",
            model_profile="base",
            reasoning="1 + 1 is a basic addition, so the answer is 2.",
        )

        reloaded = self.repository.get_message(message.id)
        self.assertIsNotNone(reloaded)
        assert reloaded is not None
        self.assertEqual(reloaded.content, "The answer is 2.")
        self.assertEqual(reloaded.reasoning, "1 + 1 is a basic addition, so the answer is 2.")

    def test_reasoning_alone_is_enough_to_satisfy_the_non_empty_check(self) -> None:
        conversation = self.repository.create_conversation()

        message = self.repository.add_message(
            conversation.id,
            "assistant",
            "",
            model_profile="base",
            reasoning="Thinking about tool calls, not a final answer yet.",
            tool_calls=[{"id": "call_1", "name": "calculator", "arguments": "{}"}],
        )

        reloaded = self.repository.get_message(message.id)
        assert reloaded is not None
        self.assertEqual(reloaded.content, "")
        self.assertEqual(reloaded.reasoning, "Thinking about tool calls, not a final answer yet.")


if __name__ == "__main__":
    unittest.main()

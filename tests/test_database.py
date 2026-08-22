"""Focused persistence tests for the SQLite chat repository."""

from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest

from intelligent_agents_chat.database import (
    AdaptiveRAGRoundRecord,
    AdaptiveRAGRunInput,
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
        conversation = self.repository.create_conversation("base")
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
        self.assertFalse(loaded_conversation.thinking_enabled)
        self.assertFalse(loaded_conversation.memory_enabled)
        self.assertFalse(loaded_conversation.rag_enabled)
        self.assertEqual([message.role for message in messages], ["user", "assistant"])
        self.assertEqual(messages[1].id, assistant.id)
        self.assertEqual(messages[1].model_profile, "base")

    def test_projects_encapsulate_their_conversations(self) -> None:
        research = self.repository.create_project("Research")
        teaching = self.repository.create_project("Teaching")
        research_chat = self.repository.create_conversation(
            "base",
            project_id=research.id,
            title="Paper notes",
        )
        teaching_chat = self.repository.create_conversation(
            "base",
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

    def test_rename_and_model_selection_are_persisted(self) -> None:
        conversation = self.repository.create_conversation("base")

        self.assertTrue(self.repository.rename_conversation(conversation.id, "Planning"))
        self.assertTrue(self.repository.set_model_profile(conversation.id, "tuned"))
        self.assertTrue(self.repository.set_thinking_enabled(conversation.id, True))
        self.assertTrue(self.repository.set_memory_enabled(conversation.id, True))
        self.assertTrue(self.repository.set_rag_enabled(conversation.id, True))

        updated = self.repository.get_conversation(conversation.id)
        self.assertIsNotNone(updated)
        assert updated is not None
        self.assertEqual(updated.title, "Planning")
        self.assertEqual(updated.model_profile, "tuned")
        self.assertTrue(updated.thinking_enabled)
        self.assertTrue(updated.memory_enabled)
        self.assertTrue(updated.rag_enabled)

    def test_new_conversation_can_start_with_thinking_enabled(self) -> None:
        conversation = self.repository.create_conversation(
            "thinking-model",
            thinking_enabled=True,
        )

        reopened = ChatRepository(self.database_path).get_conversation(conversation.id)

        self.assertIsNotNone(reopened)
        assert reopened is not None
        self.assertTrue(reopened.thinking_enabled)

    def test_new_conversation_can_start_with_memory_enabled(self) -> None:
        conversation = self.repository.create_conversation(
            "base",
            memory_enabled=True,
        )

        reopened = ChatRepository(self.database_path).get_conversation(conversation.id)

        self.assertIsNotNone(reopened)
        assert reopened is not None
        self.assertTrue(reopened.memory_enabled)

    def test_new_conversation_can_start_with_rag_enabled(self) -> None:
        conversation = self.repository.create_conversation(
            "base",
            rag_enabled=True,
        )

        reopened = ChatRepository(self.database_path).get_conversation(conversation.id)

        self.assertIsNotNone(reopened)
        assert reopened is not None
        self.assertTrue(reopened.rag_enabled)

    def test_context_source_provenance_is_persisted_and_cascades_with_message(self) -> None:
        source = self.repository.create_conversation("base", title="Architecture")
        target = self.repository.create_conversation("base", title="Implementation")
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
                    source_kind="project_memory",
                    source_id="42",
                    source_project_id=DEFAULT_PROJECT_ID,
                    source_conversation_id=source.id,
                    source_title=source.title,
                    source_locator="messages 1-2",
                    source_excerpt="The database decision was SQLite FTS5.",
                    rank=1,
                    score=0.75,
                    token_estimate=20,
                )
            ],
        )

        loaded = self.repository.list_message_context_sources(assistant.id)

        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].source_kind, "project_memory")
        self.assertEqual(loaded[0].source_conversation_id, source.id)
        self.assertEqual(loaded[0].source_title, "Architecture")
        self.assertEqual(loaded[0].source_excerpt, "The database decision was SQLite FTS5.")
        self.assertEqual(loaded[0].rank, 1)

        self.assertTrue(self.repository.delete_conversation(target.id))
        self.assertEqual(self.repository.list_message_context_sources(assistant.id), [])

    def test_adaptive_rag_trace_is_persisted_and_cascades_with_message(self) -> None:
        conversation = self.repository.create_conversation("base", rag_enabled=True)
        assistant = self.repository.add_message(
            conversation.id,
            "assistant",
            "There are 7 units.",
            model_profile="base",
        )
        self.repository.add_message_adaptive_rag_run(
            assistant.id,
            AdaptiveRAGRunInput(
                retrieval_needed=True,
                judge_reason="The question requests project inventory data.",
                rounds=(
                    AdaptiveRAGRoundRecord(
                        query="Mondkeks inventory",
                        result_count=3,
                        new_result_count=3,
                        evidence_sufficient=True,
                        assessment_reason="The stock figure is present.",
                        missing_information="",
                    ),
                ),
                evidence_sufficient=True,
                summary="The source reports 7 units.",
                fallback_reasons=(),
                duration_ms=12.5,
            ),
        )

        loaded = self.repository.get_message_adaptive_rag_run(assistant.id)

        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertTrue(loaded.retrieval_needed)
        self.assertEqual(loaded.rounds[0].query, "Mondkeks inventory")
        self.assertTrue(loaded.rounds[0].evidence_sufficient)
        self.assertEqual(loaded.summary, "The source reports 7 units.")
        self.assertEqual(loaded.duration_ms, 12.5)

        self.assertTrue(self.repository.delete_conversation(conversation.id))
        self.assertIsNone(self.repository.get_message_adaptive_rag_run(assistant.id))

    def test_initialize_migrates_a_database_without_memory_columns(self) -> None:
        legacy_path = Path(self.temporary_directory.name) / "legacy.sqlite3"
        with sqlite3.connect(legacy_path) as connection:
            connection.executescript(
                """
                CREATE TABLE projects (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE conversations (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    title TEXT NOT NULL,
                    model_profile TEXT NOT NULL,
                    thinking_enabled INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL
                        REFERENCES conversations(id) ON DELETE CASCADE,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    model_profile TEXT,
                    created_at TEXT NOT NULL
                );
                INSERT INTO projects VALUES (
                    'default', 'General', '2026-01-01T00:00:00+00:00',
                    '2026-01-01T00:00:00+00:00'
                );
                INSERT INTO conversations VALUES (
                    'legacy', 'default', 'Old chat', 'base', 0,
                    '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00'
                );
                """
            )

        migrated = ChatRepository(legacy_path)
        migrated.initialize()

        conversation = migrated.get_conversation("legacy")
        self.assertIsNotNone(conversation)
        assert conversation is not None
        self.assertFalse(conversation.memory_enabled)
        self.assertFalse(conversation.rag_enabled)
        with sqlite3.connect(legacy_path) as connection:
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 5)

    def test_deleting_a_conversation_cascades_to_messages(self) -> None:
        conversation = self.repository.create_conversation("base")
        message = self.repository.add_message(conversation.id, "user", "Temporary")

        self.assertTrue(self.repository.delete_conversation(conversation.id))
        self.assertIsNone(self.repository.get_conversation(conversation.id))
        self.assertIsNone(self.repository.get_message(message.id))

    def test_conversations_are_sorted_by_latest_activity(self) -> None:
        older = self.repository.create_conversation("base", title="Older")
        newer = self.repository.create_conversation("base", title="Newer")
        self.repository.add_message(older.id, "user", "Touch the older chat")

        conversations = self.repository.list_conversations()

        self.assertEqual(conversations[0].id, older.id)
        self.assertEqual(conversations[1].id, newer.id)


if __name__ == "__main__":
    unittest.main()

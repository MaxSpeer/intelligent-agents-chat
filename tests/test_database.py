"""Focused persistence tests for the SQLite chat repository."""

from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest

from intelligent_agents_chat.database import (
    DEFAULT_CONVERSATION_TITLE,
    DEFAULT_PROJECT_ID,
    ChatRepository,
)


class ChatRepositoryTests(unittest.TestCase):
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
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
        self.assertEqual(version, 1)
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
        self.assertEqual([message.role for message in messages], ["user", "assistant"])
        self.assertEqual(messages[1].id, assistant.id)
        self.assertEqual(messages[1].model_profile, "base")

    def test_rename_and_model_selection_are_persisted(self) -> None:
        conversation = self.repository.create_conversation("base")

        self.assertTrue(self.repository.rename_conversation(conversation.id, "Planning"))
        self.assertTrue(self.repository.set_model_profile(conversation.id, "tuned"))

        updated = self.repository.get_conversation(conversation.id)
        self.assertIsNotNone(updated)
        assert updated is not None
        self.assertEqual(updated.title, "Planning")
        self.assertEqual(updated.model_profile, "tuned")

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

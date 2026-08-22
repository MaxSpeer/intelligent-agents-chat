"""Project-memory retrieval, isolation, and lifecycle tests."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from intelligent_agents_chat.database import ChatRepository
from intelligent_agents_chat.memory import ProjectMemoryStore
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


if __name__ == "__main__":
    unittest.main()

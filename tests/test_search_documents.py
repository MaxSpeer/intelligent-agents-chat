"""Tests for the search_documents tool.

Unlike the other tools, this one is a factory (build_tool(project_id)) since
project_id has to be bound by the server, not supplied by the model -- see
the module's own docstring. documents.rag_retriever is mocked out here (no
real SQLite/embedding work), the same way test_context.py mocks other
module-level attributes.
"""

import unittest
from unittest import mock

from intelligent_agents_chat import documents
from intelligent_agents_chat.documents import DocumentPassage
from intelligent_agents_chat.tools import search_documents


def _passage(*, title: str = "notes.txt") -> DocumentPassage:
    return DocumentPassage(
        text="The orchid launch code is amber.",
        title=title,
        locator="document",
    )


class SearchDocumentsToolTests(unittest.IsolatedAsyncioTestCase):
    def test_tool_schema_name_matches_the_registry_key(self) -> None:
        tool = search_documents.build_tool("project-a")
        self.assertEqual(tool.name, tool.schema["function"]["name"])

    async def test_missing_query_is_a_reported_error(self) -> None:
        tool = search_documents.build_tool("project-a")
        result = await tool.run({})
        self.assertTrue(result.startswith("Error:"))

    async def test_results_are_formatted_with_title_locator_and_text(self) -> None:
        fake_retriever = mock.AsyncMock()
        fake_retriever.retrieve.return_value = [_passage()]
        with mock.patch.object(documents, "rag_retriever", fake_retriever):
            tool = search_documents.build_tool("project-a")
            result = await tool.run({"query": "orchid launch code"})

        self.assertIn("notes.txt", result)
        self.assertIn("document", result)
        self.assertIn("The orchid launch code is amber.", result)

    async def test_no_results_is_a_plain_message_not_an_error(self) -> None:
        fake_retriever = mock.AsyncMock()
        fake_retriever.retrieve.return_value = []
        with mock.patch.object(documents, "rag_retriever", fake_retriever):
            tool = search_documents.build_tool("project-a")
            result = await tool.run({"query": "nothing matches this"})

        self.assertFalse(result.startswith("Error:"))
        self.assertIn("No matching passages", result)

    async def test_project_id_is_bound_by_the_server_not_the_model(self) -> None:
        fake_retriever = mock.AsyncMock()
        fake_retriever.retrieve.return_value = []
        with mock.patch.object(documents, "rag_retriever", fake_retriever):
            tool = search_documents.build_tool("project-a")
            # The model's arguments have no project_id key at all -- the
            # schema doesn't expose one (see _SCHEMA) -- yet retrieval is
            # still scoped to the project build_tool was called with.
            await tool.run({"query": "anything", "project_id": "some-other-project"})

        self.assertEqual(fake_retriever.retrieve.call_args.kwargs["project_id"], "project-a")

    async def test_top_k_is_used_and_clamped_to_the_configured_maximum(self) -> None:
        fake_retriever = mock.AsyncMock()
        fake_retriever.retrieve.return_value = []
        with mock.patch.object(documents, "rag_retriever", fake_retriever):
            tool = search_documents.build_tool("project-a")
            await tool.run({"query": "orchids", "top_k": 999})

        self.assertEqual(
            fake_retriever.retrieve.call_args.kwargs["limit"], search_documents.MAX_TOP_K
        )

    async def test_missing_top_k_falls_back_to_the_default(self) -> None:
        fake_retriever = mock.AsyncMock()
        fake_retriever.retrieve.return_value = []
        with mock.patch.object(documents, "rag_retriever", fake_retriever):
            tool = search_documents.build_tool("project-a")
            await tool.run({"query": "orchids"})

        self.assertEqual(
            fake_retriever.retrieve.call_args.kwargs["limit"], search_documents.DEFAULT_TOP_K
        )

    async def test_retrieval_failure_is_reported_not_raised(self) -> None:
        fake_retriever = mock.AsyncMock()
        fake_retriever.retrieve.side_effect = RuntimeError("index unavailable")
        with mock.patch.object(documents, "rag_retriever", fake_retriever):
            tool = search_documents.build_tool("project-a")
            result = await tool.run({"query": "orchids"})

        self.assertTrue(result.startswith("Error:"))
        self.assertIn("index unavailable", result)


if __name__ == "__main__":
    unittest.main()

"""Tests for the recall_tool_output tool -- fetching one earlier tool
result back in full by the id shown in its collapsed trace note (see
context.py's completion_messages).
"""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

from intelligent_agents_chat.database import ChatRepository
from intelligent_agents_chat.tools import recall_tool_output


class RecallToolOutputTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        database_path = Path(self.temporary_directory.name) / "chats.sqlite3"
        self.repository = ChatRepository(database_path)
        self.repository.initialize()
        # The tool holds its own module-level repository (see
        # recall_tool_output.py, same pattern as subagent.py's _gateway) --
        # swapped for this test's temp one, rather than passed in explicitly.
        self.enterContext(mock.patch.object(recall_tool_output, "_repository", self.repository))
        self.conversation = self.repository.create_conversation()
        self.repository.add_message(self.conversation.id, "user", "Fetch that page.")
        self.repository.add_message(
            self.conversation.id,
            "assistant",
            "",
            tool_calls=[{"id": "call_1", "name": "web_fetch", "arguments": "{}"}],
        )
        self.tool_result = self.repository.add_message(
            self.conversation.id,
            "tool",
            "The full page content, much longer than the trace summary.",
            tool_call_id="call_1",
        )

    async def test_returns_the_full_stored_result_by_id(self) -> None:
        tool = recall_tool_output.build_tool(self.conversation.id)

        result = await tool.run({"tool_result_id": self.tool_result.id})

        self.assertEqual(result, "The full page content, much longer than the trace summary.")

    async def test_unknown_id_is_a_reported_error(self) -> None:
        tool = recall_tool_output.build_tool(self.conversation.id)

        result = await tool.run({"tool_result_id": self.tool_result.id + 1000})

        self.assertTrue(result.startswith("Error:"))

    async def test_a_message_id_from_a_different_conversation_is_rejected(self) -> None:
        other_conversation = self.repository.create_conversation()
        tool = recall_tool_output.build_tool(other_conversation.id)

        result = await tool.run({"tool_result_id": self.tool_result.id})

        self.assertTrue(result.startswith("Error:"))

    async def test_a_non_tool_message_id_is_rejected(self) -> None:
        """Only role="tool" messages are recallable -- not the user's own
        message or the assistant's, even though both have valid ids too.
        """
        messages = self.repository.list_messages(self.conversation.id)
        user_message = next(message for message in messages if message.role == "user")
        tool = recall_tool_output.build_tool(self.conversation.id)

        result = await tool.run({"tool_result_id": user_message.id})

        self.assertTrue(result.startswith("Error:"))

    async def test_a_non_integer_id_is_a_reported_error_not_a_crash(self) -> None:
        tool = recall_tool_output.build_tool(self.conversation.id)

        result = await tool.run({"tool_result_id": "not-a-number"})

        self.assertTrue(result.startswith("Error:"))

    async def test_missing_id_is_a_reported_error(self) -> None:
        tool = recall_tool_output.build_tool(self.conversation.id)

        result = await tool.run({})

        self.assertTrue(result.startswith("Error:"))

    def test_tool_schema_name_matches_the_tool_name(self) -> None:
        tool = recall_tool_output.build_tool(self.conversation.id)
        self.assertEqual(tool.name, tool.schema["function"]["name"])


if __name__ == "__main__":
    unittest.main()

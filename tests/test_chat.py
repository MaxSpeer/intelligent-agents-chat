"""Tests for reconstructing OpenAI-shape history from stored messages."""

from datetime import datetime, timezone
import unittest

from intelligent_agents_chat.chat import (
    SYSTEM_PROMPT,
    completion_messages,
    format_reasoning_entry,
    format_tool_call_entry,
    format_tool_result_entry,
)
from intelligent_agents_chat.database import Message


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
    def test_plain_messages_pass_through_with_the_system_prompt_prepended(self) -> None:
        messages = completion_messages([_message("user", "Hello")])

        self.assertEqual(
            messages,
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Hello"},
            ],
        )

    def test_assistant_tool_calls_are_reconstructed_in_the_real_api_shape(self) -> None:
        stored_call = {"id": "call_1", "name": "calculator", "arguments": '{"expression": "1+1"}'}
        messages = completion_messages([_message("assistant", "", tool_calls=(stored_call,))])

        self.assertEqual(
            messages[1],
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "calculator",
                            "arguments": '{"expression": "1+1"}',
                        },
                    }
                ],
            },
        )

    def test_tool_messages_are_reconstructed_with_their_tool_call_id(self) -> None:
        messages = completion_messages([_message("tool", "2", tool_call_id="call_1")])

        self.assertEqual(
            messages[1],
            {"role": "tool", "tool_call_id": "call_1", "content": "2"},
        )

    def test_assistant_content_survives_alongside_tool_calls(self) -> None:
        stored_call = {"id": "call_1", "name": "calculator", "arguments": "{}"}
        messages = completion_messages(
            [_message("assistant", "Let me check.", tool_calls=(stored_call,))]
        )

        self.assertEqual(messages[1]["content"], "Let me check.")

    def test_reasoning_is_never_sent_back_to_the_model(self) -> None:
        messages = completion_messages(
            [_message("assistant", "The answer is 2.", reasoning="1 + 1 = 2, obviously.")]
        )

        self.assertEqual(messages[1], {"role": "assistant", "content": "The answer is 2."})


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


if __name__ == "__main__":
    unittest.main()

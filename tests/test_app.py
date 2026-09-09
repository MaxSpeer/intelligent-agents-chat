"""Tests for chat UI formatting."""

import unittest

from intelligent_agents_chat.app import format_tool_call_entry, format_tool_result_entry


class FormatEntryTests(unittest.TestCase):
    """Tool-trace formatting for streaming and replay."""

    def test_format_tool_call_entry_includes_name_and_arguments(self) -> None:
        call = {"id": "call_1", "name": "calculator", "arguments": '{"expression": "1+1"}'}

        entry = format_tool_call_entry(call)

        self.assertIn("calculator", entry)
        self.assertIn('{"expression": "1+1"}', entry)

    def test_format_tool_call_entry_is_never_markdown(self) -> None:
        """Tool/argument names routinely contain underscores, which Markdown
        misreads as emphasis -- the call header is rendered as plain text
        (see app.py's _add_trace_step), so it must never contain Markdown
        emphasis syntax that could get misread once seen through the
        result's Markdown renderer.
        """
        call = {"id": "call_1", "name": "web_search", "arguments": "{}"}

        entry = format_tool_call_entry(call)

        self.assertNotIn("*", entry)

    def test_format_tool_result_entry_returns_the_result_as_is(self) -> None:
        entry = format_tool_result_entry("2")

        self.assertEqual(entry, "2")

    def test_format_tool_result_entry_still_running_has_no_result_yet(self) -> None:
        entry = format_tool_result_entry(None, still_running=True)

        self.assertIn("waiting", entry)

    def test_format_tool_result_entry_missing_result_reports_it_was_stopped(self) -> None:
        entry = format_tool_result_entry(None)

        self.assertIn("stopped", entry)


if __name__ == "__main__":
    unittest.main()

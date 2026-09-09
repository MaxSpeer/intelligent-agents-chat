"""Tests for the web_search tool, with `_search` mocked out (no network)."""

import unittest
from unittest import mock

from ddgs.exceptions import DDGSException

from intelligent_agents_chat.tools import websearch

_RESULTS = [
    {
        "title": "Stonehenge - Wikipedia",
        "href": "https://en.wikipedia.org/wiki/Stonehenge",
        "body": "Stonehenge is a prehistoric monument on Salisbury Plain in Wiltshire.",
    },
    {
        "title": "History of Stonehenge",
        "href": "https://www.english-heritage.org.uk/stonehenge/history/",
        "body": "Stonehenge appears to have been frequently visited in the Roman period.",
    },
]


class WebSearchToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_query_is_a_reported_error(self) -> None:
        result = await websearch.run({})
        self.assertTrue(result.startswith("Error:"))

    async def test_results_are_formatted_with_title_url_and_snippet(self) -> None:
        with mock.patch.object(websearch, "_search", return_value=_RESULTS):
            result = await websearch.run({"query": "Stonehenge"})

        self.assertIn("Stonehenge - Wikipedia", result)
        self.assertIn("https://en.wikipedia.org/wiki/Stonehenge", result)
        self.assertIn("prehistoric monument", result)
        self.assertIn("History of Stonehenge", result)

    async def test_results_include_a_follow_up_reminder(self) -> None:
        # Nudges the model to web_fetch a promising result instead of
        # settling for thin snippets -- see the real case this addresses in
        # the module history: a search alone was treated as sufficient even
        # though its snippets didn't actually answer the question.
        with mock.patch.object(websearch, "_search", return_value=_RESULTS):
            result = await websearch.run({"query": "Stonehenge"})

        self.assertIn("web_fetch", result)

    async def test_empty_results_are_a_reported_error(self) -> None:
        with mock.patch.object(websearch, "_search", return_value=[]):
            result = await websearch.run({"query": "no such thing"})

        self.assertTrue(result.startswith("Error:"))
        self.assertIn("no such thing", result)

    async def test_search_failure_is_reported_not_raised(self) -> None:
        with mock.patch.object(websearch, "_search", side_effect=DDGSException("rate limited")):
            result = await websearch.run({"query": "Stonehenge"})

        self.assertTrue(result.startswith("Error:"))
        self.assertIn("rate limited", result)

    async def test_a_missing_snippet_or_title_does_not_crash_formatting(self) -> None:
        with mock.patch.object(
            websearch, "_search", return_value=[{"href": "https://example.com"}]
        ):
            result = await websearch.run({"query": "anything"})

        self.assertIn("https://example.com", result)
        self.assertIn("(untitled)", result)

    def test_tool_schema_name_matches_the_registry_key(self) -> None:
        self.assertEqual(websearch.TOOL.name, websearch.TOOL.schema["function"]["name"])


if __name__ == "__main__":
    unittest.main()

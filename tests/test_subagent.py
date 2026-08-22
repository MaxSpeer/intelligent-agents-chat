"""Tests for the sub-agent delegation tool."""

from types import SimpleNamespace
import unittest
from unittest import mock

from intelligent_agents_chat.llm import ContentDelta, LLMError
from intelligent_agents_chat.tools import subagent


def _fake_gateway(*, texts=(), reasoning_texts=(), error=None):
    async def stream_reply(profile, messages, **kwargs):
        if error is not None:
            raise error
        for text in reasoning_texts:
            yield ContentDelta(text, is_reasoning=True)
        for text in texts:
            yield ContentDelta(text)

    return SimpleNamespace(stream_reply=stream_reply)


class SubagentToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_returns_the_subagents_final_answer(self) -> None:
        with mock.patch.object(subagent, "_gateway", _fake_gateway(texts=["Paris"])):
            result = await subagent.run({"task": "What is the capital of France?"})

        self.assertEqual(result, "Paris")

    async def test_reasoning_is_not_part_of_the_returned_answer(self) -> None:
        gateway = _fake_gateway(reasoning_texts=["hmm, let me think..."], texts=["42"])
        with mock.patch.object(subagent, "_gateway", gateway):
            result = await subagent.run({"task": "What is 6 * 7?"})

        self.assertEqual(result, "42")

    async def test_missing_task_is_a_reported_error(self) -> None:
        result = await subagent.run({})
        self.assertTrue(result.startswith("Error:"))

    async def test_empty_final_answer_is_reported_not_silently_returned(self) -> None:
        with mock.patch.object(subagent, "_gateway", _fake_gateway(texts=[])):
            result = await subagent.run({"task": "Do nothing."})

        self.assertIn("no answer", result)

    async def test_llm_errors_are_reported_as_text_not_raised(self) -> None:
        gateway = _fake_gateway(error=LLMError("endpoint unreachable"))
        with mock.patch.object(subagent, "_gateway", gateway):
            result = await subagent.run({"task": "Anything."})

        self.assertTrue(result.startswith("Error:"))
        self.assertIn("endpoint unreachable", result)

    def test_tool_schema_name_matches_the_registry_key(self) -> None:
        self.assertEqual(subagent.TOOL.name, subagent.TOOL.schema["function"]["name"])


if __name__ == "__main__":
    unittest.main()

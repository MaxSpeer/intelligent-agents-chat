"""Token-aware context assembly tests shared by memory and future RAG."""

import unittest

from intelligent_agents_chat.context import (
    MEMORY_GUARD,
    ContextAssembler,
    ContextOverflowError,
)
from intelligent_agents_chat.retrieval import ContextCandidate


def candidate(source_id: str = "1", *, text: str = "The decision was SQLite.") -> ContextCandidate:
    return ContextCandidate(
        source_kind="project_memory",
        source_id=source_id,
        project_id="project-a",
        source_conversation_id="source-chat",
        text=text,
        title="Architecture",
        locator="messages 1-2",
        score=0.5,
    )


class ContextAssemblerTests(unittest.TestCase):
    def test_includes_memory_with_guard_and_auditable_source(self) -> None:
        plan = ContextAssembler("Base system prompt", memory_budget_tokens=500).assemble(
            [
                {"role": "user", "content": "Earlier question"},
                {"role": "assistant", "content": "Earlier answer"},
                {"role": "user", "content": "What database did we choose?"},
            ],
            [candidate()],
            context_window_tokens=2_000,
            output_reserve_tokens=200,
        )

        self.assertEqual(len(plan.included_sources), 1)
        self.assertEqual(plan.included_sources[0].rank, 1)
        self.assertIn(MEMORY_GUARD, plan.messages[0]["content"])
        self.assertEqual(plan.messages[1]["role"], "user")
        self.assertIn("<project-memory>", plan.messages[1]["content"])
        self.assertIn("[project_memory:1]", plan.messages[1]["content"])
        self.assertEqual(plan.messages[-1]["content"], "What database did we choose?")
        self.assertLessEqual(plan.estimated_input_tokens, plan.input_budget_tokens)

    def test_excludes_memory_when_its_complete_wrapper_exceeds_budget(self) -> None:
        plan = ContextAssembler("System", memory_budget_tokens=20).assemble(
            [{"role": "user", "content": "Question"}],
            [candidate(text="large " * 100)],
            context_window_tokens=500,
            output_reserve_tokens=100,
        )

        self.assertEqual(plan.included_sources, ())
        self.assertEqual(len(plan.excluded_sources), 1)
        self.assertEqual(plan.excluded_sources[0].reason, "memory_budget_exceeded")
        self.assertNotIn(MEMORY_GUARD, plan.messages[0]["content"])

    def test_history_is_filled_newest_first_and_the_oldest_is_cut_when_it_does_not_fit(
        self,
    ) -> None:
        plan = ContextAssembler("", memory_budget_tokens=0).assemble(
            [
                {"role": "user", "content": "old " * 100},
                {"role": "assistant", "content": "old reply " * 100},
                {"role": "user", "content": "recent question"},
                {"role": "assistant", "content": "recent answer"},
                {"role": "user", "content": "latest question"},
            ],
            [],
            context_window_tokens=90,
            output_reserve_tokens=30,
        )

        self.assertEqual(
            [message["content"] for message in plan.messages],
            ["recent question", "recent answer", "latest question"],
        )
        self.assertEqual(plan.omitted_history_messages, 2)
        self.assertLessEqual(plan.estimated_input_tokens, plan.input_budget_tokens)

    def test_memory_has_priority_over_history_even_the_most_recent_message(self) -> None:
        """Memory gets first claim on the leftover budget -- not just on
        whatever history's newest-first fill didn't already spend. Without
        this, a long conversation could fill the whole budget on its own
        recent history and starve memory, even though memory is often the
        only way to recall something from a *different* chat.
        """
        plan = ContextAssembler("", memory_budget_tokens=200).assemble(
            [
                {"role": "user", "content": "recent question"},
                {"role": "assistant", "content": "recent answer"},
                {"role": "user", "content": "latest question"},
            ],
            [candidate()],
            context_window_tokens=250,
            output_reserve_tokens=100,
        )

        self.assertEqual(len(plan.included_sources), 1)
        all_content = "\n".join(message["content"] for message in plan.messages)
        self.assertNotIn("recent question", all_content)
        self.assertNotIn("recent answer", all_content)
        self.assertEqual(plan.messages[-1], {"role": "user", "content": "latest question"})
        self.assertEqual(plan.omitted_history_messages, 2)

    def test_raises_when_mandatory_input_cannot_fit(self) -> None:
        with self.assertRaisesRegex(ContextOverflowError, "latest user message"):
            ContextAssembler("System").assemble(
                [{"role": "user", "content": "x" * 300}],
                [],
                context_window_tokens=100,
                output_reserve_tokens=50,
            )


if __name__ == "__main__":
    unittest.main()

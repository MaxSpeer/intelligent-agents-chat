"""CPU-only checks: diagnostic prompts must never contain training answers."""

import sys
import unittest
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diagnose_plain_english import compare_question, first_question


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.row = {"id": "example", "messages": [
            {"role": "system", "content": "Neutral instruction"},
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "SECRET REFERENCE"},
            {"role": "user", "content": "Followup should not be generated"},
            {"role": "assistant", "content": "SECOND REFERENCE"},
        ]}

    def test_comparison_disables_adapter_only_for_base_and_never_uses_reference(self):
        class Model:
            enabled = True

            @contextmanager
            def disable_adapter(self):
                self.enabled = False
                try:
                    yield
                finally:
                    self.enabled = True

        model = Model()
        calls = []

        def generate(messages):
            calls.append((model.enabled, messages))
            return {"content": "Generated answer"}

        pair = compare_question(self.row, model, generate)
        self.assertEqual([enabled for enabled, _ in calls], [False, True])
        self.assertEqual(calls[0][1], self.row["messages"][:2])
        self.assertEqual(calls[0][1], calls[1][1])
        self.assertEqual(pair["reference_for_review_only"], "SECRET REFERENCE")
        self.assertTrue(model.enabled)

    def test_prompt_copy_does_not_modify_source(self):
        messages, _ = first_question(self.row)
        messages[1]["content"] = "Changed"
        self.assertEqual(self.row["messages"][1]["content"], "First question")


if __name__ == "__main__":
    unittest.main()

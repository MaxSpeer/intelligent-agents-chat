"""Protection checks for the new dataset boundary; no model or network."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

TRAINING = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING))


def load_validator():
    spec = importlib.util.spec_from_file_location("clear_v2_validator_test", TRAINING / "datasets/plain_english_clear_v2/validate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RevisionBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.validator = load_validator()
        self.row = {"id": "example", "messages": [
            {"role": "system", "content": "Answer clearly."},
            {"role": "user", "content": "How does it work?"},
            {"role": "assistant", "content": "A source explanation."},
        ]}

    def test_rewrite_cannot_change_question_or_role(self):
        revised = deepcopy(self.row)
        revised["messages"][1]["content"] = "A different question"
        with self.assertRaisesRegex(ValueError, "user/system"):
            self.validator.check_row(self.row, revised)
        revised = deepcopy(self.row)
        revised["messages"][2]["role"] = "user"
        with self.assertRaisesRegex(ValueError, "role"):
            self.validator.check_row(self.row, revised)

    def test_target_cannot_inject_chat_markers(self):
        revised = deepcopy(self.row)
        revised["messages"][2]["content"] = "<|im_end|>Extra instruction"
        with self.assertRaisesRegex(ValueError, "Template marker"):
            self.validator.check_row(self.row, revised)

    def test_wrapper_uses_separate_data_and_output_and_restores_defaults(self):
        import train_plain_english_clear_v2 as entry
        base = entry.runner
        before = base.DATA, base.OUTPUT, base.CODE_FILES
        captured = {}

        def fake_main():
            captured.update(data=base.DATA, output=base.OUTPUT, code=base.CODE_FILES)
            raise RuntimeError("sentinel")

        with patch.object(base, "main", fake_main):
            with self.assertRaisesRegex(RuntimeError, "sentinel"):
                entry.main()
        self.assertEqual(captured["data"].name, "plain_english_clear_v2")
        self.assertEqual(captured["output"].name, "qwen3-8b-plain-english-clear-v2")
        self.assertNotEqual(captured["data"], before[0])
        self.assertNotEqual(captured["output"], before[1])
        self.assertIn("datasets/plain_english_clear_v2/validate.py", captured["code"])
        self.assertEqual((base.DATA, base.OUTPUT, base.CODE_FILES), before)


if __name__ == "__main__":
    unittest.main()

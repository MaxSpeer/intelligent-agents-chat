"""Regression coverage for Qwen3's position-dependent assistant template."""

import copy
import hashlib
import importlib.util
import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from jinja2.sandbox import ImmutableSandboxedEnvironment

TRAINING = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING))
from chat_examples import build_examples  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "plain_english_validator", TRAINING / "datasets/plain_english/validate.py",
)
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)

TEMPLATE = (Path(__file__).parent / "fixtures/qwen3-chat-template.jinja").read_text()
MESSAGES = [
    {"role": "system", "content": "Explain clearly."},
    {"role": "user", "content": "What does gravity do?"},
    {"role": "assistant", "content": "Gravity pulls things together."},
    {"role": "user", "content": "What happens when I drop a ball?"},
    {"role": "assistant", "content": "The ball falls toward the ground."},
]


class CharacterTokenizer:
    """Real Qwen Jinja rendering with transparent character IDs for unit tests."""

    eos_token = "<|im_end|>"

    def __init__(self):
        self.template = ImmutableSandboxedEnvironment(
            trim_blocks=True, lstrip_blocks=True,
        ).from_string(TEMPLATE)

    def apply_chat_template(self, messages, **kwargs):
        return self.template.render(messages=messages, **kwargs)

    def __call__(self, text, **kwargs):
        return {"input_ids": [ord(c) for c in text]}

    def decode(self, ids, **kwargs):
        return "".join(chr(i) for i in ids)


class ChatExampleTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = CharacterTokenizer()

    def assert_target(self, example, content):
        self.assertTrue(all(x == -100 for x in example.labels[:example.prompt_length]))
        self.assertEqual(example.labels[example.prompt_length:],
                         example.input_ids[example.prompt_length:])
        text = self.tokenizer.decode([i for i in example.labels if i != -100])
        self.assertEqual(text, content + "<|im_end|>\n")
        self.assertNotIn("<|im_start|>", text)
        self.assertNotIn("<think>", text)

    def test_single_answer_masks_prompt_and_empty_thinking_block(self):
        example, = build_examples(self.tokenizer, MESSAGES[:3], 1024)
        self.assert_target(example, MESSAGES[2]["content"])
        prompt = self.tokenizer.decode(example.input_ids[:example.prompt_length])
        self.assertTrue(prompt.endswith("<think>\n\n</think>\n\n"))

    def test_followup_is_context_never_part_of_previous_learning_target(self):
        first, second = build_examples(self.tokenizer, MESSAGES, 1024)
        self.assert_target(first, MESSAGES[2]["content"])
        self.assert_target(second, MESSAGES[4]["content"])
        self.assertNotIn(MESSAGES[3]["content"], self.tokenizer.decode(first.input_ids))
        second_prompt = self.tokenizer.decode(second.input_ids[:second.prompt_length])
        self.assertIn(MESSAGES[2]["content"], second_prompt)
        self.assertIn(MESSAGES[3]["content"], second_prompt)

    def test_fixture_reproduces_original_non_prefix_bug(self):
        full = self.tokenizer.apply_chat_template(MESSAGES)
        through_first_answer = self.tokenizer.apply_chat_template(MESSAGES[:3])
        self.assertFalse(full.startswith(through_first_answer))
        # The old length-derived endpoint extends past the first assistant.
        start = len(self.tokenizer.apply_chat_template(
            MESSAGES[:2], add_generation_prompt=True,
        ))
        old_supervision = full[start:len(through_first_answer)]
        self.assertIn("<|im_start|>user", old_supervision)
        self.assertEqual(len(build_examples(self.tokenizer, MESSAGES, 1024)), 2)

    def test_every_answer_in_three_turn_dialogue_is_supervised_once(self):
        messages = MESSAGES + [
            {"role": "user", "content": "And on the Moon?"},
            {"role": "assistant", "content": "It falls there too, but more slowly."},
        ]
        examples = build_examples(self.tokenizer, messages, 1024)
        self.assertEqual([e.message_index for e in examples], [2, 4, 6])
        for example, index in zip(examples, [2, 4, 6]):
            self.assert_target(example, messages[index]["content"])

    def test_does_not_truncate_away_end_marker(self):
        example, = build_examples(self.tokenizer, MESSAGES[:3], 1024)
        with self.assertRaisesRegex(ValueError, "exceed max_seq_len"):
            build_examples(self.tokenizer, MESSAGES[:3], len(example.input_ids) - 1)

    def test_fails_if_tokenization_merges_across_prompt_boundary(self):
        class MergingTokenizer(CharacterTokenizer):
            def __call__(self, text, **kwargs):
                ids = super().__call__(text, **kwargs)["input_ids"]
                if text.endswith("</think>\n\n"):
                    ids[-1] = 99999
                return {"input_ids": ids}
        with self.assertRaisesRegex(ValueError, "not an exact prefix"):
            build_examples(MergingTokenizer(), MESSAGES[:3], 1024)

    def test_rejects_unsupported_messages_and_reasoning(self):
        variants = [[], MESSAGES[:-1], [{"role": "tool", "content": "x"}]]
        for marker in ("<think>", "</think>", "<|im_start|>"):
            row = copy.deepcopy(MESSAGES[:3])
            row[-1]["content"] = marker + " unwanted"
            variants.append(row)
        for messages in variants:
            with self.subTest(messages=messages), self.assertRaises(ValueError):
                build_examples(self.tokenizer, messages, 1024)

    def test_validator_rejects_a_positive_count_with_unmasked_user_tokens(self):
        bad = build_examples(self.tokenizer, MESSAGES, 1024)
        bad[0].labels[0] = bad[0].input_ids[0]
        with patch.object(validator, "build_examples", return_value=bad):
            with self.assertRaisesRegex(ValueError, "Unmasked context"):
                validator.validate_masks([{"id": "regression", "messages": MESSAGES}],
                                         self.tokenizer, 1024)


@unittest.skipUnless(os.environ.get("QWEN_TOKENIZER_PATH"), "Set QWEN_TOKENIZER_PATH")
class RealTokenizerTests(unittest.TestCase):
    def test_all_118_actual_answers_have_exact_targets(self):
        snapshot = Path(os.environ["QWEN_TOKENIZER_PATH"])
        data = TRAINING / "datasets/plain_english"
        manifest = json.loads((data / "manifest.json").read_text())
        for name, expected in manifest["tokenizer"]["file_sha256"].items():
            self.assertEqual(hashlib.sha256((snapshot / name).read_bytes()).hexdigest(), expected)
        try:
            from transformers import AutoTokenizer
        except ImportError:
            from tokenizers import Tokenizer
            class RustTokenizer(CharacterTokenizer):
                def __init__(self):
                    super().__init__()
                    self.raw = Tokenizer.from_file(str(snapshot / "tokenizer.json"))
                def __call__(self, text, **kwargs):
                    return {"input_ids": self.raw.encode(text, add_special_tokens=False).ids}
                def decode(self, ids, **kwargs):
                    return self.raw.decode(ids, skip_special_tokens=False)
            tokenizer = RustTokenizer()
        else:
            tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
        for split, count in (("train", 93), ("validation", 13), ("test", 12)):
            rows = [json.loads(line) for line in (data / f"{split}.jsonl").read_text().splitlines()]
            with self.subTest(split=split):
                report = validator.validate_masks(rows, tokenizer, 1024)
                self.assertEqual(report["answer_examples"], count)
                self.assertTrue(report["all_context_tokens_masked"])
                self.assertTrue(report["all_supervised_text_matches_reference_answer_and_end_marker"])


if __name__ == "__main__":
    unittest.main()

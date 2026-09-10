"""Tests for the configured model profiles."""

import unittest
from unittest.mock import patch

from intelligent_agents_chat.models import (
    DEFAULT_PROFILE_KEY,
    MODEL_PROFILES,
    _port,
    get_profile,
    profile_options,
)


class PortEnvironmentOverrideTests(unittest.TestCase):
    def test_missing_env_var_falls_back_to_the_default(self) -> None:
        with patch.dict("os.environ", {}, clear=False):
            self.assertEqual(_port("VLLM_DOES_NOT_EXIST_PORT", 8001), 8001)

    def test_env_var_overrides_the_default(self) -> None:
        with patch.dict("os.environ", {"VLLM_QWEN3_8B_PORT": "9001"}):
            self.assertEqual(_port("VLLM_QWEN3_8B_PORT", 8001), 9001)

    def test_non_integer_env_var_is_rejected(self) -> None:
        with patch.dict("os.environ", {"VLLM_QWEN3_8B_PORT": "not-a-port"}):
            with self.assertRaisesRegex(ValueError, "must be an integer port"):
                _port("VLLM_QWEN3_8B_PORT", 8001)


class ModelProfilesTests(unittest.TestCase):
    def test_default_profile_is_qwen3_8b(self) -> None:
        self.assertEqual(DEFAULT_PROFILE_KEY, "qwen3-8b")
        self.assertIn(DEFAULT_PROFILE_KEY, profile_options())

    def test_only_three_profiles_are_active(self) -> None:
        self.assertEqual(
            [profile.key for profile in MODEL_PROFILES],
            ["qwen3-8b", "conspiracy", "plain-english-clear-v2"],
        )

    def test_qwen3_8b_and_its_adapters_share_the_same_backend(self) -> None:
        qwen3_8b = get_profile("qwen3-8b")

        self.assertEqual(qwen3_8b.base_url, "http://127.0.0.1:8001/v1")
        self.assertFalse(qwen3_8b.supports_thinking)
        self.assertFalse(qwen3_8b.supports_tools)
        for key in ("conspiracy", "plain-english-clear-v2"):
            with self.subTest(adapter=key):
                adapter = get_profile(key)
                self.assertEqual(qwen3_8b.base_url, adapter.base_url)
                self.assertEqual(adapter.model, key)
                self.assertFalse(adapter.supports_thinking)
                self.assertFalse(adapter.supports_tools)

    def test_qwen35_9b_runs_on_a_separate_backend(self) -> None:
        profile = get_profile("qwen3.5-9b")

        self.assertEqual(profile.base_url, "http://127.0.0.1:8002/v1")
        self.assertEqual(profile.model, "qwen3.5-9b")
        self.assertTrue(profile.supports_thinking)
        self.assertTrue(profile.supports_tools)

    def test_local_ollama_profile_disables_reasoning(self) -> None:
        profile = get_profile("ollama-local")

        self.assertEqual(profile.base_url, "http://127.0.0.1:11434/v1")
        self.assertEqual(profile.model, "qwen3.5:2b")
        self.assertEqual(profile.reasoning_effort, "none")
        self.assertFalse(profile.supports_thinking)

    def test_unknown_profile_is_rejected(self) -> None:
        with self.assertRaises(KeyError):
            get_profile("missing")

    def test_profile_options_expose_labels_for_the_ui_selector(self) -> None:
        self.assertEqual(
            profile_options(),
            {
                "qwen3-8b": "Qwen3 8B",
                "conspiracy": "Qwen3 8B (Conspiracy)",
                "plain-english-clear-v2": "Qwen3 8B (Simple English)",
            },
        )

    def test_inactive_profiles_keep_history_labels_but_are_not_selectable(self) -> None:
        historical_labels = {
            "qwen3.5-9b": "Qwen3.5 9B",
            "plain-english": "Qwen3 8B (Plain English)",
            "plain-english-2k": "Qwen3 8B (Plain English 2k)",
            "plain-english-2k-simplified-v1": "Qwen3 8B (Plain English 2k simplified)",
            "ollama-local": "qwen3.5:2b (local Ollama)",
        }
        for key, label in historical_labels.items():
            with self.subTest(profile=key):
                self.assertNotIn(key, profile_options())
                self.assertEqual(get_profile(key).label, label)


if __name__ == "__main__":
    unittest.main()

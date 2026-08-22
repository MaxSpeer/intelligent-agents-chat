"""Tests for the hardcoded model profiles."""

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
    def test_default_profile_is_qwen35_9b(self) -> None:
        self.assertEqual(DEFAULT_PROFILE_KEY, "qwen3.5-9b")
        self.assertIn(DEFAULT_PROFILE_KEY, profile_options())

    def test_three_profiles_are_configured(self) -> None:
        self.assertEqual(
            {profile.key for profile in MODEL_PROFILES},
            {"qwen3.5-9b", "qwen3-8b", "conspiracy"},
        )

    def test_qwen3_8b_and_conspiracy_share_the_same_backend(self) -> None:
        qwen3_8b = get_profile("qwen3-8b")
        conspiracy = get_profile("conspiracy")

        self.assertEqual(qwen3_8b.base_url, conspiracy.base_url)
        self.assertEqual(qwen3_8b.base_url, "http://127.0.0.1:8001/v1")
        self.assertEqual(conspiracy.model, "conspiracy")
        self.assertFalse(qwen3_8b.supports_thinking)
        self.assertFalse(conspiracy.supports_thinking)
        self.assertFalse(qwen3_8b.supports_tools)
        self.assertFalse(conspiracy.supports_tools)

    def test_qwen35_9b_runs_on_a_separate_backend(self) -> None:
        profile = get_profile("qwen3.5-9b")

        self.assertEqual(profile.base_url, "http://127.0.0.1:8002/v1")
        self.assertEqual(profile.model, "qwen3.5-9b")
        self.assertTrue(profile.supports_thinking)
        self.assertTrue(profile.supports_tools)

    def test_unknown_profile_is_rejected(self) -> None:
        with self.assertRaises(KeyError):
            get_profile("missing")

    def test_profile_options_expose_labels_for_the_ui_selector(self) -> None:
        self.assertEqual(
            profile_options(),
            {
                "qwen3.5-9b": "Qwen3.5 9B",
                "qwen3-8b": "Qwen3 8B",
                "conspiracy": "Qwen3 8B (conspiracy)",
            },
        )


if __name__ == "__main__":
    unittest.main()

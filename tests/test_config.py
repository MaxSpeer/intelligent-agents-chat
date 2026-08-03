"""Tests for environment-driven application configuration."""

from pathlib import Path
import unittest

from intelligent_agents_chat.config import Settings


class SettingsTests(unittest.TestCase):
    def test_defaults_include_offline_model_and_documented_vllm_tunnel(self) -> None:
        settings = Settings.from_env({})

        self.assertEqual(settings.default_profile_key, "lorem")
        self.assertEqual(settings.profile("lorem").backend, "lorem")
        self.assertEqual(settings.profile("default").backend, "vllm")
        self.assertEqual(settings.profile("default").model, "qwen3-0.6b")
        self.assertEqual(
            settings.profile("default").base_url,
            "http://127.0.0.1:8000/v1",
        )
        self.assertEqual(settings.api_key, "not-needed")
        self.assertEqual(settings.database_path.name, "chats.sqlite3")

    def test_multiple_model_profiles_are_parsed_and_selectable(self) -> None:
        settings = Settings.from_env(
            {
                "CHAT_DB_PATH": "/tmp/agent-lab-tests.sqlite3",
                "CHAT_DEFAULT_PROFILE": "tuned",
                "VLLM_PROFILES_JSON": """
                    [
                        {
                            "key": "base",
                            "label": "Qwen Base",
                            "base_url": "http://127.0.0.1:8000/v1/",
                            "model": "qwen-base"
                        },
                        {
                            "key": "tuned",
                            "label": "Qwen Tuned",
                            "base_url": "http://127.0.0.1:8001/v1",
                            "model": "qwen-tuned"
                        }
                    ]
                """,
            }
        )

        self.assertEqual(settings.database_path, Path("/tmp/agent-lab-tests.sqlite3"))
        self.assertEqual(settings.default_profile_key, "tuned")
        self.assertEqual(
            settings.profile_options,
            {
                "lorem": "Lorem Ipsum (offline)",
                "base": "Qwen Base",
                "tuned": "Qwen Tuned",
            },
        )
        self.assertEqual(settings.profile("base").base_url, "http://127.0.0.1:8000/v1")

    def test_duplicate_profile_keys_are_rejected(self) -> None:
        profiles = """
            [
                {"key": "same", "label": "One", "base_url": "http://one/v1", "model": "a"},
                {"key": "same", "label": "Two", "base_url": "http://two/v1", "model": "b"}
            ]
        """

        with self.assertRaisesRegex(ValueError, "keys must be unique"):
            Settings.from_env({"VLLM_PROFILES_JSON": profiles})

    def test_vllm_profile_cannot_replace_the_builtin_lorem_profile(self) -> None:
        with self.assertRaisesRegex(ValueError, "keys must be unique"):
            Settings.from_env({"VLLM_PROFILE_KEY": "lorem"})

    def test_unknown_default_profile_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "CHAT_DEFAULT_PROFILE"):
            Settings.from_env({"CHAT_DEFAULT_PROFILE": "missing"})


if __name__ == "__main__":
    unittest.main()

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
        self.assertEqual(settings.profile("qwen3-8b").backend, "vllm")
        self.assertEqual(settings.profile("qwen3-8b").label, "Qwen3 8B")
        self.assertEqual(settings.profile("qwen3-8b").model, "qwen3-8b")
        self.assertTrue(settings.profile("qwen3-8b").supports_thinking)
        self.assertEqual(
            settings.profile("qwen3-8b").base_url,
            "http://127.0.0.1:8001/v1",
        )
        self.assertEqual(settings.profile("conspiracy").backend, "vllm")
        self.assertEqual(settings.profile("conspiracy").model, "conspiracy")
        self.assertTrue(settings.profile("conspiracy").supports_thinking)
        self.assertEqual(
            settings.profile("conspiracy").base_url,
            settings.profile("qwen3-8b").base_url,
        )
        self.assertEqual(settings.api_key, "not-needed")
        self.assertEqual(settings.max_tokens, 1024)
        self.assertEqual(settings.thinking_max_tokens, 8192)
        self.assertEqual(settings.database_path.name, "chats.sqlite3")
        self.assertEqual(settings.log_path.name, "agent-lab.jsonl")
        self.assertEqual(settings.log_level, "INFO")
        self.assertEqual(settings.log_max_bytes, 10_485_760)
        self.assertEqual(settings.log_backup_count, 5)

    def test_logging_configuration_is_loaded_and_validated(self) -> None:
        settings = Settings.from_env(
            {
                "CHAT_LOG_PATH": "/tmp/agent-lab-test.jsonl",
                "CHAT_LOG_LEVEL": "debug",
                "CHAT_LOG_MAX_BYTES": "2048",
                "CHAT_LOG_BACKUP_COUNT": "2",
            }
        )

        self.assertEqual(settings.log_path, Path("/tmp/agent-lab-test.jsonl"))
        self.assertEqual(settings.log_level, "DEBUG")
        self.assertEqual(settings.log_max_bytes, 2048)
        self.assertEqual(settings.log_backup_count, 2)

        with self.assertRaisesRegex(ValueError, "CHAT_LOG_LEVEL"):
            Settings.from_env({"CHAT_LOG_LEVEL": "verbose"})
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            Settings.from_env({"CHAT_LOG_BACKUP_COUNT": "0"})

    def test_bundled_9b_profile_can_follow_a_different_local_tunnel(self) -> None:
        settings = Settings.from_env(
            {
                "VLLM_9B_BASE_URL": "http://127.0.0.1:9001/v1/",
                "VLLM_9B_PROFILE_LABEL": "Qwen 9B on HPI",
                "VLLM_9B_MODEL": "custom-9b-name",
            }
        )

        profile = settings.profile("qwen3-8b")
        self.assertEqual(profile.label, "Qwen 9B on HPI")
        self.assertEqual(profile.base_url, "http://127.0.0.1:9001/v1")
        self.assertEqual(profile.model, "custom-9b-name")

    def test_lora_profile_can_be_disabled(self) -> None:
        settings = Settings.from_env({"VLLM_9B_LORA_MODEL": ""})

        with self.assertRaises(KeyError):
            settings.profile("conspiracy")

    def test_lora_profile_defaults_to_the_9b_tunnel(self) -> None:
        settings = Settings.from_env(
            {
                "VLLM_9B_BASE_URL": "http://127.0.0.1:9001/v1",
                "VLLM_9B_LORA_MODEL": "conspiracy",
            }
        )

        profile = settings.profile("conspiracy")
        self.assertEqual(profile.label, "Qwen3 8B (conspiracy)")
        self.assertEqual(profile.base_url, "http://127.0.0.1:9001/v1")
        self.assertEqual(profile.model, "conspiracy")
        self.assertTrue(profile.supports_thinking)
        self.assertIn("conspiracy", settings.profile_options)

    def test_lora_profile_key_label_and_base_url_are_overridable(self) -> None:
        settings = Settings.from_env(
            {
                "VLLM_9B_LORA_MODEL": "conspiracy",
                "VLLM_9B_LORA_PROFILE_KEY": "conspiracy-adapter",
                "VLLM_9B_LORA_PROFILE_LABEL": "Conspiracy adapter",
                "VLLM_9B_LORA_BASE_URL": "http://127.0.0.1:9002/v1",
            }
        )

        profile = settings.profile("conspiracy-adapter")
        self.assertEqual(profile.label, "Conspiracy adapter")
        self.assertEqual(profile.base_url, "http://127.0.0.1:9002/v1")
        self.assertEqual(profile.model, "conspiracy")

    def test_normal_and_thinking_token_limits_are_configurable(self) -> None:
        settings = Settings.from_env(
            {
                "VLLM_MAX_TOKENS": "2048",
                "VLLM_THINKING_MAX_TOKENS": "12288",
            }
        )

        self.assertEqual(settings.max_tokens, 2048)
        self.assertEqual(settings.thinking_max_tokens, 12288)

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
                            "model": "qwen-tuned",
                            "supports_thinking": true
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
        self.assertFalse(settings.profile("base").supports_thinking)
        self.assertTrue(settings.profile("tuned").supports_thinking)

    def test_duplicate_profile_keys_are_rejected(self) -> None:
        profiles = """
            [
                {"key": "same", "label": "One", "base_url": "http://one/v1", "model": "a"},
                {"key": "same", "label": "Two", "base_url": "http://two/v1", "model": "b"}
            ]
        """

        with self.assertRaisesRegex(ValueError, "keys must be unique"):
            Settings.from_env({"VLLM_PROFILES_JSON": profiles})

    def test_custom_profile_thinking_flag_must_be_boolean(self) -> None:
        profiles = """
            [
                {
                    "key": "thinking",
                    "label": "Thinking",
                    "base_url": "http://localhost:8000/v1",
                    "model": "thinking-model",
                    "supports_thinking": "yes"
                }
            ]
        """

        with self.assertRaisesRegex(ValueError, "supports_thinking must be a boolean"):
            Settings.from_env({"VLLM_PROFILES_JSON": profiles})

    def test_vllm_profile_cannot_replace_the_builtin_lorem_profile(self) -> None:
        with self.assertRaisesRegex(ValueError, "keys must be unique"):
            Settings.from_env({"VLLM_PROFILE_KEY": "lorem"})

    def test_unknown_default_profile_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "CHAT_DEFAULT_PROFILE"):
            Settings.from_env({"CHAT_DEFAULT_PROFILE": "missing"})


if __name__ == "__main__":
    unittest.main()

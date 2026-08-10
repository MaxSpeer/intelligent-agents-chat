"""Tests for privacy-safe structured application logs."""

from contextlib import redirect_stderr
from io import StringIO
import json
import logging
from pathlib import Path
import stat
from tempfile import TemporaryDirectory
import unittest

from intelligent_agents_chat.config import Settings
from intelligent_agents_chat.logging_config import (
    LOGGER_NAMESPACE,
    configure_logging,
    log_event,
    sanitized_endpoint,
    shutdown_logging,
)


class LoggingTests(unittest.TestCase):
    def tearDown(self) -> None:
        shutdown_logging()

    def test_json_log_contains_context_and_exception_details(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            log_path = Path(temporary_directory) / "agent-lab.jsonl"
            settings = Settings.from_env(
                {
                    "CHAT_LOG_PATH": str(log_path),
                    "CHAT_LOG_LEVEL": "DEBUG",
                    "CHAT_LOG_MAX_BYTES": "4096",
                    "CHAT_LOG_BACKUP_COUNT": "1",
                }
            )

            with redirect_stderr(StringIO()):
                configure_logging(settings)
                logger = logging.getLogger(f"{LOGGER_NAMESPACE}.test")
                log_event(
                    logger,
                    logging.INFO,
                    "test.context",
                    project_id="project-1",
                    message_chars=42,
                )
                try:
                    raise RuntimeError("test failure")
                except RuntimeError:
                    logger.exception(
                        "test.exception",
                        extra={"event": "test.exception", "operation_id": "operation-1"},
                    )
                for handler in logging.getLogger(LOGGER_NAMESPACE).handlers:
                    handler.flush()

            records = [json.loads(line) for line in log_path.read_text().splitlines()]

        context_record = next(record for record in records if record["event"] == "test.context")
        self.assertEqual(context_record["project_id"], "project-1")
        self.assertEqual(context_record["message_chars"], 42)
        self.assertIn("timestamp", context_record)
        self.assertRegex(context_record["source"], r"^test_logging_config:\d+$")

        exception_record = next(record for record in records if record["event"] == "test.exception")
        self.assertEqual(exception_record["operation_id"], "operation-1")
        self.assertIn("RuntimeError: test failure", exception_record["exception"])

    def test_endpoint_sanitization_removes_credentials_and_query(self) -> None:
        sanitized = sanitized_endpoint(
            "https://user:secret@example.test:8443/v1?api_key=secret#fragment"
        )

        self.assertEqual(sanitized, "https://example.test:8443/v1")

    def test_logs_rotate_and_are_private(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            log_path = Path(temporary_directory) / "agent-lab.jsonl"
            settings = Settings.from_env(
                {
                    "CHAT_LOG_PATH": str(log_path),
                    "CHAT_LOG_MAX_BYTES": "512",
                    "CHAT_LOG_BACKUP_COUNT": "2",
                }
            )

            with redirect_stderr(StringIO()):
                configure_logging(settings)
                logger = logging.getLogger(f"{LOGGER_NAMESPACE}.rotation-test")
                for index in range(20):
                    log_event(
                        logger,
                        logging.INFO,
                        "test.rotation",
                        record_index=index,
                        diagnostic_value_chars=128,
                    )
                for handler in logging.getLogger(LOGGER_NAMESPACE).handlers:
                    handler.flush()

            rotated_path = Path(f"{log_path}.1")
            self.assertTrue(log_path.exists())
            self.assertTrue(rotated_path.exists())
            self.assertEqual(stat.S_IMODE(log_path.stat().st_mode), 0o600)
            self.assertEqual(stat.S_IMODE(rotated_path.stat().st_mode), 0o600)

    def test_invalid_endpoint_is_safe_to_log(self) -> None:
        self.assertEqual(sanitized_endpoint("http://example.test:invalid/v1"), "<invalid endpoint>")


if __name__ == "__main__":
    unittest.main()

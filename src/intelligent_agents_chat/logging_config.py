"""Structured, rotating application logs for later debugging."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import sys
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from intelligent_agents_chat.config import Settings


LOGGER_NAMESPACE = "intelligent_agents_chat"
_STANDARD_RECORD_FIELDS = set(logging.makeLogRecord({}).__dict__) | {
    "asctime",
    "message",
}
logging.getLogger(LOGGER_NAMESPACE).addHandler(logging.NullHandler())


class JsonLineFormatter(logging.Formatter):
    """Serialize one log record per line without losing exception details."""

    def format(self, record: logging.LogRecord) -> str:
        message = record.getMessage()
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "event": getattr(record, "event", message),
            "message": message,
            "source": f"{record.module}:{record.lineno}",
            "process_id": record.process,
            "thread_name": record.threadName,
        }

        for key, value in record.__dict__.items():
            if key in _STANDARD_RECORD_FIELDS or key in payload or key == "event":
                continue
            payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)

        return json.dumps(payload, ensure_ascii=False, default=_json_default)


class PrivateRotatingFileHandler(RotatingFileHandler):
    """Keep current and rotated logs readable only by the application user."""

    def _open(self):
        stream = super()._open()
        _make_private_if_possible(Path(self.baseFilename))
        return stream


def configure_logging(settings: Settings) -> logging.Logger:
    """Configure console and rotating-file handlers for the application namespace."""
    settings.log_path.parent.mkdir(parents=True, exist_ok=True)
    for index in range(settings.log_backup_count + 1):
        candidate = settings.log_path if index == 0 else Path(f"{settings.log_path}.{index}")
        if candidate.exists():
            _make_private_if_possible(candidate)
    logger = logging.getLogger(LOGGER_NAMESPACE)
    _close_handlers(logger)
    logger.setLevel(settings.log_level)
    logger.propagate = False

    formatter = JsonLineFormatter()
    file_handler = PrivateRotatingFileHandler(
        settings.log_path,
        maxBytes=settings.log_max_bytes,
        backupCount=settings.log_backup_count,
        encoding="utf-8",
        delay=True,
    )
    file_handler.setLevel(settings.log_level)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(settings.log_level)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    log_event(
        logger,
        logging.INFO,
        "logging.configured",
        log_path=str(settings.log_path),
        log_level=settings.log_level,
        log_max_bytes=settings.log_max_bytes,
        log_backup_count=settings.log_backup_count,
    )
    return logger


def shutdown_logging() -> None:
    """Flush and close application handlers, primarily for clean tests and shutdowns."""
    _close_handlers(logging.getLogger(LOGGER_NAMESPACE))


def log_event(
    logger: logging.Logger,
    level: int,
    event: str,
    **fields: object,
) -> None:
    """Write a structured event with arbitrary JSON-safe diagnostic fields."""
    logger.log(level, event, extra={"event": event, **fields}, stacklevel=2)


def sanitized_endpoint(value: str | None) -> str | None:
    """Keep endpoint routing details while removing credentials, query, and fragment."""
    if not value:
        return None
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname or ""
        if ":" in hostname:
            hostname = f"[{hostname}]"
        netloc = hostname
        if parsed.port is not None:
            netloc = f"{netloc}:{parsed.port}"
        return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))
    except ValueError:
        return "<invalid endpoint>"


def _close_handlers(logger: logging.Logger) -> None:
    for handler in tuple(logger.handlers):
        logger.removeHandler(handler)
        handler.flush()
        handler.close()


def _json_default(value: object) -> str:
    if isinstance(value, Path):
        return str(value)
    return repr(value)


def _make_private_if_possible(path: Path) -> None:
    try:
        os.chmod(path, 0o600)
    except OSError:
        # Logging must remain available on filesystems that do not expose POSIX modes.
        pass

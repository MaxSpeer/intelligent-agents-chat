"""Initialize logging, the database schema, and project memory.

Call bootstrap() before serving requests or using storage from scripts.
Repeated calls are safe; importing application modules does not initialize storage.
"""

from __future__ import annotations

import logging

from intelligent_agents_chat.database import repository
from intelligent_agents_chat.logging_config import configure_logging, log_event
from intelligent_agents_chat.memory import memory_store

logger = logging.getLogger(__name__)


def bootstrap() -> None:
    """Configure logging, create the schema, and backfill project memory."""
    configure_logging()
    try:
        repository.initialize()
    except Exception:
        logger.exception(
            "application.database_initialization_failed",
            extra={
                "event": "application.database_initialization_failed",
                "database_path": str(repository.database_path),
            },
        )
        raise

    # Rebuild memory from stored messages; a failure must not prevent startup.
    try:
        rebuilt_entry_count = sum(
            memory_store.rebuild_project(project.id) for project in repository.list_projects()
        )
    except Exception:
        logger.exception(
            "application.memory_backfill_failed",
            extra={
                "event": "application.memory_backfill_failed",
                "database_path": str(repository.database_path),
            },
        )
    else:
        log_event(
            logger,
            logging.INFO,
            "application.memory_backfill_completed",
            entry_count=rebuilt_entry_count,
        )

    log_event(logger, logging.INFO, "application.initialized")

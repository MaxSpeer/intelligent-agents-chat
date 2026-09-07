"""Application startup: the one place that actually *does* something.

Every other module in this package is safe to import -- importing it defines
classes and builds objects, but never opens the database, writes a file, or
talks to the network. That property is worth protecting: it keeps unit tests
from touching the real database just by importing the module under test, and
it makes import order irrelevant.

The work those modules avoid at import time has to happen somewhere, though,
and this is it: create the schema, then bring project memory up to date with
whatever is already stored. main() calls bootstrap() once, before the server
starts serving -- so a request can never arrive before the database exists.

Anything else that runs outside main() and needs a working database (a
script, a REPL session) should call bootstrap() first too. It is safe to call
more than once: initialize() only creates what is missing, and the memory
rebuild is idempotent by design (see ProjectMemoryStore.rebuild_conversation).
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

    # Memory is derived data: every entry can be rebuilt from the messages
    # that are already stored. Doing it once at startup means a database that
    # was written by an older version -- or edited by hand -- still has
    # complete, current memory before the first question arrives. A failure
    # here is logged but not raised: chatting without project memory is worth
    # more than not starting at all.
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

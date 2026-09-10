"""Process entrypoint: start the NiceGUI server."""

from __future__ import annotations

import logging

from nicegui import ui

from intelligent_agents_chat import app  # noqa: F401 -- registers the "/" page route
from intelligent_agents_chat.bootstrap import bootstrap
from intelligent_agents_chat.logging_config import log_event, shutdown_logging

logger = logging.getLogger(__name__)


def main() -> None:
    """Start the NiceGUI development server."""
    # Initialize storage before serving requests.
    bootstrap()
    log_event(
        logger,
        logging.INFO,
        "application.server.starting",
        host="127.0.0.1",
        port=8080,
        reload=False,
    )
    try:
        ui.run(title="Agent Lab", favicon="✨", host="127.0.0.1", port=8080, reload=False)
    except Exception:
        logger.exception(
            "application.server.failed",
            extra={"event": "application.server.failed"},
        )
        raise
    finally:
        log_event(logger, logging.INFO, "application.server.stopped")
        shutdown_logging()


if __name__ in {"__main__", "__mp_main__"}:
    main()

#!/usr/bin/env python3
"""Explicitly initialize or migrate a local chat database.

The application runs the same idempotent migrations automatically at startup.
This helper is useful for checking a database before launching the UI:

    uv run python scripts/migrate_dev_db.py [path-to-chats.sqlite3]

"""

from __future__ import annotations

from pathlib import Path
import sys

from intelligent_agents_chat.database import ChatRepository, connect_database

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATABASE_PATH = PROJECT_ROOT / ".data" / "chats.sqlite3"


def migrate(database_path: Path) -> None:
    ChatRepository(database_path).initialize()
    with connect_database(database_path) as connection:
        message_count = connection.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        schema_version = connection.execute("PRAGMA user_version").fetchone()[0]
    print(
        f"Done. {database_path} is at schema {schema_version} "
        f"({message_count} messages kept)."
    )


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DATABASE_PATH
    migrate(target)

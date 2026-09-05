#!/usr/bin/env python3
"""Explicitly initialize or migrate a local chat database.

The application runs the same idempotent migrations automatically at startup.
This helper is useful for checking a database before launching the UI:

    uv run python scripts/migrate_dev_db.py [path-to-chats.sqlite3]

This script is written for the schema changes it currently knows about
(tool-calling support, a `reasoning` column, the `memory_enabled` column on
`conversations`, a `real_peak_total_tokens` column on
`message_context_runs`, then dropping `rag_enabled` and
`message_adaptive_rag_runs` now that project-document retrieval is a tool
instead of an automatic, per-conversation-toggle pipeline). Document/chunk
tables (documents.py's own) aren't handled here -- DocumentStore.initialize()
migrates those itself, automatically, every time the app starts. If
database.py's schema changes again later, extend or replace the migration
below to match -- it's a one-off dev tool, not a general migration framework.
"""

from __future__ import annotations

from pathlib import Path
import sqlite3
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATABASE_PATH = PROJECT_ROOT / ".data" / "chats.sqlite3"


def migrate(database_path: Path) -> None:
    if not database_path.exists():
        print(f"No database at {database_path}; nothing to migrate (a fresh one will be created).")
        return

    with sqlite3.connect(database_path) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(messages)")}

        if not {"tool_calls", "tool_call_id"} <= columns:
            print(
                "Adding tool-calling support to the messages table (rebuilding it, since "
                "SQLite can't just widen a CHECK constraint) ..."
            )
            connection.executescript(
                """
                ALTER TABLE messages RENAME TO messages_old;

                CREATE TABLE messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL
                        REFERENCES conversations(id) ON DELETE CASCADE,
                    role TEXT NOT NULL CHECK (role IN ('system', 'user', 'assistant', 'tool')),
                    content TEXT NOT NULL,
                    model_profile TEXT,
                    tool_calls TEXT,
                    tool_call_id TEXT,
                    reasoning TEXT,
                    created_at TEXT NOT NULL
                );

                INSERT INTO messages (id, conversation_id, role, content, model_profile, created_at)
                SELECT id, conversation_id, role, content, model_profile, created_at FROM messages_old;

                DROP TABLE messages_old;

                CREATE INDEX IF NOT EXISTS messages_conversation_id_idx
                    ON messages(conversation_id, id);
                """
            )
            columns = {row[1] for row in connection.execute("PRAGMA table_info(messages)")}

        if "reasoning" not in columns:
            print("Adding the reasoning column to the messages table ...")
            connection.execute("ALTER TABLE messages ADD COLUMN reasoning TEXT")

        conversation_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(conversations)")
        }
        if "memory_enabled" not in conversation_columns:
            print("Adding the memory_enabled column to the conversations table ...")
            # A fresh column with its own CHECK, not widening one on an existing
            # column -- allowed directly via ADD COLUMN since the default (0)
            # already satisfies the CHECK, unlike the messages.role rebuild above.
            connection.execute(
                """
                ALTER TABLE conversations ADD COLUMN memory_enabled INTEGER NOT NULL
                    DEFAULT 0 CHECK (memory_enabled IN (0, 1))
                """
            )

        # Empty (not missing) if the table doesn't exist yet at all -- then
        # database.py's own CREATE TABLE IF NOT EXISTS will create it with
        # every current column the first time the app runs, nothing to do here.
        context_run_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(message_context_runs)")
        }
        if context_run_columns and "real_peak_total_tokens" not in context_run_columns:
            print("Adding the real_peak_total_tokens column to message_context_runs ...")
            connection.execute(
                "ALTER TABLE message_context_runs ADD COLUMN real_peak_total_tokens INTEGER"
            )

        if "rag_enabled" in conversation_columns:
            print(
                "Dropping conversations.rag_enabled -- project-document retrieval is a tool "
                "now (search_documents), not an automatic per-conversation toggle ..."
            )
            connection.execute("ALTER TABLE conversations DROP COLUMN rag_enabled")

        if connection.execute(
            "SELECT name FROM sqlite_master WHERE name = 'message_adaptive_rag_runs'"
        ).fetchone():
            print(
                "Dropping message_adaptive_rag_runs -- there's no separate adaptive-RAG "
                "controller trace to persist any more; a search_documents tool call shows up "
                "in the normal tool-call trace instead ..."
            )
            connection.execute("DROP TABLE message_adaptive_rag_runs")

        connection.commit()
        message_count = connection.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        schema_version = connection.execute("PRAGMA user_version").fetchone()[0]
    print(f"Done. {database_path} is at schema {schema_version} ({message_count} messages kept).")


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DATABASE_PATH
    migrate(target)

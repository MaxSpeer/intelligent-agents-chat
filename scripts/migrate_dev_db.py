#!/usr/bin/env python3
"""Dev-only helper: bring an existing local chats.sqlite3 up to date with
database.py's current schema, without losing its chat history.

Not used by the app itself -- database.py only ever creates the current
schema fresh (no migrations there, by design). Run this by hand after
pulling a change that adds columns to the `messages` table, if you'd rather
keep your local chat history than let the app recreate the file from
scratch:

    uv run python scripts/migrate_dev_db.py [path-to-chats.sqlite3]

This script is written for the schema changes it currently knows about
(tool-calling support, a `reasoning` column, the `memory_enabled` column on
`conversations`, a `real_peak_total_tokens` column on
`message_context_runs`, moving model_profile/thinking_enabled/
memory_enabled off `conversations` and onto the single shared `app_settings`
row, then dropping `source_kind`/`content_hash` from `memory_entries` --
stray leftovers from a schema shape database.py has never actually defined
on this branch, silently breaking every memory insert with a NOT NULL
violation once a database file had picked them up some other way, e.g. by
briefly running a different branch against the same file). If database.py's
schema changes again later, extend or replace the migration below to
match -- it's a one-off dev tool, not a general migration framework.
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
        # Only for a conversations table that still has the old, pre-move
        # per-conversation columns (see the model_profile move below) --
        # once those are gone, memory_enabled belongs on app_settings only,
        # and must never be silently added back here on a later rerun.
        if "model_profile" in conversation_columns and "memory_enabled" not in conversation_columns:
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
            conversation_columns.add("memory_enabled")

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

        conversation_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(conversations)")
        }
        if "model_profile" in conversation_columns:
            print(
                "Moving model_profile/thinking_enabled/memory_enabled off conversations "
                "onto the single shared app_settings row ..."
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS app_settings (
                    id TEXT PRIMARY KEY,
                    model_profile TEXT NOT NULL,
                    thinking_enabled INTEGER NOT NULL DEFAULT 0
                        CHECK (thinking_enabled IN (0, 1)),
                    memory_enabled INTEGER NOT NULL DEFAULT 0
                        CHECK (memory_enabled IN (0, 1)),
                    updated_at TEXT NOT NULL
                )
                """
            )
            # Seed it from whichever conversation was last active -- the
            # closest thing to "what the user had selected" that the old,
            # per-conversation schema recorded. Only if app_settings doesn't
            # already have a row (re-running this script must stay a no-op).
            seed = connection.execute(
                """
                SELECT model_profile, thinking_enabled, memory_enabled
                FROM conversations
                ORDER BY updated_at DESC
                LIMIT 1
                """
            ).fetchone()
            if seed is not None:
                connection.execute(
                    """
                    INSERT OR IGNORE INTO app_settings
                        (id, model_profile, thinking_enabled, memory_enabled, updated_at)
                    VALUES ('global', ?, ?, ?, datetime('now'))
                    """,
                    seed,
                )
            # SQLite can drop columns directly (3.35+) as long as they're not
            # part of an index/constraint referenced elsewhere -- these three
            # are plain columns, so no table rebuild needed here.
            connection.execute("ALTER TABLE conversations DROP COLUMN model_profile")
            connection.execute("ALTER TABLE conversations DROP COLUMN thinking_enabled")
            connection.execute("ALTER TABLE conversations DROP COLUMN memory_enabled")

        # Empty (not missing) if the table doesn't exist yet -- nothing to do.
        memory_entries_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(memory_entries)")
        }
        for stray_column in ("source_kind", "content_hash"):
            if stray_column in memory_entries_columns:
                print(
                    f"Dropping stray memory_entries.{stray_column} column -- "
                    "database.py never defines it, but a NOT NULL leftover here "
                    "(e.g. from briefly running a different branch against this "
                    "same file) makes every memory insert fail silently ..."
                )
                connection.execute(f"ALTER TABLE memory_entries DROP COLUMN {stray_column}")

        connection.commit()
        message_count = connection.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    print(f"Done. {database_path} now matches the current schema ({message_count} messages kept).")


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DATABASE_PATH
    migrate(target)

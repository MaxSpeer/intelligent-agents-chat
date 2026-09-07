#!/usr/bin/env python3
"""Explicitly initialize or migrate a local chat database.

The application runs the same idempotent migrations automatically at startup.
This helper is useful for checking a database before launching the UI:

    uv run python scripts/migrate_dev_db.py [path-to-chats.sqlite3]

This script is written for the schema changes it currently knows about
(tool-calling support, a `reasoning` column, the `memory_enabled` column on
`conversations`, a `real_peak_total_tokens` column on
`message_context_runs`, moving model_profile/thinking_enabled/
memory_enabled off `conversations` and onto the single shared `app_settings`
row, then dropping `rag_enabled` and `message_adaptive_rag_runs` now that
project-document retrieval is a tool instead of an automatic,
per-conversation-toggle pipeline, and finally dropping `memory_entries`'s
unused `source_kind`/`content_hash` columns -- source_kind was always the
same hardcoded literal and content_hash was never actually read back
anywhere (rebuild_conversation's upsert guard compares `content` itself),
so both were dead weight from the start; safe to drop outright since memory
entries are derived data the app rebuilds automatically on startup, then
rebuilding `document_chunks` without its own unused id/content_hash/
page_number/section columns, and finally rebuilding
`message_context_sources` without the five columns nothing ever read back).
If database.py's schema changes again later, extend or replace the
migration below to match -- it's a one-off dev tool, not a general
migration framework.
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

        # Empty (not missing) if the table doesn't exist yet -- nothing to do; a
        # fresh CREATE TABLE IF NOT EXISTS already matches the current schema.
        memory_entries_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(memory_entries)")
        }
        for stray_column in ("source_kind", "content_hash"):
            if stray_column in memory_entries_columns:
                print(
                    f"Dropping memory_entries.{stray_column} -- unused: source_kind "
                    "was always the same hardcoded literal and content_hash was "
                    "never actually read back anywhere (rebuild_conversation's "
                    "upsert guard compares `content` itself) ..."
                )
                connection.execute(f"ALTER TABLE memory_entries DROP COLUMN {stray_column}")

        # Same idea for document chunks: `id` (a content-derived hash) and
        # `content_hash` were never read back, and page_number/section only
        # ever fed the `locator` string that's stored right next to them.
        # Rebuilt rather than dropped column by column so row_id survives
        # exactly -- document_chunks_vec joins on it (see documents.py), so
        # keeping it means the existing embeddings stay valid and nothing
        # has to be re-embedded.
        chunk_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(document_chunks)")
        }
        if chunk_columns & {"id", "content_hash", "page_number", "section"}:
            print(
                "Rebuilding document_chunks without the unused id/content_hash/"
                "page_number/section columns (embeddings are kept as they are) ..."
            )
            connection.executescript(
                """
                ALTER TABLE document_chunks RENAME TO document_chunks_old;

                CREATE TABLE document_chunks (
                    row_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    ordinal INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    locator TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(document_id, ordinal)
                );

                INSERT INTO document_chunks (
                    row_id, document_id, project_id, ordinal, content, locator, created_at
                )
                SELECT row_id, document_id, project_id, ordinal, content, locator, created_at
                FROM document_chunks_old;

                DROP TABLE document_chunks_old;

                CREATE INDEX IF NOT EXISTS document_chunks_document_idx
                    ON document_chunks(document_id, ordinal);
                CREATE INDEX IF NOT EXISTS document_chunks_project_idx
                    ON document_chunks(project_id, document_id);
                """
            )

        # message_context_sources kept five columns nothing ever read back:
        # source_kind (always 'project_memory' since documents became a tool),
        # source_id/source_project_id/source_conversation_id, and score. What
        # remains is exactly what the answer's trace step shows. Rebuilt
        # rather than dropped column by column, since UNIQUE/index have to be
        # recreated anyway -- the rows themselves are carried over.
        context_source_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(message_context_sources)")
        }
        if context_source_columns & {
            "source_kind",
            "source_id",
            "source_project_id",
            "source_conversation_id",
            "score",
        }:
            print(
                "Rebuilding message_context_sources without the write-only "
                "source_kind/source_id/source_project_id/source_conversation_id/score "
                "columns (existing provenance rows are kept) ..."
            )
            connection.executescript(
                """
                ALTER TABLE message_context_sources RENAME TO message_context_sources_old;

                CREATE TABLE message_context_sources (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    assistant_message_id INTEGER NOT NULL
                        REFERENCES messages(id) ON DELETE CASCADE,
                    source_title TEXT NOT NULL,
                    source_locator TEXT NOT NULL,
                    source_excerpt TEXT NOT NULL DEFAULT '',
                    rank INTEGER NOT NULL,
                    token_estimate INTEGER NOT NULL,
                    UNIQUE (assistant_message_id, rank)
                );

                INSERT INTO message_context_sources (
                    id, assistant_message_id, source_title, source_locator,
                    source_excerpt, rank, token_estimate
                )
                SELECT id, assistant_message_id, source_title, source_locator,
                       source_excerpt, rank, token_estimate
                FROM message_context_sources_old;

                DROP TABLE message_context_sources_old;

                CREATE INDEX IF NOT EXISTS message_context_sources_message_idx
                    ON message_context_sources(assistant_message_id, rank);
                """
            )

        connection.commit()
        message_count = connection.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        schema_version = connection.execute("PRAGMA user_version").fetchone()[0]
    print(f"Done. {database_path} is at schema {schema_version} ({message_count} messages kept).")


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DATABASE_PATH
    migrate(target)

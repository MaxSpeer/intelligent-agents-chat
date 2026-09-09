"""SQLite persistence for projects, conversations, messages, and context provenance."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sqlite3
from typing import Literal
from uuid import uuid4

import sqlite_vec

from intelligent_agents_chat.logging_config import log_event


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATABASE_PATH = PROJECT_ROOT / ".data" / "chats.sqlite3"
DEFAULT_PROJECT_ID = "default"
DEFAULT_PROJECT_NAME = "General"
DEFAULT_CONVERSATION_TITLE = "New chat"
MessageRole = Literal["system", "user", "assistant", "tool"]
VALID_ROLES = {"system", "user", "assistant", "tool"}
logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Project:
    id: str
    name: str
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class Conversation:
    id: str
    project_id: str
    title: str
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class Message:
    id: int
    conversation_id: str
    role: MessageRole
    content: str
    model_profile: str | None
    created_at: datetime
    # Assistant tool calls: {"id", "name", "arguments"}.
    tool_calls: tuple[dict, ...] | None = None
    # Matching call ID for tool-result messages.
    tool_call_id: str | None = None
    # Assistant reasoning preceding this action.
    reasoning: str | None = None


@dataclass(frozen=True, slots=True)
class ContextSourceInput:
    """A retrieved source and citation metadata supplied to an assistant message."""

    source_title: str
    source_locator: str
    source_excerpt: str
    rank: int
    token_estimate: int


@dataclass(frozen=True, slots=True)
class MessageContextSource:
    """Persisted provenance for context supplied to one assistant message."""

    id: int
    assistant_message_id: int
    source_title: str
    source_locator: str
    source_excerpt: str
    rank: int
    token_estimate: int


@dataclass(frozen=True, slots=True)
class ContextRunInput:
    """Estimated input budget and optional server-reported usage for the context inspector."""

    context_window_tokens: int
    input_budget_tokens: int
    estimated_input_tokens: int
    real_prompt_tokens: int | None
    real_completion_tokens: int | None
    # Last reported round's prompt-plus-completion total.
    real_peak_total_tokens: int | None


@dataclass(frozen=True, slots=True)
class MessageContextRun:
    """Persisted token-budget outcome for one assistant message."""

    assistant_message_id: int
    context_window_tokens: int
    input_budget_tokens: int
    estimated_input_tokens: int
    real_prompt_tokens: int | None
    real_completion_tokens: int | None
    real_peak_total_tokens: int | None
    created_at: datetime


@dataclass(frozen=True, slots=True)
class AppSettings:
    """App-wide model, thinking, and memory preferences stored in a single row."""

    model_profile: str
    thinking_enabled: bool
    memory_enabled: bool
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class ConversationCompaction:
    """A conversation's rolling summary and the last message it covers."""

    conversation_id: str
    compacted_through_message_id: int
    summary: str
    created_at: datetime
    updated_at: datetime


class ChatRepository:
    """Small synchronous repository using short-lived SQLite connections."""

    def __init__(self, database_path: Path = DEFAULT_DATABASE_PATH) -> None:
        self.database_path = database_path

    def initialize(self) -> None:
        """Create the schema and the first project if they do not exist."""
        log_event(
            logger,
            logging.INFO,
            "database.initialize.started",
            database_path=str(self.database_path),
        )
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            journal_mode = str(connection.execute("PRAGMA journal_mode = WAL").fetchone()[0])
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                -- Generation preferences (model, thinking, memory) shared by
                -- the whole app -- a single row, keyed by the fixed id
                -- 'global' (see APP_SETTINGS_ID/get_app_settings). Not
                -- per-conversation: switching or starting a chat should never
                -- reset what the user picked.
                CREATE TABLE IF NOT EXISTS app_settings (
                    id TEXT PRIMARY KEY,
                    model_profile TEXT NOT NULL,
                    thinking_enabled INTEGER NOT NULL DEFAULT 0
                        CHECK (thinking_enabled IN (0, 1)),
                    memory_enabled INTEGER NOT NULL DEFAULT 0
                        CHECK (memory_enabled IN (0, 1)),
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS messages (
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

                CREATE INDEX IF NOT EXISTS conversations_project_updated_idx
                    ON conversations(project_id, updated_at DESC);
                CREATE INDEX IF NOT EXISTS messages_conversation_id_idx
                    ON messages(conversation_id, id);
                """
            )
            messages_schema_migrated = _migrate_messages_schema(connection)
            if not _column_exists(connection, "conversations", "memory_enabled"):
                connection.execute(
                    """
                    ALTER TABLE conversations
                    ADD COLUMN memory_enabled INTEGER NOT NULL DEFAULT 0
                        CHECK (memory_enabled IN (0, 1))
                    """
                )
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS memory_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    source_conversation_id TEXT NOT NULL
                        REFERENCES conversations(id) ON DELETE CASCADE,
                    source_message_start_id INTEGER NOT NULL,
                    source_message_end_id INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    enabled INTEGER NOT NULL DEFAULT 1 CHECK (enabled IN (0, 1)),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE (
                        source_conversation_id,
                        source_message_start_id,
                        source_message_end_id
                    )
                );

                CREATE INDEX IF NOT EXISTS memory_entries_project_idx
                    ON memory_entries(project_id, enabled, updated_at DESC);
                CREATE INDEX IF NOT EXISTS memory_entries_conversation_idx
                    ON memory_entries(source_conversation_id, source_message_start_id);

                CREATE VIRTUAL TABLE IF NOT EXISTS memory_entries_fts USING fts5(
                    content,
                    content='memory_entries',
                    content_rowid='id',
                    tokenize='unicode61 remove_diacritics 2'
                );

                CREATE TRIGGER IF NOT EXISTS memory_entries_after_insert
                AFTER INSERT ON memory_entries WHEN new.enabled = 1 BEGIN
                    INSERT INTO memory_entries_fts(rowid, content)
                    VALUES (new.id, new.content);
                END;

                CREATE TRIGGER IF NOT EXISTS memory_entries_after_delete
                AFTER DELETE ON memory_entries WHEN old.enabled = 1 BEGIN
                    INSERT INTO memory_entries_fts(memory_entries_fts, rowid, content)
                    VALUES ('delete', old.id, old.content);
                END;

                CREATE TRIGGER IF NOT EXISTS memory_entries_after_update
                AFTER UPDATE OF content, enabled ON memory_entries BEGIN
                    INSERT INTO memory_entries_fts(memory_entries_fts, rowid, content)
                    SELECT 'delete', old.id, old.content WHERE old.enabled = 1;
                    INSERT INTO memory_entries_fts(rowid, content)
                    SELECT new.id, new.content WHERE new.enabled = 1;
                END;

                CREATE TABLE IF NOT EXISTS message_context_sources (
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

                CREATE INDEX IF NOT EXISTS message_context_sources_message_idx
                    ON message_context_sources(assistant_message_id, rank);

                CREATE TABLE IF NOT EXISTS message_context_runs (
                    assistant_message_id INTEGER PRIMARY KEY
                        REFERENCES messages(id) ON DELETE CASCADE,
                    context_window_tokens INTEGER NOT NULL,
                    input_budget_tokens INTEGER NOT NULL,
                    estimated_input_tokens INTEGER NOT NULL,
                    real_prompt_tokens INTEGER,
                    real_completion_tokens INTEGER,
                    real_peak_total_tokens INTEGER,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS conversation_compactions (
                    conversation_id TEXT PRIMARY KEY
                        REFERENCES conversations(id) ON DELETE CASCADE,
                    compacted_through_message_id INTEGER NOT NULL,
                    summary TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    display_name TEXT NOT NULL,
                    storage_key TEXT NOT NULL UNIQUE,
                    media_type TEXT NOT NULL,
                    extension TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    byte_size INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    error_message TEXT,
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    embedding_model TEXT,
                    embedding_dimension INTEGER,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    indexed_at TEXT,
                    UNIQUE(project_id, sha256)
                );

                CREATE INDEX IF NOT EXISTS documents_project_updated_idx
                    ON documents(project_id, updated_at DESC);
                CREATE INDEX IF NOT EXISTS documents_project_status_idx
                    ON documents(project_id, status);

                CREATE TABLE IF NOT EXISTS document_chunks (
                    row_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    ordinal INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    locator TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(document_id, ordinal)
                );

                CREATE INDEX IF NOT EXISTS document_chunks_document_idx
                    ON document_chunks(document_id, ordinal);
                CREATE INDEX IF NOT EXISTS document_chunks_project_idx
                    ON document_chunks(project_id, document_id);
                """
            )
            # vec0 requires a literal vector width and manual row_id synchronization with chunks.
            # Import the dimension here to avoid a circular import; model changes require migration.
            from intelligent_agents_chat.embeddings import EMBEDDING_DIMENSION

            connection.execute(
                f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS document_chunks_vec USING vec0(
                    row_id INTEGER PRIMARY KEY,
                    project_id TEXT PARTITION KEY,
                    embedding FLOAT[{EMBEDDING_DIMENSION}] DISTANCE_METRIC=COSINE
                )
                """
            )
            _migrate_context_run_schema(connection)
            if not _column_exists(connection, "message_context_sources", "source_excerpt"):
                connection.execute(
                    """
                    ALTER TABLE message_context_sources
                    ADD COLUMN source_excerpt TEXT NOT NULL DEFAULT ''
                    """
                )
            connection.execute("PRAGMA user_version = 6")
            now = _timestamp()
            connection.execute(
                """
                INSERT OR IGNORE INTO projects (id, name, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (DEFAULT_PROJECT_ID, DEFAULT_PROJECT_NAME, now, now),
            )
        log_event(
            logger,
            logging.INFO,
            "database.initialize.completed",
            database_path=str(self.database_path),
            sqlite_version=sqlite3.sqlite_version,
            journal_mode=journal_mode,
            schema_version=6,
            messages_schema_migrated=messages_schema_migrated,
        )

    def get_project(self, project_id: str = DEFAULT_PROJECT_ID) -> Project | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT id, name, created_at, updated_at FROM projects WHERE id = ?",
                (project_id,),
            ).fetchone()
        return _project_from_row(row) if row else None

    def list_projects(self) -> list[Project]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, name, created_at, updated_at
                FROM projects
                ORDER BY
                    CASE WHEN id = ? THEN 0 ELSE 1 END,
                    name COLLATE NOCASE,
                    created_at
                """,
                (DEFAULT_PROJECT_ID,),
            ).fetchall()
        projects = [_project_from_row(row) for row in rows]
        log_event(
            logger,
            logging.DEBUG,
            "database.projects.listed",
            project_count=len(projects),
        )
        return projects

    def create_project(self, name: str) -> Project:
        clean_name = " ".join(name.split())
        if not clean_name:
            raise ValueError("Project name cannot be empty")

        project_id = str(uuid4())
        now = _timestamp()
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT 1 FROM projects WHERE name = ? COLLATE NOCASE",
                (clean_name,),
            ).fetchone()
            if existing is not None:
                raise ValueError("Project name already exists")
            connection.execute(
                """
                INSERT INTO projects (id, name, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (project_id, clean_name, now, now),
            )

        project = self.get_project(project_id)
        if project is None:
            raise RuntimeError("Failed to read the project after creating it")
        log_event(
            logger,
            logging.INFO,
            "database.project.created",
            project_id=project.id,
            project_name_chars=len(project.name),
        )
        return project

    def create_conversation(
        self,
        *,
        project_id: str = DEFAULT_PROJECT_ID,
        title: str = DEFAULT_CONVERSATION_TITLE,
    ) -> Conversation:
        conversation_id = str(uuid4())
        now = _timestamp()
        clean_title = title.strip()
        if not clean_title:
            raise ValueError("Conversation title cannot be empty")

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO conversations (id, project_id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (conversation_id, project_id, clean_title, now, now),
            )
        conversation = self.get_conversation(conversation_id)
        if conversation is None:
            raise RuntimeError("Failed to read the conversation after creating it")
        log_event(
            logger,
            logging.INFO,
            "database.conversation.created",
            project_id=conversation.project_id,
            conversation_id=conversation.id,
            title_chars=len(conversation.title),
        )
        return conversation

    def get_conversation(self, conversation_id: str) -> Conversation | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, project_id, title, created_at, updated_at
                FROM conversations
                WHERE id = ?
                """,
                (conversation_id,),
            ).fetchone()
        return _conversation_from_row(row) if row else None

    def list_conversations(self, project_id: str = DEFAULT_PROJECT_ID) -> list[Conversation]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, project_id, title, created_at, updated_at
                FROM conversations
                WHERE project_id = ?
                ORDER BY updated_at DESC, created_at DESC
                """,
                (project_id,),
            ).fetchall()
        conversations = [_conversation_from_row(row) for row in rows]
        log_event(
            logger,
            logging.DEBUG,
            "database.conversations.listed",
            project_id=project_id,
            conversation_count=len(conversations),
        )
        return conversations

    def rename_conversation(self, conversation_id: str, title: str) -> bool:
        clean_title = title.strip()
        if not clean_title:
            raise ValueError("Conversation title cannot be empty")
        now = _timestamp()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE conversations
                SET title = ?, updated_at = ?
                WHERE id = ?
                """,
                (clean_title, now, conversation_id),
            )
        renamed = cursor.rowcount == 1
        log_event(
            logger,
            logging.INFO,
            "database.conversation.renamed",
            conversation_id=conversation_id,
            title_chars=len(clean_title),
            updated=renamed,
        )
        return renamed

    # Fixed ID for the single app-wide settings row.
    APP_SETTINGS_ID = "global"

    def get_app_settings(self) -> AppSettings | None:
        """Return app-wide settings, or None if they have not been saved."""
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT model_profile, thinking_enabled, memory_enabled, updated_at
                FROM app_settings
                WHERE id = ?
                """,
                (self.APP_SETTINGS_ID,),
            ).fetchone()
        return _app_settings_from_row(row) if row is not None else None

    def set_app_settings(
        self, *, model_profile: str, thinking_enabled: bool, memory_enabled: bool
    ) -> AppSettings:
        """Create or update the single app-wide settings row."""
        if not model_profile.strip():
            raise ValueError("Model profile cannot be empty")
        now = _timestamp()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO app_settings (id, model_profile, thinking_enabled, memory_enabled, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT (id) DO UPDATE SET
                    model_profile = excluded.model_profile,
                    thinking_enabled = excluded.thinking_enabled,
                    memory_enabled = excluded.memory_enabled,
                    updated_at = excluded.updated_at
                """,
                (self.APP_SETTINGS_ID, model_profile, int(thinking_enabled), int(memory_enabled), now),
            )
        settings = self.get_app_settings()
        if settings is None:
            raise RuntimeError("Failed to read app settings after writing them")
        log_event(
            logger,
            logging.INFO,
            "database.app_settings.changed",
            model_profile=settings.model_profile,
            thinking_enabled=settings.thinking_enabled,
            memory_enabled=settings.memory_enabled,
        )
        return settings

    def delete_conversation(self, conversation_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM conversations WHERE id = ?",
                (conversation_id,),
            )
        deleted = cursor.rowcount == 1
        log_event(
            logger,
            logging.INFO,
            "database.conversation.deleted",
            conversation_id=conversation_id,
            deleted=deleted,
        )
        return deleted

    def add_message(
        self,
        conversation_id: str,
        role: MessageRole,
        content: str,
        *,
        model_profile: str | None = None,
        tool_calls: Sequence[dict] | None = None,
        tool_call_id: str | None = None,
        reasoning: str | None = None,
    ) -> Message:
        if role not in VALID_ROLES:
            raise ValueError(f"Unsupported message role: {role}")
        # Assistant messages may have empty content when tool calls or reasoning are present.
        if not content.strip() and not tool_calls and not reasoning:
            raise ValueError("Message content cannot be empty")

        now = _timestamp()
        tool_calls_json = json.dumps(list(tool_calls)) if tool_calls else None
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO messages (
                    conversation_id, role, content, model_profile,
                    tool_calls, tool_call_id, reasoning, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    conversation_id,
                    role,
                    content,
                    model_profile,
                    tool_calls_json,
                    tool_call_id,
                    reasoning,
                    now,
                ),
            )
            connection.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (now, conversation_id),
            )
            message_id = cursor.lastrowid
        if message_id is None:
            raise RuntimeError("Failed to obtain the new message ID")
        message = self.get_message(message_id)
        if message is None:
            raise RuntimeError("Failed to read the message after creating it")
        log_event(
            logger,
            logging.INFO,
            "database.message.created",
            message_id=message.id,
            conversation_id=message.conversation_id,
            role=message.role,
            message_chars=len(message.content),
            model_profile=message.model_profile,
            tool_call_count=len(message.tool_calls) if message.tool_calls else 0,
            reasoning_chars=len(message.reasoning) if message.reasoning else 0,
        )
        return message

    def get_message(self, message_id: int) -> Message | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, conversation_id, role, content, model_profile,
                       tool_calls, tool_call_id, reasoning, created_at
                FROM messages
                WHERE id = ?
                """,
                (message_id,),
            ).fetchone()
        return _message_from_row(row) if row else None

    def list_messages(self, conversation_id: str) -> list[Message]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, conversation_id, role, content, model_profile,
                       tool_calls, tool_call_id, reasoning, created_at
                FROM messages
                WHERE conversation_id = ?
                ORDER BY id ASC
                """,
                (conversation_id,),
            ).fetchall()
        messages = [_message_from_row(row) for row in rows]
        log_event(
            logger,
            logging.DEBUG,
            "database.messages.listed",
            conversation_id=conversation_id,
            message_count=len(messages),
        )
        return messages

    def add_message_context_sources(
        self,
        assistant_message_id: int,
        sources: list[ContextSourceInput],
    ) -> None:
        if not sources:
            return
        with self._connect() as connection:
            connection.executemany(
                """
                INSERT INTO message_context_sources (
                    assistant_message_id, source_title, source_locator, source_excerpt,
                    rank, token_estimate
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        assistant_message_id,
                        source.source_title,
                        source.source_locator,
                        source.source_excerpt,
                        source.rank,
                        source.token_estimate,
                    )
                    for source in sources
                ],
            )
        log_event(
            logger,
            logging.INFO,
            "database.message_context_sources.created",
            assistant_message_id=assistant_message_id,
            source_count=len(sources),
        )

    def list_message_context_sources(self, assistant_message_id: int) -> list[MessageContextSource]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, assistant_message_id, source_title, source_locator,
                       source_excerpt, rank, token_estimate
                FROM message_context_sources
                WHERE assistant_message_id = ?
                ORDER BY rank ASC
                """,
                (assistant_message_id,),
            ).fetchall()
        return [_message_context_source_from_row(row) for row in rows]

    def add_message_context_run(self, assistant_message_id: int, run: ContextRunInput) -> None:
        now = _timestamp()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO message_context_runs (
                    assistant_message_id, context_window_tokens, input_budget_tokens,
                    estimated_input_tokens, real_prompt_tokens, real_completion_tokens,
                    real_peak_total_tokens, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    assistant_message_id,
                    run.context_window_tokens,
                    run.input_budget_tokens,
                    run.estimated_input_tokens,
                    run.real_prompt_tokens,
                    run.real_completion_tokens,
                    run.real_peak_total_tokens,
                    now,
                ),
            )
        log_event(
            logger,
            logging.DEBUG,
            "database.message_context_run.created",
            assistant_message_id=assistant_message_id,
            estimated_input_tokens=run.estimated_input_tokens,
            real_prompt_tokens=run.real_prompt_tokens,
            real_completion_tokens=run.real_completion_tokens,
            real_peak_total_tokens=run.real_peak_total_tokens,
        )

    def get_message_context_run(self, assistant_message_id: int) -> MessageContextRun | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT assistant_message_id, context_window_tokens, input_budget_tokens,
                       estimated_input_tokens, real_prompt_tokens, real_completion_tokens,
                       real_peak_total_tokens, created_at
                FROM message_context_runs
                WHERE assistant_message_id = ?
                """,
                (assistant_message_id,),
            ).fetchone()
        return _message_context_run_from_row(row) if row is not None else None

    def get_conversation_compaction(self, conversation_id: str) -> ConversationCompaction | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT conversation_id, compacted_through_message_id, summary,
                       created_at, updated_at
                FROM conversation_compactions
                WHERE conversation_id = ?
                """,
                (conversation_id,),
            ).fetchone()
        return _conversation_compaction_from_row(row) if row is not None else None

    def set_conversation_compaction(
        self,
        conversation_id: str,
        *,
        compacted_through_message_id: int,
        summary: str,
    ) -> None:
        """Create or update the conversation's summary and compaction boundary."""
        now = _timestamp()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO conversation_compactions (
                    conversation_id, compacted_through_message_id, summary,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT (conversation_id) DO UPDATE SET
                    compacted_through_message_id = excluded.compacted_through_message_id,
                    summary = excluded.summary,
                    updated_at = excluded.updated_at
                """,
                (conversation_id, compacted_through_message_id, summary, now, now),
            )
        log_event(
            logger,
            logging.INFO,
            "database.conversation_compaction.updated",
            conversation_id=conversation_id,
            compacted_through_message_id=compacted_through_message_id,
            summary_chars=len(summary),
        )

    def _connect(self) -> sqlite3.Connection:
        return connect_database(self.database_path)


def connect_database(database_path: Path) -> sqlite3.Connection:
    """Open a SQLite connection with sqlite-vec, foreign keys, and a busy timeout."""
    connection = sqlite3.connect(database_path, timeout=5.0)
    connection.row_factory = sqlite3.Row
    connection.enable_load_extension(True)
    sqlite_vec.load(connection)
    connection.enable_load_extension(False)
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 5000")
    return connection


def _column_exists(connection: sqlite3.Connection, table: str, column: str) -> bool:
    return any(row[1] == column for row in connection.execute(f"PRAGMA table_info({table})"))


def _migrate_context_run_schema(connection: sqlite3.Connection) -> bool:
    """Ensure the context-usage table has the required columns, preserving usage data."""
    desired_columns = (
        "assistant_message_id",
        "context_window_tokens",
        "input_budget_tokens",
        "estimated_input_tokens",
        "real_prompt_tokens",
        "real_completion_tokens",
        "real_peak_total_tokens",
        "created_at",
    )
    existing_columns = {
        str(row[1]) for row in connection.execute("PRAGMA table_info(message_context_runs)")
    }
    if existing_columns == set(desired_columns):
        return False

    select_expressions = {
        "assistant_message_id": "assistant_message_id",
        "context_window_tokens": "context_window_tokens",
        "input_budget_tokens": "input_budget_tokens",
        "estimated_input_tokens": "estimated_input_tokens",
        "real_prompt_tokens": (
            "real_prompt_tokens" if "real_prompt_tokens" in existing_columns else "NULL"
        ),
        "real_completion_tokens": (
            "real_completion_tokens" if "real_completion_tokens" in existing_columns else "NULL"
        ),
        "real_peak_total_tokens": (
            "real_peak_total_tokens" if "real_peak_total_tokens" in existing_columns else "NULL"
        ),
        "created_at": "created_at",
    }
    connection.execute("DROP TABLE IF EXISTS message_context_runs_migrated")
    connection.execute(
        """
        CREATE TABLE message_context_runs_migrated (
            assistant_message_id INTEGER PRIMARY KEY
                REFERENCES messages(id) ON DELETE CASCADE,
            context_window_tokens INTEGER NOT NULL,
            input_budget_tokens INTEGER NOT NULL,
            estimated_input_tokens INTEGER NOT NULL,
            real_prompt_tokens INTEGER,
            real_completion_tokens INTEGER,
            real_peak_total_tokens INTEGER,
            created_at TEXT NOT NULL
        )
        """
    )
    connection.execute(
        f"""
        INSERT INTO message_context_runs_migrated ({", ".join(desired_columns)})
        SELECT {", ".join(select_expressions[column] for column in desired_columns)}
        FROM message_context_runs
        """
    )
    connection.execute("DROP TABLE message_context_runs")
    connection.execute("ALTER TABLE message_context_runs_migrated RENAME TO message_context_runs")
    return True


def _migrate_messages_schema(connection: sqlite3.Connection) -> bool:
    """Ensure tool-call and reasoning support while preserving messages and provenance."""
    columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(messages)")}
    table_row = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'messages'"
    ).fetchone()
    table_sql = str(table_row[0]) if table_row and table_row[0] else ""
    needs_rebuild = not {"tool_calls", "tool_call_id"} <= columns or "'tool'" not in table_sql

    if not needs_rebuild:
        if "reasoning" not in columns:
            connection.execute("ALTER TABLE messages ADD COLUMN reasoning TEXT")
            return True
        return False

    copy_columns = [
        column
        for column in (
            "id",
            "conversation_id",
            "role",
            "content",
            "model_profile",
            "tool_calls",
            "tool_call_id",
            "reasoning",
            "created_at",
        )
        if column in columns
    ]
    copy_column_sql = ", ".join(copy_columns)
    foreign_keys_enabled = bool(connection.execute("PRAGMA foreign_keys").fetchone()[0])
    connection.commit()
    if foreign_keys_enabled:
        connection.execute("PRAGMA foreign_keys = OFF")
    try:
        connection.executescript(
            f"""
            BEGIN IMMEDIATE;

            DROP TABLE IF EXISTS messages_migrated;

            CREATE TABLE messages_migrated (
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

            INSERT INTO messages_migrated ({copy_column_sql})
            SELECT {copy_column_sql} FROM messages;

            DROP TABLE messages;
            ALTER TABLE messages_migrated RENAME TO messages;

            CREATE INDEX messages_conversation_id_idx
                ON messages(conversation_id, id);

            COMMIT;
            """
        )
    except Exception:
        connection.rollback()
        raise
    finally:
        if foreign_keys_enabled:
            connection.execute("PRAGMA foreign_keys = ON")

    violations = connection.execute("PRAGMA foreign_key_check").fetchall()
    if violations:
        raise sqlite3.IntegrityError(
            f"Message schema migration left {len(violations)} foreign-key violations"
        )
    return True


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _project_from_row(row: sqlite3.Row) -> Project:
    return Project(
        id=row["id"],
        name=row["name"],
        created_at=_parse_datetime(row["created_at"]),
        updated_at=_parse_datetime(row["updated_at"]),
    )


def _conversation_from_row(row: sqlite3.Row) -> Conversation:
    return Conversation(
        id=row["id"],
        project_id=row["project_id"],
        title=row["title"],
        created_at=_parse_datetime(row["created_at"]),
        updated_at=_parse_datetime(row["updated_at"]),
    )


def _app_settings_from_row(row: sqlite3.Row) -> AppSettings:
    return AppSettings(
        model_profile=row["model_profile"],
        thinking_enabled=bool(row["thinking_enabled"]),
        memory_enabled=bool(row["memory_enabled"]),
        updated_at=_parse_datetime(row["updated_at"]),
    )


def _message_from_row(row: sqlite3.Row) -> Message:
    tool_calls_json = row["tool_calls"]
    return Message(
        id=row["id"],
        conversation_id=row["conversation_id"],
        role=row["role"],
        content=row["content"],
        model_profile=row["model_profile"],
        created_at=_parse_datetime(row["created_at"]),
        tool_calls=tuple(json.loads(tool_calls_json)) if tool_calls_json else None,
        tool_call_id=row["tool_call_id"],
        reasoning=row["reasoning"],
    )


def _message_context_source_from_row(row: sqlite3.Row) -> MessageContextSource:
    return MessageContextSource(
        id=row["id"],
        assistant_message_id=row["assistant_message_id"],
        source_title=row["source_title"],
        source_locator=row["source_locator"],
        source_excerpt=row["source_excerpt"],
        rank=row["rank"],
        token_estimate=row["token_estimate"],
    )


def _message_context_run_from_row(row: sqlite3.Row) -> MessageContextRun:
    return MessageContextRun(
        assistant_message_id=row["assistant_message_id"],
        context_window_tokens=row["context_window_tokens"],
        input_budget_tokens=row["input_budget_tokens"],
        estimated_input_tokens=row["estimated_input_tokens"],
        real_prompt_tokens=row["real_prompt_tokens"],
        real_completion_tokens=row["real_completion_tokens"],
        real_peak_total_tokens=row["real_peak_total_tokens"],
        created_at=datetime.fromisoformat(row["created_at"]),
    )


def _conversation_compaction_from_row(row: sqlite3.Row) -> ConversationCompaction:
    return ConversationCompaction(
        conversation_id=row["conversation_id"],
        compacted_through_message_id=row["compacted_through_message_id"],
        summary=row["summary"],
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
    )


# Constructing the shared repository does not open or initialize the database.
repository = ChatRepository()

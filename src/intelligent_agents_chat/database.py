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
    model_profile: str
    thinking_enabled: bool
    memory_enabled: bool
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
    # Set only on assistant messages that made tool calls: [{"id", "name", "arguments"}, ...].
    tool_calls: tuple[dict, ...] | None = None
    # Set only on role="tool" messages: which tool_calls entry this is the result of.
    tool_call_id: str | None = None
    # Set only on assistant messages that reasoned before this action
    reasoning: str | None = None


@dataclass(frozen=True, slots=True)
class ContextSourceInput:
    """One retrieval source supplied while generating an assistant message."""

    source_kind: str
    source_id: str
    source_project_id: str
    source_conversation_id: str | None
    source_title: str
    source_locator: str
    source_excerpt: str
    rank: int
    score: float
    token_estimate: int


@dataclass(frozen=True, slots=True)
class MessageContextSource:
    """Persisted provenance for context supplied to one assistant message."""

    id: int
    assistant_message_id: int
    source_kind: str
    source_id: str
    source_project_id: str
    source_conversation_id: str | None
    source_title: str
    source_locator: str
    source_excerpt: str
    rank: int
    score: float
    token_estimate: int


@dataclass(frozen=True, slots=True)
class ContextRunInput:
    """The token-budget outcome of assembling one request, for the context
    inspector -- estimated (chars/3, from ContextAssembler) alongside real
    (from the model server's own usage report, may be unavailable).
    """

    context_window_tokens: int
    input_budget_tokens: int
    estimated_input_tokens: int
    real_prompt_tokens: int | None
    real_completion_tokens: int | None
    # The turn's peak prompt+completion total (its last round -- see
    # chat.py's UsageEvent docstring for why that's always the peak), i.e.
    # the most the model ever held in context at once this turn. Compare
    # against context_window_tokens to see how close a turn came to actually
    # running out mid-turn.
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
class ConversationCompaction:
    """A rolling summary standing in for everything in one conversation older
    than its most recent kept-raw turns (see chat.py's
    COMPACTION_KEEP_RECENT_TURNS) -- one row per conversation, extended in
    place rather than re-summarized from scratch as more history ages out.
    """

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
                    model_profile TEXT NOT NULL,
                    thinking_enabled INTEGER NOT NULL DEFAULT 0
                        CHECK (thinking_enabled IN (0, 1)),
                    memory_enabled INTEGER NOT NULL DEFAULT 0
                        CHECK (memory_enabled IN (0, 1)),
                    created_at TEXT NOT NULL,
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
                    source_kind TEXT NOT NULL DEFAULT 'conversation_turn',
                    content TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
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
                    source_kind TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    source_project_id TEXT NOT NULL,
                    source_conversation_id TEXT,
                    source_title TEXT NOT NULL,
                    source_locator TEXT NOT NULL,
                    source_excerpt TEXT NOT NULL DEFAULT '',
                    rank INTEGER NOT NULL,
                    score REAL NOT NULL,
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
        model_profile: str,
        *,
        project_id: str = DEFAULT_PROJECT_ID,
        title: str = DEFAULT_CONVERSATION_TITLE,
        thinking_enabled: bool = False,
        memory_enabled: bool = False,
    ) -> Conversation:
        conversation_id = str(uuid4())
        now = _timestamp()
        clean_title = title.strip()
        if not clean_title:
            raise ValueError("Conversation title cannot be empty")
        if not model_profile.strip():
            raise ValueError("Model profile cannot be empty")

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO conversations (
                    id, project_id, title, model_profile, thinking_enabled, memory_enabled,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    conversation_id,
                    project_id,
                    clean_title,
                    model_profile,
                    int(thinking_enabled),
                    int(memory_enabled),
                    now,
                    now,
                ),
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
            model_profile=conversation.model_profile,
            thinking_enabled=conversation.thinking_enabled,
            memory_enabled=conversation.memory_enabled,
            title_chars=len(conversation.title),
        )
        return conversation

    def get_conversation(self, conversation_id: str) -> Conversation | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, project_id, title, model_profile, thinking_enabled, memory_enabled,
                       created_at, updated_at
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
                SELECT id, project_id, title, model_profile, thinking_enabled, memory_enabled,
                       created_at, updated_at
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

    def set_model_profile(self, conversation_id: str, model_profile: str) -> bool:
        if not model_profile.strip():
            raise ValueError("Model profile cannot be empty")
        now = _timestamp()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE conversations
                SET model_profile = ?, updated_at = ?
                WHERE id = ?
                """,
                (model_profile, now, conversation_id),
            )
        updated = cursor.rowcount == 1
        log_event(
            logger,
            logging.INFO,
            "database.conversation.model_changed",
            conversation_id=conversation_id,
            model_profile=model_profile,
            updated=updated,
        )
        return updated

    def set_thinking_enabled(self, conversation_id: str, enabled: bool) -> bool:
        now = _timestamp()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE conversations
                SET thinking_enabled = ?, updated_at = ?
                WHERE id = ?
                """,
                (int(enabled), now, conversation_id),
            )
        updated = cursor.rowcount == 1
        log_event(
            logger,
            logging.INFO,
            "database.conversation.thinking_changed",
            conversation_id=conversation_id,
            thinking_enabled=enabled,
            updated=updated,
        )
        return updated

    def set_memory_enabled(self, conversation_id: str, enabled: bool) -> bool:
        now = _timestamp()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE conversations
                SET memory_enabled = ?, updated_at = ?
                WHERE id = ?
                """,
                (int(enabled), now, conversation_id),
            )
        updated = cursor.rowcount == 1
        log_event(
            logger,
            logging.INFO,
            "database.conversation.memory_changed",
            conversation_id=conversation_id,
            memory_enabled=enabled,
            updated=updated,
        )
        return updated

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
        # An assistant message that only made tool calls (or only reasoned, e.g. a
        # round that got stopped mid-thought) has no natural-language content --
        # that's valid as long as it has tool calls or reasoning attached.
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
                    assistant_message_id, source_kind, source_id, source_project_id,
                    source_conversation_id, source_title, source_locator, source_excerpt,
                    rank, score, token_estimate
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        assistant_message_id,
                        source.source_kind,
                        source.source_id,
                        source.source_project_id,
                        source.source_conversation_id,
                        source.source_title,
                        source.source_locator,
                        source.source_excerpt,
                        source.rank,
                        source.score,
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
            source_kinds=sorted({source.source_kind for source in sources}),
        )

    def list_message_context_sources(self, assistant_message_id: int) -> list[MessageContextSource]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, assistant_message_id, source_kind, source_id, source_project_id,
                       source_conversation_id, source_title, source_locator, rank, score,
                       source_excerpt, token_estimate
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
        """Create or extend the one compaction row for a conversation --
        never a second row, since there's only ever one current boundary.
        """
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
    """Open one consistently configured SQLite connection for repository services.

    Loads the sqlite-vec extension on every connection -- it's the one
    shared factory used by ChatRepository, DocumentStore, and
    ProjectMemoryStore, so this is the one place that guarantees
    document_chunks_vec (see documents.py) is queryable regardless of which
    of those opens the connection.
    """
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
    """Replace the older detailed inspector table with PR 14's usage table."""
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
    """Upgrade pre-tool-calling databases without losing messages or provenance."""
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
        model_profile=row["model_profile"],
        thinking_enabled=bool(row["thinking_enabled"]),
        memory_enabled=bool(row["memory_enabled"]),
        created_at=_parse_datetime(row["created_at"]),
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
        source_kind=row["source_kind"],
        source_id=row["source_id"],
        source_project_id=row["source_project_id"],
        source_conversation_id=row["source_conversation_id"],
        source_title=row["source_title"],
        source_locator=row["source_locator"],
        source_excerpt=row["source_excerpt"],
        rank=row["rank"],
        score=row["score"],
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

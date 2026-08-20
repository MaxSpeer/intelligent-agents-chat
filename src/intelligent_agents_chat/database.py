"""SQLite persistence for projects, conversations, and messages."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
from pathlib import Path
import sqlite3
from typing import Literal
from uuid import uuid4

from intelligent_agents_chat.logging_config import log_event


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATABASE_PATH = PROJECT_ROOT / ".data" / "chats.sqlite3"
DEFAULT_PROJECT_ID = "default"
DEFAULT_PROJECT_NAME = "General"
DEFAULT_CONVERSATION_TITLE = "New chat"
MessageRole = Literal["system", "user", "assistant"]
VALID_ROLES = {"system", "user", "assistant"}
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
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL
                        REFERENCES conversations(id) ON DELETE CASCADE,
                    role TEXT NOT NULL CHECK (role IN ('system', 'user', 'assistant')),
                    content TEXT NOT NULL,
                    model_profile TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS conversations_project_updated_idx
                    ON conversations(project_id, updated_at DESC);
                CREATE INDEX IF NOT EXISTS messages_conversation_id_idx
                    ON messages(conversation_id, id);
                """
            )
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
                    id, project_id, title, model_profile, thinking_enabled, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    conversation_id,
                    project_id,
                    clean_title,
                    model_profile,
                    int(thinking_enabled),
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
            title_chars=len(conversation.title),
        )
        return conversation

    def get_conversation(self, conversation_id: str) -> Conversation | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, project_id, title, model_profile, thinking_enabled,
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
                SELECT id, project_id, title, model_profile, thinking_enabled,
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
    ) -> Message:
        if role not in VALID_ROLES:
            raise ValueError(f"Unsupported message role: {role}")
        if not content.strip():
            raise ValueError("Message content cannot be empty")

        now = _timestamp()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO messages (
                    conversation_id, role, content, model_profile, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (conversation_id, role, content, model_profile, now),
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
        )
        return message

    def get_message(self, message_id: int) -> Message | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, conversation_id, role, content, model_profile, created_at
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
                SELECT id, conversation_id, role, content, model_profile, created_at
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

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection


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
        created_at=_parse_datetime(row["created_at"]),
        updated_at=_parse_datetime(row["updated_at"]),
    )


def _message_from_row(row: sqlite3.Row) -> Message:
    return Message(
        id=row["id"],
        conversation_id=row["conversation_id"],
        role=row["role"],
        content=row["content"],
        model_profile=row["model_profile"],
        created_at=_parse_datetime(row["created_at"]),
    )

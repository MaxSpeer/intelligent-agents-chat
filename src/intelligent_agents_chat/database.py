"""SQLite persistence for projects, conversations, and messages."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
from typing import Literal
from uuid import uuid4


SCHEMA_VERSION = 1
DEFAULT_PROJECT_ID = "default"
DEFAULT_PROJECT_NAME = "General"
DEFAULT_CONVERSATION_TITLE = "New chat"
MessageRole = Literal["system", "user", "assistant"]
VALID_ROLES = {"system", "user", "assistant"}


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

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path

    def initialize(self) -> None:
        """Create the schema and the first project if they do not exist."""
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version > SCHEMA_VERSION:
                raise RuntimeError(
                    f"Database schema {version} is newer than supported schema {SCHEMA_VERSION}"
                )

            connection.execute("PRAGMA journal_mode = WAL")
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
            connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    def get_project(self, project_id: str = DEFAULT_PROJECT_ID) -> Project | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT id, name, created_at, updated_at FROM projects WHERE id = ?",
                (project_id,),
            ).fetchone()
        return _project_from_row(row) if row else None

    def create_conversation(
        self,
        model_profile: str,
        *,
        project_id: str = DEFAULT_PROJECT_ID,
        title: str = DEFAULT_CONVERSATION_TITLE,
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
                    id, project_id, title, model_profile, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (conversation_id, project_id, clean_title, model_profile, now, now),
            )
        conversation = self.get_conversation(conversation_id)
        if conversation is None:
            raise RuntimeError("Failed to read the conversation after creating it")
        return conversation

    def get_conversation(self, conversation_id: str) -> Conversation | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, project_id, title, model_profile, created_at, updated_at
                FROM conversations
                WHERE id = ?
                """,
                (conversation_id,),
            ).fetchone()
        return _conversation_from_row(row) if row else None

    def list_conversations(
        self, project_id: str = DEFAULT_PROJECT_ID
    ) -> list[Conversation]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, project_id, title, model_profile, created_at, updated_at
                FROM conversations
                WHERE project_id = ?
                ORDER BY updated_at DESC, created_at DESC
                """,
                (project_id,),
            ).fetchall()
        return [_conversation_from_row(row) for row in rows]

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
        return cursor.rowcount == 1

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
        return cursor.rowcount == 1

    def delete_conversation(self, conversation_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM conversations WHERE id = ?",
                (conversation_id,),
            )
        return cursor.rowcount == 1

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
        return [_message_from_row(row) for row in rows]

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

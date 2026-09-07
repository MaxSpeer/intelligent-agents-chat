"""Project-scoped, rebuildable conversation memory backed by SQLite FTS5."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
from pathlib import Path
import re
import sqlite3

from intelligent_agents_chat.database import DEFAULT_DATABASE_PATH, connect_database
from intelligent_agents_chat.logging_config import log_event


MAX_QUERY_TERMS = 12
MAX_RETRIEVED_CHARS = 3_000
DEFAULT_RETRIEVAL_LIMIT = 6
_STOP_WORDS = {
    "aber",
    "auch",
    "das",
    "der",
    "die",
    "ein",
    "eine",
    "einer",
    "für",
    "ist",
    "mit",
    "oder",
    "the",
    "und",
    "von",
    "was",
    "wie",
    "wir",
    "zu",
}
logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class MemoryEntry:
    """One stored turn, as the "Manage project memory" dialog lists it."""

    id: int
    project_id: str
    source_conversation_id: str
    source_message_start_id: int
    source_message_end_id: int
    content: str
    enabled: bool
    title: str
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class MemoryCandidate:
    """One retrieved entry on its way into a turn's context: the text to
    quote plus what to cite it as. That's all anything downstream reads --
    ContextAssembler renders exactly these three into the prompt's retrieval
    block, and app.py persists the same three as the answer's provenance
    (see ContextSourceInput in database.py and the trace step it feeds).
    Ranking lives in the result order, not in a field: retrieve() already
    returns the best match first.
    """

    text: str
    title: str
    locator: str


class ProjectMemoryStore:
    """Index chat turns and retrieve them only through a mandatory project scope."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path

    def rebuild_project(self, project_id: str) -> int:
        with self._connect() as connection:
            conversation_ids = [
                row["id"]
                for row in connection.execute(
                    "SELECT id FROM conversations WHERE project_id = ?",
                    (project_id,),
                )
            ]
        count = sum(
            self.rebuild_conversation(conversation_id) for conversation_id in conversation_ids
        )
        log_event(
            logger,
            logging.INFO,
            "memory.project.rebuilt",
            project_id=project_id,
            conversation_count=len(conversation_ids),
            entry_count=count,
        )
        return count

    def rebuild_conversation(self, conversation_id: str) -> int:
        now = datetime.now(timezone.utc).isoformat(timespec="microseconds")
        with self._connect() as connection:
            conversation = connection.execute(
                "SELECT id, project_id, title FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
            if conversation is None:
                return 0
            rows = connection.execute(
                """
                SELECT id, role, content
                FROM messages
                WHERE conversation_id = ? AND role IN ('user', 'assistant')
                ORDER BY id ASC
                """,
                (conversation_id,),
            ).fetchall()
            chunks = _conversation_chunks(conversation["title"], rows)
            desired_ranges = {(chunk[0], chunk[1]) for chunk in chunks}
            existing_rows = connection.execute(
                """
                SELECT id, source_message_start_id, source_message_end_id
                FROM memory_entries
                WHERE source_conversation_id = ?
                """,
                (conversation_id,),
            ).fetchall()
            for existing in existing_rows:
                source_range = (
                    existing["source_message_start_id"],
                    existing["source_message_end_id"],
                )
                if source_range not in desired_ranges:
                    connection.execute("DELETE FROM memory_entries WHERE id = ?", (existing["id"],))

            for start_id, end_id, content in chunks:
                # rebuild_conversation reprocesses the whole conversation on every
                # call (see the module docstring's rationale), so most chunks here
                # are unchanged from the previous rebuild. The WHERE guard keeps
                # that a true no-op -- without it, every unchanged chunk would
                # still get its updated_at bumped and its FTS row deleted and
                # reinserted (see the sync triggers in database.py), which would
                # make "most recently updated" meaningless (it'd really mean
                # "this conversation had any activity recently") and cause
                # needless FTS churn on every turn.
                connection.execute(
                    """
                    INSERT INTO memory_entries (
                        project_id, source_conversation_id, source_message_start_id,
                        source_message_end_id, content, enabled, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, 1, ?, ?)
                    ON CONFLICT (
                        source_conversation_id, source_message_start_id, source_message_end_id
                    ) DO UPDATE SET
                        project_id = excluded.project_id,
                        content = excluded.content,
                        updated_at = excluded.updated_at
                    WHERE memory_entries.content IS NOT excluded.content
                    """,
                    (
                        conversation["project_id"],
                        conversation_id,
                        start_id,
                        end_id,
                        content,
                        now,
                        now,
                    ),
                )

        log_event(
            logger,
            logging.DEBUG,
            "memory.conversation.rebuilt",
            project_id=conversation["project_id"],
            conversation_id=conversation_id,
            source_message_count=len(rows),
            entry_count=len(chunks),
        )
        return len(chunks)

    def retrieve(
        self,
        *,
        project_id: str,
        text: str,
        exclude_conversation_id: str | None = None,
        limit: int = DEFAULT_RETRIEVAL_LIMIT,
    ) -> list[MemoryCandidate]:
        """The turns from this project most relevant to `text`.

        exclude_conversation_id skips the conversation being answered right
        now: its own messages are already in the history that gets sent, so
        quoting them back as "memory" would just be noise.
        """
        if not project_id.strip():
            raise ValueError("Project-scoped retrieval requires a project ID")
        if limit <= 0:
            return []
        fts_query = _fts_query(text)
        if not fts_query:
            return []

        parameters: list[object] = [fts_query, project_id]
        exclusion = ""
        if exclude_conversation_id is not None:
            exclusion = "AND me.source_conversation_id != ?"
            parameters.append(exclude_conversation_id)
        parameters.append(limit)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT me.id, me.project_id, me.source_conversation_id,
                       me.source_message_start_id, me.source_message_end_id, me.content,
                       conversation.title, bm25(memory_entries_fts) AS relevance
                FROM memory_entries_fts
                JOIN memory_entries AS me ON me.id = memory_entries_fts.rowid
                JOIN conversations AS conversation ON conversation.id = me.source_conversation_id
                WHERE memory_entries_fts MATCH ?
                  AND me.project_id = ?
                  AND me.enabled = 1
                  {exclusion}
                ORDER BY relevance ASC, me.updated_at DESC, me.id DESC
                LIMIT ?
                """,
                parameters,
            ).fetchall()

        candidates = [
            MemoryCandidate(
                text=_bounded_text(row["content"]),
                title=row["title"],
                locator=(
                    f"messages {row['source_message_start_id']}-{row['source_message_end_id']}"
                ),
            )
            for row in rows
        ]
        log_event(
            logger,
            logging.INFO,
            "memory.retrieve.completed",
            project_id=project_id,
            excluded_conversation_id=exclude_conversation_id,
            query_chars=len(text),
            query_term_count=fts_query.count(" OR ") + 1,
            requested_limit=limit,
            result_count=len(candidates),
        )
        return candidates

    def list_entries(self, project_id: str) -> list[MemoryEntry]:
        if not project_id.strip():
            raise ValueError("Project ID cannot be empty")
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT me.id, me.project_id, me.source_conversation_id,
                       me.source_message_start_id, me.source_message_end_id,
                       me.content, me.enabled, conversation.title,
                       me.created_at, me.updated_at
                FROM memory_entries AS me
                JOIN conversations AS conversation ON conversation.id = me.source_conversation_id
                WHERE me.project_id = ?
                ORDER BY me.updated_at DESC, me.id DESC
                """,
                (project_id,),
            ).fetchall()
        entries = [_memory_entry_from_row(row) for row in rows]
        log_event(
            logger,
            logging.DEBUG,
            "memory.entries.listed",
            project_id=project_id,
            entry_count=len(entries),
            enabled_count=sum(entry.enabled for entry in entries),
        )
        return entries

    def set_enabled(self, project_id: str, entry_id: int, enabled: bool) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE memory_entries
                SET enabled = ?, updated_at = ?
                WHERE id = ? AND project_id = ?
                """,
                (
                    int(enabled),
                    datetime.now(timezone.utc).isoformat(timespec="microseconds"),
                    entry_id,
                    project_id,
                ),
            )
        updated = cursor.rowcount == 1
        log_event(
            logger,
            logging.INFO,
            "memory.entry.enabled_changed",
            project_id=project_id,
            entry_id=entry_id,
            enabled=enabled,
            updated=updated,
        )
        return updated

    def _connect(self) -> sqlite3.Connection:
        return connect_database(self.database_path)


def _conversation_chunks(title: str, rows: list[sqlite3.Row]) -> list[tuple[int, int, str]]:
    """One chunk per turn: a user message plus that turn's final assistant
    answer. A turn can contain several tool-call rounds (extra assistant rows
    -- e.g. narration like "let me check that page" right before a tool call;
    tool results themselves are already excluded by the caller's query), but
    those rounds don't close the chunk early or leak into it -- only the next
    user message starts a new turn, mirroring the UI's own trace-vs-answer
    distinction (see _format_chunk).
    """
    chunks: list[tuple[int, int, str]] = []
    current: list[sqlite3.Row] = []
    for row in rows:
        if row["role"] == "user" and current:
            chunk = _format_chunk(title, current)
            if chunk is not None:
                chunks.append(chunk)
            current = []
        current.append(row)
    if current:
        chunk = _format_chunk(title, current)
        if chunk is not None:
            chunks.append(chunk)
    return chunks


def _format_chunk(title: str, rows: list[sqlite3.Row]) -> tuple[int, int, str] | None:
    """Render one turn as its user message plus its final assistant answer,
    skipping any assistant rows in between (tool calls and reasoning). Returns None for a
    turn that has no answer yet (e.g. the pre-emptive rebuild right after the
    user message is saved, before generation finishes, or a turn stopped
    before producing any content) -- the next rebuild fills it in once an
    answer exists.
    """
    user_row = next((row for row in rows if row["role"] == "user"), None)
    final_answer = next(
        (
            row["content"]
            for row in reversed(rows)
            if row["role"] == "assistant" and row["content"].strip()
        ),
        None,
    )
    if user_row is None or final_answer is None:
        return None
    lines = [
        f"Conversation: {title}",
        f"User: {user_row['content']}",
        f"Assistant: {final_answer}",
    ]
    return rows[0]["id"], rows[-1]["id"], "\n".join(lines)


# Decimal numbers are matched whole (not split into two tokens on the ".")
# before falling back to plain word characters. A lone "3" or "878" is useless search
# signal, but "3.878" is specific enough to matter.
_TOKEN_PATTERN = re.compile(r"\d+\.\d+|[^\W_]+", flags=re.UNICODE)


def _fts_query(text: str) -> str:
    terms: list[str] = []
    for term in _TOKEN_PATTERN.findall(text.casefold()):
        if len(term) < 2 or term in _STOP_WORDS or term in terms:
            continue
        terms.append(term)
        if len(terms) == MAX_QUERY_TERMS:
            break
    return " OR ".join(f'"{term.replace(chr(34), chr(34) * 2)}"' for term in terms)


def _bounded_text(value: str) -> str:
    if len(value) <= MAX_RETRIEVED_CHARS:
        return value
    return f"{value[: MAX_RETRIEVED_CHARS - 14].rstrip()}\n[truncated]"


def _memory_entry_from_row(row: sqlite3.Row) -> MemoryEntry:
    return MemoryEntry(
        id=row["id"],
        project_id=row["project_id"],
        source_conversation_id=row["source_conversation_id"],
        source_message_start_id=row["source_message_start_id"],
        source_message_end_id=row["source_message_end_id"],
        content=row["content"],
        enabled=bool(row["enabled"]),
        title=row["title"],
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
    )


memory_store = ProjectMemoryStore(DEFAULT_DATABASE_PATH)

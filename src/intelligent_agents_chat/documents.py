"""Project-scoped documents end to end: parsing, chunking, SQLite persistence,
embedding-vector storage, and the retrieval the search_documents tool runs.

Everything about a project's uploaded documents lives here, in one place --
the same way memory.py holds all of project memory (store, rebuild, and
retrieval together). Reading top to bottom: the storage primitives
(BlobStore for the original files, DocumentParser, Chunker, DocumentStore
for SQLite and sqlite-vec), then DocumentService, which is what the UI
actually calls to upload/reindex/delete, and finally ProjectRAGRetriever,
which turns a query into ranked, citable passages for
tools/search_documents.py.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import logging
import os
from pathlib import Path
import re
import sqlite3
from typing import Literal, Sequence
from uuid import uuid4

import sqlite_vec
from pypdf import PdfReader

from intelligent_agents_chat.database import (
    DEFAULT_DATABASE_PATH,
    PROJECT_ROOT,
    connect_database,
)
from intelligent_agents_chat.embeddings import (
    EMBEDDING_DIMENSION,
    EmbeddingError,
    EmbeddingGateway,
    create_embedding_gateway,
)
from intelligent_agents_chat.logging_config import log_event


DEFAULT_DOCUMENT_ROOT = PROJECT_ROOT / ".data" / "documents"
MAX_DOCUMENT_BYTES = 10 * 1024 * 1024
MAX_DOCUMENTS_PER_PROJECT = 100
DEFAULT_CHUNK_CHARS = 1_200
DEFAULT_CHUNK_OVERLAP_CHARS = 180
MAX_RETRIEVED_CHARS = 4_000
SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".pdf"}
GENERIC_MEDIA_TYPES = {"application/octet-stream", ""}
SUPPORTED_MEDIA_TYPES_BY_EXTENSION = {
    ".txt": {"text/plain"},
    ".md": {"text/markdown", "text/x-markdown", "text/plain"},
    ".markdown": {"text/markdown", "text/x-markdown", "text/plain"},
    ".pdf": {"application/pdf", "application/x-pdf"},
}
DocumentStatus = Literal["processing", "indexed", "failed"]
logger = logging.getLogger(__name__)


class DocumentError(RuntimeError):
    """Base class for recoverable document lifecycle failures."""


class DocumentValidationError(DocumentError, ValueError):
    """An uploaded file violates a documented input constraint."""


@dataclass(frozen=True, slots=True)
class Document:
    id: str
    project_id: str
    display_name: str
    storage_key: str
    media_type: str
    extension: str
    sha256: str
    byte_size: int
    status: DocumentStatus
    error_message: str | None
    chunk_count: int
    embedding_model: str | None
    embedding_dimension: int | None
    created_at: datetime
    updated_at: datetime
    indexed_at: datetime | None


@dataclass(frozen=True, slots=True)
class ParsedSegment:
    """One locatable piece of a parsed document -- a PDF page or a Markdown
    section. `locator` is the human-readable citation ("page 3", "section:
    Retrieval") the model gets to quote, and it's the only place that
    structure is kept: nothing downstream needs the page number or heading
    as separate values.
    """

    text: str
    locator: str


@dataclass(frozen=True, slots=True)
class ChunkDraft:
    ordinal: int
    content: str
    locator: str


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    document_id: str
    project_id: str
    ordinal: int
    content: str
    locator: str


@dataclass(frozen=True, slots=True)
class DocumentPassage:
    """One retrieved passage, as search_documents shows it to the model:
    the text plus what to cite it as. Nothing more -- unlike project memory
    (see memory.py's MemoryCandidate), a passage never enters the assembled
    context and is never persisted as provenance, so there is nothing here
    to carry a score, an id, or a project through.
    """

    text: str
    title: str
    locator: str


class BlobStore:
    """Persist original uploads under generated project/document paths only."""

    def __init__(self, root: Path = DEFAULT_DOCUMENT_ROOT) -> None:
        self.root = root

    def storage_key(self, project_id: str, document_id: str, extension: str) -> str:
        _validate_storage_component(project_id)
        _validate_storage_component(document_id)
        normalized_extension = _normalized_extension(extension)
        return f"{project_id}/{document_id}/source{normalized_extension}"

    def write(self, storage_key: str, data: bytes) -> None:
        path = self._resolve(storage_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(data)
        temporary.replace(path)

    def read(self, storage_key: str) -> bytes:
        return self._resolve(storage_key).read_bytes()

    def delete(self, storage_key: str) -> None:
        path = self._resolve(storage_key)
        path.unlink(missing_ok=True)
        document_directory = path.parent
        if document_directory.exists() and not any(document_directory.iterdir()):
            document_directory.rmdir()
        project_directory = document_directory.parent
        if project_directory.exists() and not any(project_directory.iterdir()):
            project_directory.rmdir()

    def exists(self, storage_key: str) -> bool:
        return self._resolve(storage_key).is_file()

    def _resolve(self, storage_key: str) -> Path:
        relative = Path(storage_key)
        if relative.is_absolute() or ".." in relative.parts:
            raise DocumentValidationError("Invalid generated document storage key")
        root = self.root.resolve()
        resolved = (root / relative).resolve()
        if resolved != root and root not in resolved.parents:
            raise DocumentValidationError("Document path escapes the configured storage root")
        return resolved


class DocumentParser:
    """Extract text and stable locators from supported document types."""

    def parse(self, data: bytes, *, display_name: str, media_type: str) -> list[ParsedSegment]:
        extension = validate_document_input(display_name, media_type, len(data))
        if extension == ".pdf":
            return self._parse_pdf(data)
        return self._parse_text(data, markdown=extension in {".md", ".markdown"})

    def _parse_text(self, data: bytes, *, markdown: bool) -> list[ParsedSegment]:
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError as error:
            raise DocumentValidationError("Text and Markdown files must use UTF-8") from error
        if "\x00" in text:
            raise DocumentValidationError("Text file contains binary null bytes")
        normalized = _normalize_document_text(text)
        if not normalized:
            raise DocumentValidationError("Document contains no extractable text")
        if not markdown:
            return [ParsedSegment(text=normalized, locator="document")]

        segments: list[ParsedSegment] = []
        current_section = "Document"
        current_lines: list[str] = []
        for line in normalized.splitlines():
            heading = re.match(r"^#{1,6}\s+(.+?)\s*$", line)
            if heading and current_lines:
                section_text = "\n".join(current_lines).strip()
                if section_text:
                    segments.append(
                        ParsedSegment(text=section_text, locator=f"section: {current_section}")
                    )
                current_lines = []
            if heading:
                current_section = heading.group(1).strip()[:200]
            current_lines.append(line)
        section_text = "\n".join(current_lines).strip()
        if section_text:
            segments.append(
                ParsedSegment(text=section_text, locator=f"section: {current_section}")
            )
        return segments

    def _parse_pdf(self, data: bytes) -> list[ParsedSegment]:
        try:
            reader = PdfReader(BytesIO(data), strict=False)
            if reader.is_encrypted and reader.decrypt("") == 0:
                raise DocumentValidationError("Password-protected PDFs are not supported")
            segments = []
            for page_index, page in enumerate(reader.pages, start=1):
                text = _normalize_document_text(page.extract_text() or "")
                if text:
                    segments.append(ParsedSegment(text=text, locator=f"page {page_index}"))
        except DocumentValidationError:
            raise
        except Exception as error:
            raise DocumentValidationError("PDF could not be parsed safely") from error
        if not segments:
            raise DocumentValidationError(
                "PDF contains no extractable text; scanned PDFs require OCR"
            )
        return segments


class Chunker:
    """Create deterministic, overlapping chunks without crossing source locators."""

    def __init__(
        self,
        *,
        chunk_chars: int = DEFAULT_CHUNK_CHARS,
        overlap_chars: int = DEFAULT_CHUNK_OVERLAP_CHARS,
    ) -> None:
        if chunk_chars < 200:
            raise ValueError("chunk_chars must be at least 200")
        if overlap_chars < 0 or overlap_chars >= chunk_chars:
            raise ValueError("overlap_chars must be non-negative and smaller than chunk_chars")
        self.chunk_chars = chunk_chars
        self.overlap_chars = overlap_chars

    def chunk(self, segments: Sequence[ParsedSegment]) -> list[ChunkDraft]:
        drafts: list[ChunkDraft] = []
        for segment in segments:
            pieces = _split_text(segment.text, self.chunk_chars, self.overlap_chars)
            for piece_index, piece in enumerate(pieces, start=1):
                locator = segment.locator
                if len(pieces) > 1:
                    locator = f"{locator}, chunk {piece_index}/{len(pieces)}"
                drafts.append(
                    ChunkDraft(ordinal=len(drafts), content=piece, locator=locator)
                )
        if not drafts:
            raise DocumentValidationError("Document contains no indexable text chunks")
        return drafts


class DocumentStore:
    """Document metadata, chunks, and their vector embeddings in SQLite."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path

    def initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                f"""
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

                -- `row_id` deliberately shares values with document_chunks.row_id
                -- above (kept in sync by hand in replace_chunks) so a plain JOIN
                -- recovers chunk content/locator/etc. after a vector search --
                -- vec0 has no foreign keys or content_rowid-style linkage of its
                -- own. The dimension below must match embeddings.py's pinned
                -- model; changing the model means migrating this column too.
                CREATE VIRTUAL TABLE IF NOT EXISTS document_chunks_vec USING vec0(
                    row_id INTEGER PRIMARY KEY,
                    project_id TEXT PARTITION KEY,
                    embedding FLOAT[{EMBEDDING_DIMENSION}] DISTANCE_METRIC=COSINE
                );
                """
            )
        log_event(
            logger,
            logging.INFO,
            "documents.store.initialized",
            database_path=str(self.database_path),
        )

    def create_pending(
        self,
        *,
        project_id: str,
        display_name: str,
        storage_key: str,
        media_type: str,
        extension: str,
        sha256: str,
        byte_size: int,
        document_id: str | None = None,
    ) -> Document:
        now = _timestamp()
        created_id = document_id or str(uuid4())
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO documents (
                    id, project_id, display_name, storage_key, media_type, extension,
                    sha256, byte_size, status, error_message, chunk_count,
                    embedding_model, embedding_dimension, created_at, updated_at, indexed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'processing', NULL, 0, NULL, NULL, ?, ?, NULL)
                """,
                (
                    created_id,
                    project_id,
                    display_name,
                    storage_key,
                    media_type,
                    extension,
                    sha256,
                    byte_size,
                    now,
                    now,
                ),
            )
        document = self.get_document(created_id)
        if document is None:
            raise RuntimeError("Failed to read document after creating it")
        return document

    def find_duplicate(self, project_id: str, sha256: str) -> Document | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM documents WHERE project_id = ? AND sha256 = ?
                """,
                (project_id, sha256),
            ).fetchone()
        return _document_from_row(row) if row else None

    def get_document(self, document_id: str) -> Document | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE id = ?",
                (document_id,),
            ).fetchone()
        return _document_from_row(row) if row else None

    def list_documents(self, project_id: str) -> list[Document]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM documents
                WHERE project_id = ?
                ORDER BY updated_at DESC, display_name COLLATE NOCASE, id
                """,
                (project_id,),
            ).fetchall()
        return [_document_from_row(row) for row in rows]

    def count_documents(self, project_id: str) -> int:
        with self._connect() as connection:
            return int(
                connection.execute(
                    "SELECT COUNT(*) FROM documents WHERE project_id = ?",
                    (project_id,),
                ).fetchone()[0]
            )

    def mark_processing(self, document_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE documents
                SET status = 'processing', error_message = NULL, updated_at = ?
                WHERE id = ?
                """,
                (_timestamp(), document_id),
            )
        return cursor.rowcount == 1

    def replace_chunks(
        self,
        document_id: str,
        chunks: Sequence[ChunkDraft],
        *,
        embeddings: Sequence[Sequence[float]],
        embedding_model: str,
    ) -> Document:
        if len(embeddings) != len(chunks):
            raise ValueError("Embedding count must match chunk count")
        if embeddings:
            dimensions = {len(vector) for vector in embeddings}
            if len(dimensions) != 1 or 0 in dimensions:
                raise ValueError("All embeddings must have one non-zero dimension")
        now = _timestamp()
        with self._connect() as connection:
            document = connection.execute(
                "SELECT project_id FROM documents WHERE id = ?",
                (document_id,),
            ).fetchone()
            if document is None:
                raise DocumentError("Document no longer exists")
            project_id = document["project_id"]
            # document_chunks_vec has no foreign key of its own -- clear its
            # rows for this document's old chunks by hand before the old
            # document_chunks rows (and their row_ids) disappear.
            old_row_ids = [
                row["row_id"]
                for row in connection.execute(
                    "SELECT row_id FROM document_chunks WHERE document_id = ?",
                    (document_id,),
                )
            ]
            if old_row_ids:
                connection.executemany(
                    "DELETE FROM document_chunks_vec WHERE row_id = ?",
                    [(row_id,) for row_id in old_row_ids],
                )
            connection.execute("DELETE FROM document_chunks WHERE document_id = ?", (document_id,))
            for index, chunk in enumerate(chunks):
                cursor = connection.execute(
                    """
                    INSERT INTO document_chunks (
                        document_id, project_id, ordinal, content, locator, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        document_id,
                        project_id,
                        chunk.ordinal,
                        chunk.content,
                        chunk.locator,
                        now,
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO document_chunks_vec (row_id, project_id, embedding)
                    VALUES (?, ?, ?)
                    """,
                    (
                        cursor.lastrowid,
                        project_id,
                        sqlite_vec.serialize_float32(embeddings[index]),
                    ),
                )
            connection.execute(
                """
                UPDATE documents
                SET status = 'indexed', error_message = NULL, chunk_count = ?,
                    embedding_model = ?, embedding_dimension = ?, updated_at = ?, indexed_at = ?
                WHERE id = ?
                """,
                (
                    len(chunks),
                    embedding_model,
                    EMBEDDING_DIMENSION,
                    now,
                    now,
                    document_id,
                ),
            )
        result = self.get_document(document_id)
        if result is None:
            raise RuntimeError("Document disappeared after indexing")
        return result

    def mark_failed(self, document_id: str, error_message: str) -> bool:
        safe_message = " ".join(error_message.split())[:500] or "Unknown ingestion error"
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE documents
                SET status = 'failed', error_message = ?, updated_at = ?
                WHERE id = ?
                """,
                (safe_message, _timestamp(), document_id),
            )
        return cursor.rowcount == 1

    def delete_document(self, document_id: str, project_id: str) -> Document | None:
        document = self.get_document(document_id)
        if document is None or document.project_id != project_id:
            return None
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM documents WHERE id = ? AND project_id = ?",
                (document_id, project_id),
            )
        return document

    def list_chunks(self, document_id: str) -> list[DocumentChunk]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT document_id, project_id, ordinal, content, locator
                FROM document_chunks
                WHERE document_id = ?
                ORDER BY ordinal
                """,
                (document_id,),
            ).fetchall()
        return [_chunk_from_row(row) for row in rows]

    def vector_search(
        self, project_id: str, query_vector: Sequence[float], limit: int
    ) -> list[tuple[DocumentChunk, float, str]]:
        """The `limit` nearest chunks to `query_vector`, scoped to one
        project via vec0's partition key (see initialize) so this never
        scans another project's vectors. `distance` is cosine distance
        (smaller = more similar); `document.status = 'indexed'` is a
        belt-and-braces filter -- a document whose *reindex* failed keeps
        its previous chunks/vectors in place with status='failed' rather
        than being cleaned up, so this keeps searches from surfacing them.
        """
        if limit <= 0:
            return []
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT dc.document_id, dc.project_id, dc.ordinal, dc.content, dc.locator,
                       document.display_name, v.distance AS distance
                FROM document_chunks_vec AS v
                JOIN document_chunks AS dc ON dc.row_id = v.row_id
                JOIN documents AS document ON document.id = dc.document_id
                WHERE v.embedding MATCH ?
                  AND v.k = ?
                  AND v.project_id = ?
                  AND document.status = 'indexed'
                ORDER BY v.distance
                """,
                (sqlite_vec.serialize_float32(list(query_vector)), limit, project_id),
            ).fetchall()
        return [(_chunk_from_row(row), float(row["distance"]), row["display_name"]) for row in rows]

    def _connect(self) -> sqlite3.Connection:
        return connect_database(self.database_path)


@dataclass(frozen=True, slots=True)
class UploadResult:
    document: Document
    duplicate: bool


class DocumentService:
    """Coordinate safe blob storage, parsing, chunking, embeddings, and
    metadata -- the one entry point app.py's document dialog uses for
    everything a user can do to a project's documents (upload, retry a
    failed one, delete). The pieces above each do one job and know nothing
    about each other; this is what puts them in order and keeps the
    `documents` row's status honest when a step fails halfway.
    """

    def __init__(
        self,
        store: DocumentStore,
        blob_store: BlobStore,
        parser: DocumentParser,
        chunker: Chunker,
        embedding_gateway: EmbeddingGateway,
    ) -> None:
        self.store = store
        self.blob_store = blob_store
        self.parser = parser
        self.chunker = chunker
        self.embedding_gateway = embedding_gateway

    @property
    def embedding_model(self) -> str:
        return self.embedding_gateway.model_name

    async def upload(
        self,
        *,
        project_id: str,
        display_name: str,
        media_type: str,
        data: bytes,
    ) -> UploadResult:
        clean_name = Path(display_name).name.strip()
        extension = validate_document_input(clean_name, media_type, len(data))
        digest = hashlib.sha256(data).hexdigest()
        duplicate = self.store.find_duplicate(project_id, digest)
        if duplicate is not None:
            if duplicate.status == "failed":
                duplicate = await self.reindex(duplicate.id, project_id)
            log_event(
                logger,
                logging.INFO,
                "documents.document.duplicate",
                project_id=project_id,
                document_id=duplicate.id,
                byte_size=len(data),
            )
            return UploadResult(document=duplicate, duplicate=True)
        if self.store.count_documents(project_id) >= MAX_DOCUMENTS_PER_PROJECT:
            raise DocumentValidationError(
                f"Project document limit of {MAX_DOCUMENTS_PER_PROJECT} has been reached"
            )

        document_id = str(uuid4())
        storage_key = self.blob_store.storage_key(project_id, document_id, extension)
        try:
            document = self.store.create_pending(
                project_id=project_id,
                display_name=clean_name,
                storage_key=storage_key,
                media_type=media_type.split(";", 1)[0].strip().lower(),
                extension=extension,
                sha256=digest,
                byte_size=len(data),
                document_id=document_id,
            )
        except sqlite3.IntegrityError:
            raced_duplicate = self.store.find_duplicate(project_id, digest)
            if raced_duplicate is None:
                raise
            return UploadResult(document=raced_duplicate, duplicate=True)

        try:
            await asyncio.to_thread(self.blob_store.write, storage_key, data)
            indexed = await self._index(document, data)
        except Exception as error:
            self.store.mark_failed(document.id, str(error))
            log_event(
                logger,
                logging.WARNING,
                "documents.document.ingestion_failed",
                project_id=project_id,
                document_id=document.id,
                byte_size=len(data),
                error_type=type(error).__name__,
            )
            raise

        log_event(
            logger,
            logging.INFO,
            "documents.document.uploaded",
            project_id=project_id,
            document_id=indexed.id,
            byte_size=indexed.byte_size,
            chunk_count=indexed.chunk_count,
            embedding_model=indexed.embedding_model,
        )
        return UploadResult(document=indexed, duplicate=False)

    async def reindex(self, document_id: str, project_id: str) -> Document:
        document = self._project_document(document_id, project_id)
        self.store.mark_processing(document.id)
        try:
            data = await asyncio.to_thread(self.blob_store.read, document.storage_key)
            indexed = await self._index(document, data)
        except Exception as error:
            self.store.mark_failed(document.id, str(error))
            log_event(
                logger,
                logging.WARNING,
                "documents.document.reindex_failed",
                project_id=project_id,
                document_id=document.id,
                error_type=type(error).__name__,
            )
            raise
        log_event(
            logger,
            logging.INFO,
            "documents.document.reindexed",
            project_id=project_id,
            document_id=document.id,
            chunk_count=indexed.chunk_count,
            embedding_model=indexed.embedding_model,
        )
        return indexed

    async def delete(self, document_id: str, project_id: str) -> bool:
        deleted = self.store.delete_document(document_id, project_id)
        if deleted is None:
            return False
        await asyncio.to_thread(self.blob_store.delete, deleted.storage_key)
        log_event(
            logger,
            logging.INFO,
            "documents.document.deleted",
            project_id=project_id,
            document_id=document_id,
            chunk_count=deleted.chunk_count,
        )
        return True

    async def _index(self, document: Document, data: bytes) -> Document:
        segments = await asyncio.to_thread(
            self.parser.parse,
            data,
            display_name=document.display_name,
            media_type=document.media_type,
        )
        chunks = self.chunker.chunk(segments)
        embeddings = await self._embed_chunks(chunks)
        return self.store.replace_chunks(
            document.id,
            chunks,
            embeddings=embeddings,
            embedding_model=self.embedding_model,
        )

    async def _embed_chunks(self, chunks: Sequence[ChunkDraft]) -> list[list[float]]:
        return await self.embedding_gateway.embed([chunk.content for chunk in chunks])

    def _project_document(self, document_id: str, project_id: str) -> Document:
        document = self.store.get_document(document_id)
        if document is None or document.project_id != project_id:
            raise DocumentError("Document does not exist in this project")
        return document


class ProjectRAGRetriever:
    """Turn a query into ranked, citable passages from one project's
    documents: embed the query with the same model the chunks were embedded
    with, then let sqlite-vec find the nearest ones (see
    DocumentStore.vector_search). Only ever reached through the
    search_documents tool -- documents are never pulled into a turn's
    context automatically (see tools/search_documents.py), which is why a
    passage carries nothing but what the model needs to read and cite it.
    """

    def __init__(self, store: DocumentStore, embedding_gateway: EmbeddingGateway) -> None:
        self.store = store
        self.embedding_gateway = embedding_gateway

    async def retrieve(
        self, *, project_id: str, text: str, limit: int
    ) -> list[DocumentPassage]:
        if not project_id.strip():
            raise ValueError("Project-scoped retrieval requires a project ID")
        if limit <= 0 or not text.strip():
            return []

        try:
            query_vectors = await self.embedding_gateway.embed([text])
        except EmbeddingError as error:
            log_event(
                logger,
                logging.WARNING,
                "documents.retrieve.query_embedding_failed",
                project_id=project_id,
                query_chars=len(text),
                error_type=type(error).__name__,
            )
            return []

        matches = self.store.vector_search(project_id, query_vectors[0], limit)
        passages = [
            DocumentPassage(
                text=bounded_chunk_text(chunk.content),
                title=title,
                locator=chunk.locator,
            )
            for chunk, _distance, title in matches
        ]
        log_event(
            logger,
            logging.INFO,
            "documents.retrieve.completed",
            project_id=project_id,
            query_chars=len(text),
            requested_limit=limit,
            result_count=len(passages),
            embedding_model=self.embedding_gateway.model_name,
        )
        return passages

def validate_document_input(display_name: str, media_type: str, byte_size: int) -> str:
    clean_name = Path(display_name).name.strip()
    if not clean_name:
        raise DocumentValidationError("Document filename cannot be empty")
    extension = _normalized_extension(Path(clean_name).suffix)
    if extension not in SUPPORTED_EXTENSIONS:
        raise DocumentValidationError("Supported document types are TXT, Markdown, and PDF")
    normalized_media_type = media_type.split(";", 1)[0].strip().lower()
    allowed_media_types = GENERIC_MEDIA_TYPES | SUPPORTED_MEDIA_TYPES_BY_EXTENSION[extension]
    if normalized_media_type not in allowed_media_types:
        raise DocumentValidationError(
            f"Unsupported media type for {extension.removeprefix('.').upper()}: "
            f"{normalized_media_type or '(empty)'}"
        )
    if byte_size <= 0:
        raise DocumentValidationError("Document is empty")
    if byte_size > MAX_DOCUMENT_BYTES:
        raise DocumentValidationError(
            f"Document exceeds the {MAX_DOCUMENT_BYTES // (1024 * 1024)} MiB limit"
        )
    return extension


def bounded_chunk_text(value: str) -> str:
    if len(value) <= MAX_RETRIEVED_CHARS:
        return value
    return value[: MAX_RETRIEVED_CHARS - 3].rstrip() + "..."


def _split_text(text: str, chunk_chars: int, overlap_chars: int) -> list[str]:
    normalized = text.strip()
    if len(normalized) <= chunk_chars:
        return [normalized] if normalized else []
    pieces: list[str] = []
    start = 0
    while start < len(normalized):
        maximum_end = min(len(normalized), start + chunk_chars)
        end = maximum_end
        if maximum_end < len(normalized):
            candidates = [
                normalized.rfind("\n\n", start + chunk_chars // 2, maximum_end),
                normalized.rfind("\n", start + chunk_chars // 2, maximum_end),
                normalized.rfind(" ", start + chunk_chars // 2, maximum_end),
            ]
            boundary = max(candidates)
            if boundary > start:
                end = boundary
        piece = normalized[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= len(normalized):
            break
        next_start = max(start + 1, end - overlap_chars)
        while next_start < end and not normalized[next_start].isspace():
            next_start += 1
        start = next_start
    return pieces


def _normalize_document_text(value: str) -> str:
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in value.splitlines()]
    return "\n".join(lines).strip()


def _normalized_extension(value: str) -> str:
    extension = value.casefold()
    if extension and not extension.startswith("."):
        extension = "." + extension
    if not re.fullmatch(r"\.[a-z0-9]+", extension):
        raise DocumentValidationError("Document extension is invalid")
    return extension


def _validate_storage_component(value: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise DocumentValidationError("Generated storage identifier is invalid")


def _document_from_row(row: sqlite3.Row) -> Document:
    return Document(
        id=row["id"],
        project_id=row["project_id"],
        display_name=row["display_name"],
        storage_key=row["storage_key"],
        media_type=row["media_type"],
        extension=row["extension"],
        sha256=row["sha256"],
        byte_size=int(row["byte_size"]),
        status=row["status"],
        error_message=row["error_message"],
        chunk_count=int(row["chunk_count"]),
        embedding_model=row["embedding_model"],
        embedding_dimension=row["embedding_dimension"],
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
        indexed_at=datetime.fromisoformat(row["indexed_at"]) if row["indexed_at"] else None,
    )


def _chunk_from_row(row: sqlite3.Row) -> DocumentChunk:
    return DocumentChunk(
        document_id=row["document_id"],
        project_id=row["project_id"],
        ordinal=int(row["ordinal"]),
        content=row["content"],
        locator=row["locator"],
    )


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


#
# The one document stack the whole app shares
#

document_store = DocumentStore(DEFAULT_DATABASE_PATH)
document_store.initialize()
_document_root = Path(os.environ.get("RAG_DOCUMENT_ROOT", str(DEFAULT_DOCUMENT_ROOT)))
_embedding_gateway = create_embedding_gateway()
document_service = DocumentService(
    document_store,
    BlobStore(_document_root),
    DocumentParser(),
    Chunker(),
    _embedding_gateway,
)
rag_retriever = ProjectRAGRetriever(document_store, _embedding_gateway)

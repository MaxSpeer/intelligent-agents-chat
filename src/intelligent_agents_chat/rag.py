"""Project document ingestion and embedding vector retrieval."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import hashlib
import logging
from pathlib import Path
import sqlite3
from typing import Sequence
from uuid import uuid4

from intelligent_agents_chat.documents import (
    MAX_DOCUMENTS_PER_PROJECT,
    BlobStore,
    ChunkDraft,
    Chunker,
    Document,
    DocumentError,
    DocumentParser,
    DocumentStore,
    DocumentValidationError,
    bounded_chunk_text,
    validate_document_input,
)
from intelligent_agents_chat.embeddings import EmbeddingError, EmbeddingGateway
from intelligent_agents_chat.logging_config import log_event
from intelligent_agents_chat.retrieval import ContextCandidate, RetrievalQuery


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class UploadResult:
    document: Document
    duplicate: bool


class DocumentService:
    """Coordinate safe blob storage, parsing, chunking, embeddings, and metadata."""

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
                "rag.document.duplicate",
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
                "rag.document.ingestion_failed",
                project_id=project_id,
                document_id=document.id,
                byte_size=len(data),
                error_type=type(error).__name__,
            )
            raise

        log_event(
            logger,
            logging.INFO,
            "rag.document.uploaded",
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
                "rag.document.reindex_failed",
                project_id=project_id,
                document_id=document.id,
                error_type=type(error).__name__,
            )
            raise
        log_event(
            logger,
            logging.INFO,
            "rag.document.reindexed",
            project_id=project_id,
            document_id=document.id,
            chunk_count=indexed.chunk_count,
            embedding_model=indexed.embedding_model,
        )
        return indexed

    async def replace(
        self,
        document_id: str,
        project_id: str,
        *,
        display_name: str,
        media_type: str,
        data: bytes,
    ) -> Document:
        previous = self._project_document(document_id, project_id)
        clean_name = Path(display_name).name.strip()
        extension = validate_document_input(clean_name, media_type, len(data))
        digest = hashlib.sha256(data).hexdigest()
        duplicate = self.store.find_duplicate(project_id, digest)
        if duplicate is not None and duplicate.id != document_id:
            raise DocumentValidationError(
                f'The same content is already stored as "{duplicate.display_name}"'
            )

        parsed = await asyncio.to_thread(
            self.parser.parse,
            data,
            display_name=clean_name,
            media_type=media_type,
        )
        chunks = self.chunker.chunk(document_id, parsed)
        embeddings = await self._embed_chunks(chunks)
        new_storage_key = self.blob_store.storage_key(project_id, document_id, extension)
        await asyncio.to_thread(self.blob_store.write, new_storage_key, data)
        try:
            updated = self.store.update_original(
                document_id,
                display_name=clean_name,
                storage_key=new_storage_key,
                media_type=media_type.split(";", 1)[0].strip().lower(),
                extension=extension,
                sha256=digest,
                byte_size=len(data),
            )
            if not updated:
                raise DocumentError("Document no longer exists")
            indexed = self.store.replace_chunks(
                document_id,
                chunks,
                embeddings=embeddings,
                embedding_model=self.embedding_model,
            )
        except Exception:
            if new_storage_key != previous.storage_key:
                await asyncio.to_thread(self.blob_store.delete, new_storage_key)
            raise
        if new_storage_key != previous.storage_key:
            await asyncio.to_thread(self.blob_store.delete, previous.storage_key)
        log_event(
            logger,
            logging.INFO,
            "rag.document.replaced",
            project_id=project_id,
            document_id=document_id,
            byte_size=len(data),
            chunk_count=indexed.chunk_count,
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
            "rag.document.deleted",
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
        chunks = self.chunker.chunk(document.id, segments)
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
    """embedding retrieval (sqlite-vec, see documents.py)"""

    def __init__(self, store: DocumentStore, embedding_gateway: EmbeddingGateway) -> None:
        self.store = store
        self.embedding_gateway = embedding_gateway

    async def retrieve(self, query: RetrievalQuery) -> list[ContextCandidate]:
        if not query.project_id.strip():
            raise ValueError("Project-scoped retrieval requires a project ID")
        if query.limit <= 0 or not query.text.strip():
            return []

        try:
            query_vectors = await self.embedding_gateway.embed([query.text])
        except EmbeddingError as error:
            log_event(
                logger,
                logging.WARNING,
                "rag.query_embedding_failed",
                project_id=query.project_id,
                query_chars=len(query.text),
                error_type=type(error).__name__,
            )
            return []

        matches = self.store.vector_search(query.project_id, query_vectors[0], query.limit)
        candidates = [
            ContextCandidate(
                source_kind="project_document",
                source_id=chunk.id,
                project_id=query.project_id,
                source_conversation_id=None,
                text=bounded_chunk_text(chunk.content),
                title=title,
                locator=chunk.locator,
                # Cosine distance, inverted so higher means "more similar" --
                # matches the convention of every other candidate's score
                # (see memory.py), even though it isn't a strict [0, 1] bound.
                score=1.0 - distance,
            )
            for chunk, distance, title in matches
        ]
        log_event(
            logger,
            logging.INFO,
            "rag.retrieve.completed",
            project_id=query.project_id,
            query_chars=len(query.text),
            requested_limit=query.limit,
            result_count=len(candidates),
            embedding_model=self.embedding_gateway.model_name,
        )
        return candidates

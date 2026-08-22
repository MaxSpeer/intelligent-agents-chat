"""Shared retrieval contracts for project memory, RAG, tools, and web context."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class RetrievalQuery:
    """A project-scoped request for optional context candidates."""

    project_id: str
    text: str
    exclude_conversation_id: str | None = None
    limit: int = 6


@dataclass(frozen=True, slots=True)
class ContextCandidate:
    """One ranked piece of context independent of its backing store."""

    source_kind: str
    source_id: str
    project_id: str
    text: str
    title: str
    locator: str
    score: float
    source_conversation_id: str | None = None


class Retriever(Protocol):
    """Common retrieval seam implemented by memory now and document RAG later."""

    def retrieve(self, query: RetrievalQuery) -> list[ContextCandidate]: ...


class AsyncRetriever(Protocol):
    """Retrieval seam for sources that need network I/O, such as embeddings."""

    async def retrieve(self, query: RetrievalQuery) -> list[ContextCandidate]: ...

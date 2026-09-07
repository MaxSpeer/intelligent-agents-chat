"""Local text embeddings, run in-process on this same server -- never routed
through vLLM or any remote endpoint.

Uses fastembed (ONNX Runtime), not sentence-transformers/transformers: the
latter drags in a full CUDA-targeted torch installation even for CPU-only
use, which is the opposite of "lightweight". fastembed runs a small,
well-known sentence-embedding model directly via ONNX, with a much smaller
dependency footprint. The model is downloaded once, on first use (needs
network for that one time only), and cached under EMBEDDING_CACHE_DIR --
every embed() call after that is fully local and offline.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
import logging
from pathlib import Path
from typing import Protocol

from intelligent_agents_chat.database import PROJECT_ROOT
from intelligent_agents_chat.logging_config import log_event


# A small, general-purpose sentence embedding model, English only, 384 dimensions
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
EMBEDDING_CACHE_DIR = PROJECT_ROOT / ".data" / "embedding-models"
logger = logging.getLogger(__name__)


class EmbeddingError(RuntimeError):
    """The local embedding model failed to load or embed the given text."""


class EmbeddingGateway(Protocol):
    """Common seam for turning text into vectors -- a Protocol (not a base
    class) so tests can supply a fake without depending on fastembed at all.
    """

    @property
    def model_name(self) -> str: ...

    async def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class LocalEmbeddingGateway:
    """Wraps fastembed's TextEmbedding. The model loads lazily, on the first
    embed() call, and every call (including that first load) runs in a
    worker thread via asyncio.to_thread -- both model loading and ONNX
    inference are CPU-bound and would otherwise block the event loop
    serving every other connected browser tab (the same reasoning already
    applied to PDF parsing in documents.py).
    """

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL_NAME,
        *,
        cache_dir: Path = EMBEDDING_CACHE_DIR,
    ) -> None:
        self._model_name = model_name
        self._cache_dir = cache_dir
        self._model = None
        # Guards first-load against two concurrent embed() calls both
        # deciding the model isn't loaded yet and racing to load it twice.
        self._load_lock = asyncio.Lock()

    @property
    def model_name(self) -> str:
        return self._model_name

    async def _ensure_model(self):
        if self._model is not None:
            return self._model
        async with self._load_lock:
            if self._model is None:  # re-check: another call may have won the race
                from fastembed import TextEmbedding

                self._cache_dir.mkdir(parents=True, exist_ok=True)
                try:
                    self._model = await asyncio.to_thread(
                        TextEmbedding,
                        model_name=self._model_name,
                        cache_dir=str(self._cache_dir),
                    )
                except Exception as error:
                    raise EmbeddingError(
                        f"could not load embedding model {self._model_name!r}: {error}"
                    ) from error
                log_event(
                    logger,
                    logging.INFO,
                    "embeddings.model.loaded",
                    model=self._model_name,
                    cache_dir=str(self._cache_dir),
                )
        return self._model

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        model = await self._ensure_model()
        try:
            vectors = await asyncio.to_thread(lambda: list(model.embed(list(texts))))
        except Exception as error:
            raise EmbeddingError(f"embedding failed: {error}") from error
        log_event(
            logger,
            logging.DEBUG,
            "embeddings.request.completed",
            model=self._model_name,
            input_count=len(texts),
        )
        return [vector.tolist() for vector in vectors]


def create_embedding_gateway() -> EmbeddingGateway:
    """The one embedding gateway the whole app shares -- always local and
    always available. Unlike the old remote-endpoint gateway this replaces,
    there's no "unconfigured, fall back to lexical search" state any more
    (see documents.py, which now requires embeddings for every retrieval).
    """
    return LocalEmbeddingGateway()

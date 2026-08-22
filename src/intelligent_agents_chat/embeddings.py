"""Independent OpenAI-compatible embedding configuration and gateway."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from typing import Protocol, Sequence
from urllib.parse import urlsplit

from openai import AsyncOpenAI

from intelligent_agents_chat.logging_config import log_event, sanitized_endpoint


DEFAULT_EMBEDDING_BATCH_SIZE = 32
DEFAULT_EMBEDDING_TIMEOUT_SECONDS = 60.0
logger = logging.getLogger(__name__)


class EmbeddingError(RuntimeError):
    """The independently configured embedding service failed."""


class EmbeddingGateway(Protocol):
    @property
    def model_name(self) -> str: ...

    async def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


@dataclass(frozen=True, slots=True)
class EmbeddingSettings:
    base_url: str
    model: str
    api_key: str
    batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE
    timeout_seconds: float = DEFAULT_EMBEDDING_TIMEOUT_SECONDS
    expected_dimension: int | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.base_url and self.model)

    @classmethod
    def from_environment(cls) -> EmbeddingSettings:
        base_url = os.environ.get("RAG_EMBEDDING_BASE_URL", "").rstrip("/")
        model = os.environ.get("RAG_EMBEDDING_MODEL", "").strip()
        api_key = os.environ.get("RAG_EMBEDDING_API_KEY", "local-rag-placeholder")
        batch_size = _positive_int_env("RAG_EMBEDDING_BATCH_SIZE", DEFAULT_EMBEDDING_BATCH_SIZE)
        timeout_seconds = _positive_float_env(
            "RAG_EMBEDDING_TIMEOUT_SECONDS", DEFAULT_EMBEDDING_TIMEOUT_SECONDS
        )
        dimension_value = os.environ.get("RAG_EMBEDDING_DIMENSION", "").strip()
        expected_dimension = int(dimension_value) if dimension_value else None
        if expected_dimension is not None and expected_dimension <= 0:
            raise ValueError("RAG_EMBEDDING_DIMENSION must be positive")
        if bool(base_url) != bool(model):
            raise ValueError(
                "RAG_EMBEDDING_BASE_URL and RAG_EMBEDDING_MODEL must be configured together"
            )
        if base_url:
            parsed = urlsplit(base_url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError("RAG_EMBEDDING_BASE_URL must be an absolute HTTP(S) URL")
        return cls(
            base_url=base_url,
            model=model,
            api_key=api_key,
            batch_size=batch_size,
            timeout_seconds=timeout_seconds,
            expected_dimension=expected_dimension,
        )


class OpenAIEmbeddingGateway:
    """Batch embeddings through a service distinct from the generation profile."""

    def __init__(self, settings: EmbeddingSettings) -> None:
        if not settings.enabled:
            raise ValueError("Embedding settings are disabled")
        self.settings = settings

    @property
    def model_name(self) -> str:
        return self.settings.model

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        try:
            async with AsyncOpenAI(
                base_url=self.settings.base_url,
                api_key=self.settings.api_key,
                timeout=self.settings.timeout_seconds,
                max_retries=1,
            ) as client:
                for start in range(0, len(texts), self.settings.batch_size):
                    batch = list(texts[start : start + self.settings.batch_size])
                    response = await client.embeddings.create(
                        model=self.settings.model,
                        input=batch,
                    )
                    ordered = sorted(response.data, key=lambda item: item.index)
                    if len(ordered) != len(batch):
                        raise EmbeddingError("Embedding service returned an incomplete batch")
                    vectors.extend([list(item.embedding) for item in ordered])
        except EmbeddingError:
            raise
        except Exception as error:
            log_event(
                logger,
                logging.WARNING,
                "embeddings.request.failed",
                endpoint=sanitized_endpoint(self.settings.base_url),
                model=self.settings.model,
                input_count=len(texts),
                error_type=type(error).__name__,
            )
            raise EmbeddingError(f"Embedding service failed: {error}") from error

        dimensions = {len(vector) for vector in vectors}
        if len(dimensions) != 1 or 0 in dimensions:
            raise EmbeddingError("Embedding service returned inconsistent vector dimensions")
        dimension = dimensions.pop()
        if (
            self.settings.expected_dimension is not None
            and dimension != self.settings.expected_dimension
        ):
            raise EmbeddingError(
                "Embedding dimension mismatch: expected "
                f"{self.settings.expected_dimension}, received {dimension}"
            )
        log_event(
            logger,
            logging.INFO,
            "embeddings.request.completed",
            endpoint=sanitized_endpoint(self.settings.base_url),
            model=self.settings.model,
            input_count=len(texts),
            dimension=dimension,
        )
        return vectors


def create_embedding_gateway(
    settings: EmbeddingSettings | None = None,
) -> EmbeddingGateway | None:
    resolved = settings or EmbeddingSettings.from_environment()
    if not resolved.enabled:
        log_event(logger, logging.INFO, "embeddings.disabled")
        return None
    log_event(
        logger,
        logging.INFO,
        "embeddings.configured",
        endpoint=sanitized_endpoint(resolved.base_url),
        model=resolved.model,
        batch_size=resolved.batch_size,
        timeout_seconds=resolved.timeout_seconds,
        expected_dimension=resolved.expected_dimension,
    )
    return OpenAIEmbeddingGateway(resolved)


def _positive_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    value = int(raw) if raw else default
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _positive_float_env(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    value = float(raw) if raw else default
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value

"""Chat backend: model/repository bootstrap, request assembly, and reply streaming.

This is the seam where tool-calling, RAG, and context compression will be added later
(as a loop around `stream_reply` instead of a single `gateway.stream_reply` call) --
app.py should not need to change when that happens.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from importlib.metadata import version
import logging
import os
from pathlib import Path
import platform

from intelligent_agents_chat.adaptive_rag import (
    AdaptiveRAGController,
    AdaptiveRAGTrace,
    OpenAIAdaptiveRAGReasoner,
)
from intelligent_agents_chat.context import ContextAssembler, ContextPlan
from intelligent_agents_chat.database import ChatRepository, Conversation, Message
from intelligent_agents_chat.documents import (
    DEFAULT_DOCUMENT_ROOT,
    BlobStore,
    Chunker,
    DocumentParser,
    DocumentStore,
)
from intelligent_agents_chat.embeddings import create_embedding_gateway
from intelligent_agents_chat.llm import (
    MAX_TOKENS,
    SYSTEM_PROMPT,
    THINKING_MAX_TOKENS,
    VLLMGateway,
    check_model_available,
)
from intelligent_agents_chat.logging_config import configure_logging, log_event, sanitized_endpoint
from intelligent_agents_chat.memory import ProjectMemoryStore
from intelligent_agents_chat.models import DEFAULT_PROFILE_KEY, MODEL_PROFILES, ModelProfile
from intelligent_agents_chat.rag import DocumentService, ProjectRAGRetriever
from intelligent_agents_chat.retrieval import ContextCandidate, RetrievalQuery

PROFILE_STATUS_POLL_INTERVAL_SECONDS = 15.0

configure_logging()
logger = logging.getLogger(__name__)

log_event(
    logger,
    logging.INFO,
    "application.runtime.loaded",
    python_version=platform.python_version(),
    operating_system=platform.platform(),
    dependency_versions={
        "nicegui": version("nicegui"),
        "openai": version("openai"),
        "openpyxl": version("openpyxl"),
        "pypdf": version("pypdf"),
        "python-docx": version("python-docx"),
    },
)
log_event(
    logger,
    logging.INFO,
    "application.configuration.loaded",
    default_profile_key=DEFAULT_PROFILE_KEY,
    profile_count=len(MODEL_PROFILES),
    profiles=[
        {
            "key": profile.key,
            "model": profile.model,
            "endpoint": sanitized_endpoint(profile.base_url),
            "supports_thinking": profile.supports_thinking,
            "reasoning_effort": profile.reasoning_effort,
            "context_window_tokens": profile.context_window_tokens,
        }
        for profile in MODEL_PROFILES
    ],
    max_tokens=MAX_TOKENS,
    thinking_max_tokens=THINKING_MAX_TOKENS,
    system_prompt_chars=len(SYSTEM_PROMPT),
)

repository = ChatRepository()
try:
    repository.initialize()
except Exception:
    logger.exception(
        "application.database_initialization_failed",
        extra={
            "event": "application.database_initialization_failed",
            "database_path": str(repository.database_path),
        },
    )
    raise

gateway = VLLMGateway()
memory_store = ProjectMemoryStore(repository.database_path)
document_store = DocumentStore(repository.database_path)
document_store.initialize()
document_root = Path(os.environ.get("RAG_DOCUMENT_ROOT", str(DEFAULT_DOCUMENT_ROOT)))
embedding_gateway = create_embedding_gateway()
document_service = DocumentService(
    document_store,
    BlobStore(document_root),
    DocumentParser(),
    Chunker(),
    embedding_gateway,
)
rag_retriever = ProjectRAGRetriever(document_store, embedding_gateway)
adaptive_rag_controller = AdaptiveRAGController(
    rag_retriever,
    OpenAIAdaptiveRAGReasoner(gateway),
)
context_assembler = ContextAssembler(SYSTEM_PROMPT)
active_generations: set[str] = set()

try:
    rebuilt_entry_count = sum(
        memory_store.rebuild_project(project.id) for project in repository.list_projects()
    )
except Exception:
    logger.exception(
        "application.memory_backfill_failed",
        extra={
            "event": "application.memory_backfill_failed",
            "database_path": str(repository.database_path),
        },
    )
else:
    log_event(
        logger,
        logging.INFO,
        "application.memory_backfill_completed",
        entry_count=rebuilt_entry_count,
    )

# `None` until the first background check completes; then True/False. Mutated in place
# (never reassigned) so that importers of this dict see updates made by poll_profile_status().
profile_status: dict[str, bool | None] = {profile.key: None for profile in MODEL_PROFILES}

log_event(logger, logging.INFO, "application.initialized")


async def refresh_profile_status() -> None:
    """Check every profile endpoint concurrently and update `profile_status` in place."""
    results = await asyncio.gather(
        *(check_model_available(profile) for profile in MODEL_PROFILES),
        return_exceptions=True,
    )
    for profile, result in zip(MODEL_PROFILES, results, strict=True):
        if isinstance(result, BaseException):
            logger.exception(
                "chat.profile_status.check_failed",
                exc_info=result,
                extra={"event": "chat.profile_status.check_failed", "model_profile": profile.key},
            )
            is_available = False
        else:
            is_available = result
        previous = profile_status[profile.key]
        profile_status[profile.key] = is_available
        if previous is not None and previous != is_available:
            log_event(
                logger,
                logging.INFO if is_available else logging.WARNING,
                "chat.profile_status.changed",
                model_profile=profile.key,
                available=is_available,
            )


async def poll_profile_status() -> None:
    """Continuously refresh `profile_status` in the background."""
    while True:
        await refresh_profile_status()
        await asyncio.sleep(PROFILE_STATUS_POLL_INTERVAL_SECONDS)


def completion_messages(messages: list[Message]) -> list[dict[str, str]]:
    """Build the OpenAI-style message list for a completion request."""
    result: list[dict[str, str]] = []
    if SYSTEM_PROMPT:
        result.append({"role": "system", "content": SYSTEM_PROMPT})
    result.extend({"role": message.role, "content": message.content} for message in messages)
    return result


def rebuild_conversation_memory(conversation_id: str) -> int:
    """Refresh the rebuildable memory index for one changed conversation."""
    return memory_store.rebuild_conversation(conversation_id)


def retrieve_project_memory(
    *,
    project_id: str,
    conversation_id: str,
    query_text: str,
) -> list[ContextCandidate]:
    """Retrieve relevant turns from other chats in the same project."""
    return memory_store.retrieve(
        RetrievalQuery(
            project_id=project_id,
            text=query_text,
            exclude_conversation_id=conversation_id,
        )
    )


async def retrieve_project_documents(
    *,
    project_id: str,
    query_text: str,
) -> list[ContextCandidate]:
    """Retrieve relevant chunks from indexed documents in the same project."""
    return await rag_retriever.retrieve(
        RetrievalQuery(
            project_id=project_id,
            text=query_text,
        )
    )


def prepare_context(
    profile: ModelProfile,
    messages: list[Message],
    candidates: list[ContextCandidate],
    *,
    thinking_enabled: bool,
    retrieval_summary: str | None = None,
    adaptive_rag_trace: AdaptiveRAGTrace | None = None,
) -> ContextPlan:
    """Create a token-aware request while preserving auditable retrieval decisions."""
    output_reserve_tokens = THINKING_MAX_TOKENS if thinking_enabled else MAX_TOKENS
    plan = context_assembler.assemble(
        [{"role": message.role, "content": message.content} for message in messages],
        candidates,
        context_window_tokens=profile.context_window_tokens,
        output_reserve_tokens=output_reserve_tokens,
        retrieval_summary=retrieval_summary,
        adaptive_rag_trace=adaptive_rag_trace,
    )
    log_event(
        logger,
        logging.INFO,
        "chat.context.prepared",
        model_profile=profile.key,
        context_window_tokens=profile.context_window_tokens,
        output_reserve_tokens=output_reserve_tokens,
        input_budget_tokens=plan.input_budget_tokens,
        estimated_input_tokens=plan.estimated_input_tokens,
        history_message_count=len(messages),
        omitted_history_messages=plan.omitted_history_messages,
        retrieval_candidate_count=len(candidates),
        included_source_count=len(plan.included_sources),
        excluded_source_count=len(plan.excluded_sources),
        excluded_reasons=sorted({source.reason for source in plan.excluded_sources}),
        source_kinds=sorted({source.candidate.source_kind for source in plan.included_sources}),
        adaptive_rag_enabled=adaptive_rag_trace is not None,
        adaptive_rag_round_count=(
            len(adaptive_rag_trace.rounds) if adaptive_rag_trace is not None else 0
        ),
        adaptive_rag_evidence_sufficient=(
            adaptive_rag_trace.evidence_sufficient if adaptive_rag_trace is not None else None
        ),
        adaptive_rag_fallback_count=(
            len(adaptive_rag_trace.fallback_reasons) if adaptive_rag_trace is not None else 0
        ),
    )
    return plan


async def prepare_conversation_context(
    conversation: Conversation,
    profile: ModelProfile,
    messages: list[Message],
    query_text: str,
    *,
    request_id: str | None = None,
) -> ContextPlan:
    """Collect optional sources and assemble one request at the backend seam."""
    memory_candidates = (
        retrieve_project_memory(
            project_id=conversation.project_id,
            conversation_id=conversation.id,
            query_text=query_text,
        )
        if conversation.memory_enabled
        else []
    )
    adaptive_result = (
        await adaptive_rag_controller.run(
            project_id=conversation.project_id,
            query=query_text,
            history=[{"role": message.role, "content": message.content} for message in messages],
            profile=profile,
            request_id=request_id,
        )
        if conversation.rag_enabled
        else None
    )
    document_candidates = list(adaptive_result.candidates) if adaptive_result is not None else []
    candidates = _interleave_candidates(document_candidates, memory_candidates)
    return prepare_context(
        profile,
        messages,
        candidates,
        thinking_enabled=conversation.thinking_enabled,
        retrieval_summary=(adaptive_result.summary if adaptive_result is not None else None),
        adaptive_rag_trace=(adaptive_result.trace if adaptive_result is not None else None),
    )


def _interleave_candidates(
    primary: list[ContextCandidate],
    secondary: list[ContextCandidate],
) -> list[ContextCandidate]:
    """Give enabled retrieval sources a fair chance within the shared token budget."""
    result: list[ContextCandidate] = []
    for index in range(max(len(primary), len(secondary))):
        if index < len(primary):
            result.append(primary[index])
        if index < len(secondary):
            result.append(secondary[index])
    return result


def stream_reply(
    profile: ModelProfile,
    messages: list[dict[str, str]],
    *,
    request_id: str | None = None,
    thinking_enabled: bool = False,
) -> AsyncIterator[str]:
    """Stream the assistant's reply for an already-assembled message list."""
    return gateway.stream_reply(
        profile,
        messages,
        request_id=request_id,
        thinking_enabled=thinking_enabled,
    )

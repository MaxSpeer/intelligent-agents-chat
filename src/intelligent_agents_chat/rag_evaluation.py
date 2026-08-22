"""Small reproducible retrieval evaluation runnable before adding a reranker."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import json
import mimetypes
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Sequence

from intelligent_agents_chat.database import PROJECT_ROOT, ChatRepository
from intelligent_agents_chat.documents import BlobStore, Chunker, DocumentParser, DocumentStore
from intelligent_agents_chat.embeddings import create_embedding_gateway
from intelligent_agents_chat.rag import DocumentService, ProjectRAGRetriever
from intelligent_agents_chat.retrieval import RetrievalQuery


DEFAULT_EVALUATION_DATASET = PROJECT_ROOT / "evaluation" / "rag" / "cases.jsonl"
DEFAULT_EVALUATION_FIXTURES = PROJECT_ROOT / "evaluation" / "rag" / "fixtures"


@dataclass(frozen=True, slots=True)
class RetrievalEvaluationCase:
    query: str
    expected_documents: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RetrievalEvaluationResult:
    case_count: int
    expected_document_count: int
    recalled_document_count: int
    recall_at_k: float
    hit_rate_at_k: float
    mean_reciprocal_rank: float
    k: int


async def evaluate_retriever(
    retriever: ProjectRAGRetriever,
    project_id: str,
    cases: Sequence[RetrievalEvaluationCase],
    *,
    k: int = 6,
) -> RetrievalEvaluationResult:
    if k <= 0:
        raise ValueError("k must be positive")
    expected_count = 0
    recalled_count = 0
    case_hits = 0
    reciprocal_ranks: list[float] = []
    for case in cases:
        expected = set(case.expected_documents)
        expected_count += len(expected)
        candidates = await retriever.retrieve(
            RetrievalQuery(project_id=project_id, text=case.query, limit=k)
        )
        titles = [candidate.title for candidate in candidates]
        recalled = expected.intersection(titles)
        recalled_count += len(recalled)
        if recalled:
            case_hits += 1
        first_rank = next(
            (rank for rank, title in enumerate(titles, start=1) if title in expected),
            None,
        )
        reciprocal_ranks.append(1.0 / first_rank if first_rank is not None else 0.0)

    case_count = len(cases)
    return RetrievalEvaluationResult(
        case_count=case_count,
        expected_document_count=expected_count,
        recalled_document_count=recalled_count,
        recall_at_k=(recalled_count / expected_count if expected_count else 0.0),
        hit_rate_at_k=(case_hits / case_count if case_count else 0.0),
        mean_reciprocal_rank=(sum(reciprocal_ranks) / case_count if case_count else 0.0),
        k=k,
    )


def load_cases(path: Path) -> list[RetrievalEvaluationCase]:
    cases: list[RetrievalEvaluationCase] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            query = str(record["query"]).strip()
            expected = tuple(str(value) for value in record["expected_documents"])
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid evaluation case at line {line_number}") from error
        if not query or not expected:
            raise ValueError(f"Evaluation case at line {line_number} is empty")
        cases.append(RetrievalEvaluationCase(query=query, expected_documents=expected))
    if not cases:
        raise ValueError("Evaluation dataset is empty")
    return cases


async def run_fixture_evaluation(
    dataset_path: Path = DEFAULT_EVALUATION_DATASET,
    fixture_directory: Path = DEFAULT_EVALUATION_FIXTURES,
    *,
    k: int = 6,
) -> RetrievalEvaluationResult:
    cases = load_cases(dataset_path)
    with TemporaryDirectory() as temporary_directory:
        temporary_root = Path(temporary_directory)
        database_path = temporary_root / "evaluation.sqlite3"
        repository = ChatRepository(database_path)
        repository.initialize()
        project = repository.create_project("RAG retrieval evaluation")
        store = DocumentStore(database_path)
        store.initialize()
        embedding_gateway = create_embedding_gateway()
        service = DocumentService(
            store,
            BlobStore(temporary_root / "documents"),
            DocumentParser(),
            Chunker(),
            embedding_gateway,
        )
        fixture_paths = sorted(
            path
            for path in fixture_directory.iterdir()
            if path.is_file() and not path.name.startswith(".")
        )
        if not fixture_paths:
            raise ValueError("Evaluation fixture directory is empty")
        for path in fixture_paths:
            media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            await service.upload(
                project_id=project.id,
                display_name=path.name,
                media_type=media_type,
                data=path.read_bytes(),
            )
        return await evaluate_retriever(
            ProjectRAGRetriever(store, embedding_gateway),
            project.id,
            cases,
            k=k,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_EVALUATION_DATASET)
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_EVALUATION_FIXTURES)
    parser.add_argument("--k", type=int, default=6)
    arguments = parser.parse_args()
    result = asyncio.run(
        run_fixture_evaluation(arguments.dataset, arguments.fixtures, k=arguments.k)
    )
    print(
        json.dumps(
            {
                "case_count": result.case_count,
                "expected_document_count": result.expected_document_count,
                "recalled_document_count": result.recalled_document_count,
                "recall_at_k": result.recall_at_k,
                "hit_rate_at_k": result.hit_rate_at_k,
                "mean_reciprocal_rank": result.mean_reciprocal_rank,
                "k": result.k,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

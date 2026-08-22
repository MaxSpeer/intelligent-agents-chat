"""Tests for the bounded adaptive document-retrieval state machine."""

from collections.abc import Mapping, Sequence
import unittest

from intelligent_agents_chat.adaptive_rag import (
    AdaptiveRAGControlError,
    AdaptiveRAGController,
    EvidenceAssessment,
    RetrievalPlan,
    _json_object,
)
from intelligent_agents_chat.models import ModelProfile
from intelligent_agents_chat.retrieval import ContextCandidate, RetrievalQuery


PROFILE = ModelProfile(
    key="test",
    label="Test model",
    base_url="http://127.0.0.1:1/v1",
    model="test-model",
)


def candidate(source_id: str, text: str) -> ContextCandidate:
    return ContextCandidate(
        source_kind="project_document",
        source_id=source_id,
        project_id="project-a",
        text=text,
        title="inventory.csv",
        locator=f"row {source_id}",
        score=0.5,
    )


class RecordingRetriever:
    def __init__(self, responses: Mapping[str, list[ContextCandidate]]) -> None:
        self.responses = responses
        self.queries: list[RetrievalQuery] = []

    async def retrieve(self, query: RetrievalQuery) -> list[ContextCandidate]:
        self.queries.append(query)
        return list(self.responses.get(query.text, []))


class ScriptedReasoner:
    def __init__(
        self,
        *,
        plan: RetrievalPlan | Exception,
        assessments: Sequence[EvidenceAssessment | Exception] = (),
        summary: str | Exception = "Evidence summary",
    ) -> None:
        self.planned = plan
        self.assessments = list(assessments)
        self.summary = summary
        self.assessment_calls = 0
        self.summary_calls = 0

    async def plan(
        self,
        profile: ModelProfile,
        query: str,
        history: Sequence[Mapping[str, str]],
        *,
        request_id: str | None = None,
    ) -> RetrievalPlan:
        del profile, query, history, request_id
        if isinstance(self.planned, Exception):
            raise self.planned
        return self.planned

    async def assess(
        self,
        profile: ModelProfile,
        original_query: str,
        retrieval_query: str,
        candidates: Sequence[ContextCandidate],
        *,
        request_id: str | None = None,
    ) -> EvidenceAssessment:
        del profile, original_query, retrieval_query, candidates, request_id
        result = self.assessments[self.assessment_calls]
        self.assessment_calls += 1
        if isinstance(result, Exception):
            raise result
        return result

    async def summarize(
        self,
        profile: ModelProfile,
        original_query: str,
        candidates: Sequence[ContextCandidate],
        *,
        request_id: str | None = None,
    ) -> str:
        del profile, original_query, candidates, request_id
        self.summary_calls += 1
        if isinstance(self.summary, Exception):
            raise self.summary
        return self.summary


class AdaptiveRAGControllerTests(unittest.IsolatedAsyncioTestCase):
    async def test_judge_can_skip_document_retrieval(self) -> None:
        retriever = RecordingRetriever({})
        reasoner = ScriptedReasoner(
            plan=RetrievalPlan(False, "", "A greeting needs no project evidence.")
        )

        result = await AdaptiveRAGController(retriever, reasoner).run(
            project_id="project-a",
            query="Hello",
            history=[{"role": "user", "content": "Hello"}],
            profile=PROFILE,
        )

        self.assertEqual(retriever.queries, [])
        self.assertEqual(result.candidates, ())
        self.assertFalse(result.trace.retrieval_needed)
        self.assertEqual(result.trace.rounds, ())
        self.assertEqual(reasoner.summary_calls, 0)

    async def test_rewrites_query_and_runs_a_second_pass_when_evidence_is_incomplete(self) -> None:
        stock = candidate("stock", "Mondkeks has 7 units in stock.")
        duplicate = candidate("shared", "Product identifier: Mondkeks.")
        delivery = candidate("delivery", "Restock date: 2026-09-12.")
        retriever = RecordingRetriever(
            {
                "Mondkeks inventory": [stock, duplicate],
                "Mondkeks delivery date": [duplicate, delivery],
            }
        )
        reasoner = ScriptedReasoner(
            plan=RetrievalPlan(True, "Mondkeks inventory", "The answer is in project data."),
            assessments=[
                EvidenceAssessment(
                    False,
                    "The stock is known but the delivery date is missing.",
                    "delivery date",
                    "Mondkeks delivery date",
                ),
                EvidenceAssessment(True, "Both requested facts are present.", "", ""),
            ],
            summary="[stock] 7 units; [delivery] restock on 2026-09-12.",
        )

        result = await AdaptiveRAGController(retriever, reasoner).run(
            project_id="project-a",
            query="How many are there and when is it restocked?",
            history=[
                {"role": "user", "content": "Tell me about Mondkeks."},
                {"role": "assistant", "content": "What would you like to know?"},
            ],
            profile=PROFILE,
        )

        self.assertEqual(
            [query.text for query in retriever.queries],
            ["Mondkeks inventory", "Mondkeks delivery date"],
        )
        self.assertEqual(len(result.trace.rounds), 2)
        self.assertTrue(result.trace.evidence_sufficient)
        self.assertEqual(result.trace.rounds[1].new_result_count, 1)
        self.assertEqual(
            [item.source_id for item in result.candidates],
            ["stock", "delivery", "shared"],
        )
        self.assertEqual(result.summary, "[stock] 7 units; [delivery] restock on 2026-09-12.")

    async def test_control_failures_fall_back_without_losing_retrieved_evidence(self) -> None:
        source = candidate("one", "A useful fact.")
        retriever = RecordingRetriever({"Original question": [source]})
        reasoner = ScriptedReasoner(
            plan=RuntimeError("router offline"),
            assessments=[RuntimeError("judge offline")],
            summary=RuntimeError("summarizer offline"),
        )

        result = await AdaptiveRAGController(retriever, reasoner).run(
            project_id="project-a",
            query="Original question",
            history=[],
            profile=PROFILE,
        )

        self.assertEqual([query.text for query in retriever.queries], ["Original question"])
        self.assertEqual(result.candidates, (source,))
        self.assertIsNone(result.summary)
        self.assertEqual(len(result.trace.rounds), 1)
        self.assertIn("routing_failed:RuntimeError", result.trace.fallback_reasons)
        self.assertIn("assessment_failed:RuntimeError", result.trace.fallback_reasons)
        self.assertIn("summarization_failed:RuntimeError", result.trace.fallback_reasons)

    async def test_stops_without_a_second_assessment_when_follow_up_adds_no_evidence(self) -> None:
        source = candidate("one", "The only available fact.")
        retriever = RecordingRetriever({"first query": [source], "follow-up query": [source]})
        reasoner = ScriptedReasoner(
            plan=RetrievalPlan(True, "first query", "Documents are required."),
            assessments=[
                EvidenceAssessment(
                    False,
                    "A second fact is missing.",
                    "second fact",
                    "follow-up query",
                )
            ],
        )

        result = await AdaptiveRAGController(retriever, reasoner).run(
            project_id="project-a",
            query="Question",
            history=[],
            profile=PROFILE,
        )

        self.assertEqual(reasoner.assessment_calls, 1)
        self.assertEqual(len(result.trace.rounds), 2)
        self.assertEqual(result.trace.rounds[1].new_result_count, 0)
        self.assertIn("no new evidence", result.trace.rounds[1].assessment_reason)

    async def test_empty_rewrite_result_retries_original_query_before_model_assessment(
        self,
    ) -> None:
        source = candidate("one", "The requested value is 7.")
        retriever = RecordingRetriever({"translated query": [], "Original query": [source]})
        reasoner = ScriptedReasoner(
            plan=RetrievalPlan(True, "translated query", "Documents are required."),
            assessments=[EvidenceAssessment(True, "The value is present.", "", "")],
        )

        result = await AdaptiveRAGController(retriever, reasoner).run(
            project_id="project-a",
            query="Original query",
            history=[],
            profile=PROFILE,
        )

        self.assertEqual(
            [query.text for query in retriever.queries],
            ["translated query", "Original query"],
        )
        self.assertEqual(reasoner.assessment_calls, 1)
        self.assertTrue(result.trace.evidence_sufficient)
        self.assertEqual(result.candidates, (source,))


class ControlResponseParsingTests(unittest.TestCase):
    def test_extracts_json_from_a_fenced_control_response(self) -> None:
        self.assertEqual(
            _json_object('```json\n{"retrieve": true, "query": "inventory"}\n```'),
            {"retrieve": True, "query": "inventory"},
        )

    def test_rejects_a_control_response_without_json(self) -> None:
        with self.assertRaises(AdaptiveRAGControlError):
            _json_object("I would retrieve the documents.")


if __name__ == "__main__":
    unittest.main()

"""Evidence flow tests; fake scores test contracts, not model quality."""

from unittest.mock import MagicMock, patch

import pytest

from piragi import AsyncRagi, DecisionResult, Ragi
from piragi.evidence import NO_EVIDENCE, EvidenceJudge
from piragi.pipelines.retrieval import RetrievalPipeline
from piragi.retrieval import Retriever
from piragi.types import Citation


def judgment(p):
    return DecisionResult(choice="yes" if p >= 0.5 else "no", scores={"yes": p, "no": 1 - p})


@pytest.fixture
def candidates():
    return [Citation(source=f"doc{i}", chunk=f"passage {i}", score=1 - i / 10) for i in range(4)]


@pytest.fixture
def engine():
    engine = MagicMock()
    engine.decide_many.return_value = {
        f"c{i}": judgment(p) for i, p in enumerate([0.2, 0.95, 0.8, 0.6])
    }
    engine.decide.return_value = judgment(0.96)
    return engine


def test_batch_filter_then_answerability_on_exact_top_k(engine, candidates):
    selection = EvidenceJudge(engine).select("question", candidates, 1)
    assert [c.source for c in selection.citations] == ["doc1"]
    assert selection.report["status"] == "accepted"
    engine.decide_many.assert_called_once()
    assert len(engine.decide_many.call_args.kwargs["questions"]) == 4
    assert engine.decide.call_args.kwargs["state"] == {
        "query": "question",
        "passages": [{"source": "doc1", "text": "passage 1"}],
    }
    assert selection.citations[0].metadata["evidence_score"] == 0.95
    assert candidates[1].metadata == {}


@pytest.mark.parametrize("p", [0.05, 0.5, 0.65])
def test_insufficient_or_uncertain_answerability_abstains(engine, candidates, p):
    engine.decide.return_value = judgment(p)
    selection = EvidenceJudge(engine).select("question", candidates, 2)
    assert selection.abstained
    assert selection.citations == []


def test_all_rejected_skips_answerability(engine, candidates):
    engine.decide_many.return_value = {f"c{i}": judgment(0.1) for i in range(4)}
    assert EvidenceJudge(engine).select("question", candidates, 2).abstained
    engine.decide.assert_not_called()


@pytest.mark.parametrize("failure", ["missing", "invalid", "batch", "answerability"])
def test_invalid_or_unavailable_judge_is_explicit_fallback(engine, candidates, failure):
    if failure == "missing":
        engine.decide_many.return_value.pop("c0")
    elif failure == "invalid":
        engine.decide_many.return_value["c0"] = DecisionResult.model_construct(
            choice="yes", scores={"yes": float("nan"), "no": 0}
        )
    elif failure == "batch":
        engine.decide_many.side_effect = ImportError("no runtime")
    else:
        engine.decide.side_effect = ValueError("context too long")
    selection = EvidenceJudge(engine).select("question", candidates, 2)
    assert selection.report["status"] == "unavailable"
    assert not selection.abstained
    assert selection.citations == candidates[:2]


def pipeline(engine, candidates):
    store, embedder, client = MagicMock(), MagicMock(), MagicMock()
    store.search.return_value = candidates
    client.complete.return_value.choices[0].message.content = "Grounded answer"
    retriever = Retriever(llm_client=client, enable_query_expansion=False, enable_reranking=False)
    return RetrievalPipeline(
        embedder, store, retriever, evidence_judge=EvidenceJudge(engine)
    ), client


def test_pipeline_retrieves_wide_and_only_generates_with_surviving_evidence(engine, candidates):
    flow, client = pipeline(engine, candidates)
    answer = flow.ask("question", top_k=2, filters={"team": "docs"})
    assert flow.store.search.call_args.kwargs["top_k"] == 20
    assert flow.store.search.call_args.kwargs["filters"] == {"team": "docs"}
    assert [c.source for c in answer.citations] == ["doc1", "doc2"]
    prompt = client.complete.call_args.kwargs["messages"][1]["content"]
    assert "passage 0" not in prompt and "passage 3" not in prompt
    assert "passage 1" in prompt and "passage 2" in prompt
    client.complete.assert_called_once()


def test_pipeline_abstention_skips_llm(engine, candidates):
    flow, client = pipeline(engine, candidates)
    engine.decide.return_value = judgment(0.1)
    answer = flow.ask("question", top_k=2)
    assert answer.text == NO_EVIDENCE
    assert answer.citations == []
    assert answer.decision["status"] == "abstained"
    client.complete.assert_not_called()


def test_pure_retrieve_does_not_judge(engine, candidates):
    flow, client = pipeline(engine, candidates)
    assert len(flow.retrieve("question", top_k=2)) == 2
    engine.decide_many.assert_not_called()
    client.complete.assert_not_called()


def test_hyde_and_hierarchical_context_are_judged(engine, candidates):
    flow, _ = pipeline(engine, candidates)
    flow.hyde = MagicMock()
    flow.hyde.transform_query.return_value = "hypothetical"
    flow.use_hierarchical = True
    candidates[1].metadata["parent_text"] = "Actual parent evidence"
    answer = flow.ask("question", top_k=1)
    assert answer.citations[0].chunk == "Actual parent evidence"
    assert (
        "Actual parent evidence" in engine.decide_many.call_args.kwargs["questions"]["c1"].question
    )
    assert engine.decide_many.call_args.kwargs["state"] == {"query": "question"}


@pytest.mark.asyncio
async def test_sync_async_use_post_retrieval_engine(engine, candidates):
    with (
        patch("piragi.core.create_store"),
        patch("piragi.core.EmbeddingGenerator"),
        patch("piragi.core.Chunker"),
        patch("piragi.llm_client.LLMClient"),
    ):
        config = {"decision": {"engine": "openjev"}, "auto_update": {"enabled": False}}
        for cls in (Ragi, AsyncRagi):
            kb = cls(decision_engine=engine, config=config)
            sync = kb if cls is Ragi else kb._sync
            sync.llm_client.reset_mock()
            sync.store.search.return_value = candidates
            sync.llm_client.complete.return_value.choices[0].message.content = "Answer"
            answer = (
                kb.ask("question", top_k=1) if cls is Ragi else await kb.ask("question", top_k=1)
            )
            assert answer.decision["status"] == "accepted"
            assert answer.citations[0].source == "doc1"
            sync.llm_client.complete.assert_called_once()

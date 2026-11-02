"""Tests for StepBackPrompting integration in RetrievalConfig and RetrievalPipeline."""

from unittest.mock import MagicMock, patch

from piragi import Ragi, RagiConfig
from piragi.config import RetrievalConfig
from piragi.pipelines.retrieval import RetrievalPipeline
from piragi.query_transform import StepBackPrompting
from piragi.types import Answer, Citation


def test_retrieval_config_use_step_back_default():
    """Verify use_step_back defaults to False in RetrievalConfig."""
    cfg = RetrievalConfig()
    assert cfg.use_step_back is False


def test_retrieval_config_use_step_back_custom():
    """Verify use_step_back can be enabled in RagiConfig."""
    config = RagiConfig(retrieval={"use_step_back": True})
    assert config.retrieval.use_step_back is True


def test_ragi_initializes_step_back():
    """Verify Ragi instantiates StepBackPrompting when configured."""
    with patch("piragi.core.create_store"):
        kb = Ragi(config={"retrieval": {"use_step_back": True}})
        assert kb._use_step_back is True
        assert kb._step_back is not None
        assert isinstance(kb._step_back, StepBackPrompting)
        assert kb._retrieval_pipeline.step_back is kb._step_back


def test_pipeline_retrieve_executes_step_back():
    """Verify retrieve() searches both original and step-back queries."""
    mock_embedder = MagicMock()
    mock_embedder.embed_query.return_value = [0.1, 0.2]

    mock_store = MagicMock()
    c1 = Citation(source="doc1.txt", chunk="specific chunk", score=0.9)
    c2 = Citation(source="doc2.txt", chunk="abstract chunk", score=0.85)
    mock_store.search.side_effect = [[c1], [c2]]

    mock_retriever = MagicMock()
    mock_step_back = MagicMock()
    mock_step_back.get_queries.return_value = [
        "What is the timeout for cache in Redis?",
        "How does Redis caching work?"
    ]

    pipeline = RetrievalPipeline(
        embedder=mock_embedder,
        store=mock_store,
        retriever=mock_retriever,
        step_back=mock_step_back,
    )

    results = pipeline.retrieve("What is the timeout for cache in Redis?", top_k=5)
    mock_step_back.get_queries.assert_called_once_with("What is the timeout for cache in Redis?")
    assert mock_store.search.call_count == 2
    assert len(results) == 2
    assert {c.source for c in results} == {"doc1.txt", "doc2.txt"}


def test_pipeline_ask_merges_step_back_and_generates_answer():
    """Verify ask() fuses step-back results and generates grounded answer."""
    mock_embedder = MagicMock()
    mock_embedder.embed_query.return_value = [0.1, 0.2]

    mock_store = MagicMock()
    c1 = Citation(source="doc1.txt", chunk="specific chunk", score=0.9)
    c2 = Citation(source="doc2.txt", chunk="abstract chunk", score=0.85)
    mock_store.search.side_effect = [[c1], [c2]]

    mock_retriever = MagicMock()
    mock_retriever.expand_query.return_value = ["original query"]
    mock_retriever.generate_answer.return_value = Answer(
        text="Grounded answer citing both specific and abstract context.",
        citations=[c1, c2],
        query="original query",
    )

    mock_step_back = MagicMock()
    mock_step_back.generate_stepback_query.return_value = "abstract query"

    pipeline = RetrievalPipeline(
        embedder=mock_embedder,
        store=mock_store,
        retriever=mock_retriever,
        step_back=mock_step_back,
    )

    answer = pipeline.ask("original query", top_k=5)
    assert answer.text.startswith("Grounded answer")
    mock_step_back.generate_stepback_query.assert_called_once_with("original query")
    mock_retriever.generate_answer.assert_called_once()
    passed_citations = mock_retriever.generate_answer.call_args[1]["citations"]
    assert len(passed_citations) == 2


def test_pipeline_step_back_graceful_fallback_on_error():
    """Verify pipeline continues seamlessly if step-back generation fails."""
    mock_embedder = MagicMock()
    mock_embedder.embed_query.return_value = [0.1, 0.2]

    mock_store = MagicMock()
    c1 = Citation(source="doc1.txt", chunk="fallback chunk", score=0.9)
    mock_store.search.return_value = [c1]

    mock_retriever = MagicMock()
    mock_step_back = MagicMock()
    mock_step_back.get_queries.side_effect = RuntimeError("LLM API connection timed out")

    pipeline = RetrievalPipeline(
        embedder=mock_embedder,
        store=mock_store,
        retriever=mock_retriever,
        step_back=mock_step_back,
    )

    results = pipeline.retrieve("query", top_k=5)
    assert len(results) == 1
    assert results[0].chunk == "fallback chunk"

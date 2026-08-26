import pytest
from unittest.mock import MagicMock
from piragi.pipelines.retrieval import RetrievalPipeline
from piragi.types import Citation, Answer

@pytest.fixture
def mock_dependencies():
    embedder = MagicMock()
    embedder.embed_query.return_value = [0.1, 0.2]
    
    store = MagicMock()
    citation = Citation(source="doc1.txt", chunk="text chunk", score=0.9, metadata={})
    store.search.return_value = [citation]
    
    retriever = MagicMock()
    retriever.expand_query.return_value = ["original query"]
    retriever.generate_answer.return_value = Answer(text="answer", citations=[citation], query="query")
    
    return {
        "embedder": embedder,
        "store": store,
        "retriever": retriever,
        "citation": citation
    }

def test_basic_retrieve(mock_dependencies):
    pipeline = RetrievalPipeline(
        embedder=mock_dependencies["embedder"],
        store=mock_dependencies["store"],
        retriever=mock_dependencies["retriever"]
    )
    
    results = pipeline.retrieve("test query")
    
    assert len(results) == 1
    assert results[0].source == "doc1.txt"
    mock_dependencies["embedder"].embed_query.assert_called_once_with("test query")
    mock_dependencies["store"].search.assert_called_once()

def test_retrieve_with_hyde(mock_dependencies):
    hyde = MagicMock()
    hyde.transform_query.return_value = "hypothetical doc"
    
    pipeline = RetrievalPipeline(
        embedder=mock_dependencies["embedder"],
        store=mock_dependencies["store"],
        retriever=mock_dependencies["retriever"],
        hyde=hyde
    )
    
    pipeline.retrieve("test query")
    hyde.transform_query.assert_called_once_with("test query")
    mock_dependencies["embedder"].embed_query.assert_called_once_with("hypothetical doc")

def test_retrieve_with_hybrid_search(mock_dependencies):
    hybrid_searcher = MagicMock()
    hybrid_searcher.search.return_value = [mock_dependencies["citation"]]
    
    pipeline = RetrievalPipeline(
        embedder=mock_dependencies["embedder"],
        store=mock_dependencies["store"],
        retriever=mock_dependencies["retriever"],
        hybrid_searcher=hybrid_searcher
    )
    
    pipeline.retrieve("test query")
    hybrid_searcher.search.assert_called_once()

def test_retrieve_with_cross_encoder(mock_dependencies):
    cross_encoder = MagicMock()
    cross_encoder.rerank.return_value = [mock_dependencies["citation"]]
    
    pipeline = RetrievalPipeline(
        embedder=mock_dependencies["embedder"],
        store=mock_dependencies["store"],
        retriever=mock_dependencies["retriever"],
        cross_encoder=cross_encoder
    )
    
    pipeline.retrieve("test query")
    cross_encoder.rerank.assert_called_once()

def test_ask_generates_answer(mock_dependencies):
    pipeline = RetrievalPipeline(
        embedder=mock_dependencies["embedder"],
        store=mock_dependencies["store"],
        retriever=mock_dependencies["retriever"]
    )
    
    answer = pipeline.ask("test query")
    assert answer.text == "answer"
    mock_dependencies["retriever"].generate_answer.assert_called_once()

def test_ask_with_filters(mock_dependencies):
    pipeline = RetrievalPipeline(
        embedder=mock_dependencies["embedder"],
        store=mock_dependencies["store"],
        retriever=mock_dependencies["retriever"]
    )
    
    filters = {"type": "api"}
    pipeline.ask("test query", filters=filters)
    
    mock_dependencies["store"].search.assert_called_once_with(
        query_embedding=[0.1, 0.2],
        top_k=5,
        filters=filters
    )

def test_retrieve_bm25_only_mode_uses_bm25_index(mock_dependencies):
    bm25_index = MagicMock()
    bm25_index.search.return_value = [mock_dependencies["citation"]]

    pipeline = RetrievalPipeline(
        embedder=mock_dependencies["embedder"],
        store=mock_dependencies["store"],
        retriever=mock_dependencies["retriever"],
        mode="bm25_only",
        bm25_index=bm25_index,
    )

    results = pipeline.retrieve("test query", top_k=3)

    bm25_index.search.assert_called_once_with("test query", top_k=3)
    mock_dependencies["store"].search.assert_not_called()
    assert results == [mock_dependencies["citation"]]


def test_retrieve_on_the_fly_mode_uses_on_the_fly_index(mock_dependencies):
    on_the_fly_index = MagicMock()
    on_the_fly_index.search.return_value = [mock_dependencies["citation"]]

    pipeline = RetrievalPipeline(
        embedder=mock_dependencies["embedder"],
        store=mock_dependencies["store"],
        retriever=mock_dependencies["retriever"],
        mode="on_the_fly",
        on_the_fly_index=on_the_fly_index,
    )

    results = pipeline.retrieve("test query", top_k=3)

    on_the_fly_index.search.assert_called_once_with(
        mock_dependencies["embedder"], "test query", top_k=3
    )
    mock_dependencies["store"].search.assert_not_called()
    assert results == [mock_dependencies["citation"]]


def test_retrieve_hot_cold_mode_merges_hot_and_cold_sorted_by_score(mock_dependencies):
    hot_citation = Citation(source="hot.txt", chunk="hot chunk", score=0.5, metadata={})
    cold_citation = Citation(source="cold.txt", chunk="cold chunk", score=0.9, metadata={})
    mock_dependencies["store"].search.return_value = [hot_citation]

    on_the_fly_index = MagicMock()
    on_the_fly_index.count.return_value = 1
    on_the_fly_index.search.return_value = [cold_citation]

    pipeline = RetrievalPipeline(
        embedder=mock_dependencies["embedder"],
        store=mock_dependencies["store"],
        retriever=mock_dependencies["retriever"],
        mode="hot_cold",
        on_the_fly_index=on_the_fly_index,
    )

    results = pipeline.retrieve("test query", top_k=2)

    # cold_citation scores higher, should be first
    assert [c.source for c in results] == ["cold.txt", "hot.txt"]


def test_retrieve_hot_cold_mode_skips_cold_when_on_the_fly_empty(mock_dependencies):
    on_the_fly_index = MagicMock()
    on_the_fly_index.count.return_value = 0

    pipeline = RetrievalPipeline(
        embedder=mock_dependencies["embedder"],
        store=mock_dependencies["store"],
        retriever=mock_dependencies["retriever"],
        mode="hot_cold",
        on_the_fly_index=on_the_fly_index,
    )

    pipeline.retrieve("test query")

    on_the_fly_index.search.assert_not_called()


def test_non_dense_search_applies_cross_encoder(mock_dependencies):
    bm25_index = MagicMock()
    bm25_index.search.return_value = [mock_dependencies["citation"]]
    cross_encoder = MagicMock()
    cross_encoder.rerank.return_value = [mock_dependencies["citation"]]

    pipeline = RetrievalPipeline(
        embedder=mock_dependencies["embedder"],
        store=mock_dependencies["store"],
        retriever=mock_dependencies["retriever"],
        mode="bm25_only",
        bm25_index=bm25_index,
        cross_encoder=cross_encoder,
    )

    pipeline.retrieve("test query", top_k=3)

    cross_encoder.rerank.assert_called_once_with(
        query="test query", citations=[mock_dependencies["citation"]], top_k=3
    )


def test_non_dense_search_cross_encoder_failure_falls_back(mock_dependencies):
    bm25_index = MagicMock()
    bm25_index.search.return_value = [mock_dependencies["citation"]]
    cross_encoder = MagicMock()
    cross_encoder.rerank.side_effect = Exception("reranker down")

    pipeline = RetrievalPipeline(
        embedder=mock_dependencies["embedder"],
        store=mock_dependencies["store"],
        retriever=mock_dependencies["retriever"],
        mode="bm25_only",
        bm25_index=bm25_index,
        cross_encoder=cross_encoder,
    )

    results = pipeline.retrieve("test query", top_k=3)

    assert results == [mock_dependencies["citation"]]


def test_query_rewriter_used_in_non_dense_search(mock_dependencies):
    bm25_index = MagicMock()
    bm25_index.search.return_value = [mock_dependencies["citation"]]
    query_rewriter = MagicMock()
    query_rewriter.rewrite.return_value = "rewritten keyword query"

    pipeline = RetrievalPipeline(
        embedder=mock_dependencies["embedder"],
        store=mock_dependencies["store"],
        retriever=mock_dependencies["retriever"],
        mode="bm25_only",
        bm25_index=bm25_index,
        query_rewriter=query_rewriter,
    )

    pipeline.retrieve("what's the deal with AUTH_4011", top_k=3)

    query_rewriter.rewrite.assert_called_once_with("what's the deal with AUTH_4011")
    bm25_index.search.assert_called_once_with("rewritten keyword query", top_k=3)


def test_query_rewriter_failure_falls_back_to_original_query(mock_dependencies):
    bm25_index = MagicMock()
    bm25_index.search.return_value = [mock_dependencies["citation"]]
    query_rewriter = MagicMock()
    query_rewriter.rewrite.side_effect = Exception("llm unreachable")

    pipeline = RetrievalPipeline(
        embedder=mock_dependencies["embedder"],
        store=mock_dependencies["store"],
        retriever=mock_dependencies["retriever"],
        mode="bm25_only",
        bm25_index=bm25_index,
        query_rewriter=query_rewriter,
    )

    pipeline.retrieve("original query", top_k=3)

    bm25_index.search.assert_called_once_with("original query", top_k=3)


def test_dense_hybrid_search_receives_rewritten_query(mock_dependencies):
    hybrid_searcher = MagicMock()
    hybrid_searcher.search.return_value = [mock_dependencies["citation"]]
    query_rewriter = MagicMock()
    query_rewriter.rewrite.return_value = "rewritten query"

    pipeline = RetrievalPipeline(
        embedder=mock_dependencies["embedder"],
        store=mock_dependencies["store"],
        retriever=mock_dependencies["retriever"],
        hybrid_searcher=hybrid_searcher,
        query_rewriter=query_rewriter,
    )

    pipeline.retrieve("original query")

    hybrid_searcher.search.assert_called_once_with(
        query="rewritten query",
        vector_citations=[mock_dependencies["citation"]],
        top_k=5,
    )


def test_ask_bm25_only_mode_generates_answer_from_non_dense_search(mock_dependencies):
    bm25_index = MagicMock()
    bm25_index.search.return_value = [mock_dependencies["citation"]]

    pipeline = RetrievalPipeline(
        embedder=mock_dependencies["embedder"],
        store=mock_dependencies["store"],
        retriever=mock_dependencies["retriever"],
        mode="bm25_only",
        bm25_index=bm25_index,
    )

    answer = pipeline.ask("test query")

    assert answer.text == "answer"
    mock_dependencies["store"].search.assert_not_called()
    mock_dependencies["retriever"].generate_answer.assert_called_once_with(
        query="test query",
        citations=[mock_dependencies["citation"]],
        system_prompt=None,
    )


def test_ragi_ask_uses_pipeline():
    from piragi import Ragi
    from unittest.mock import patch
    
    # Using a dummy persist dir to avoid actually hitting standard persist directory
    with patch("piragi.pipelines.retrieval.RetrievalPipeline") as MockPipeline:
        with patch("piragi.core.create_store") as mock_create_store:
            mock_pipeline_instance = MockPipeline.return_value
            mock_pipeline_instance.ask.return_value = Answer(text="mocked answer", citations=[], query="query")
            
            kb = Ragi(store="test") 
            answer = kb.ask("test query")
            
            assert answer.text == "mocked answer"
            mock_pipeline_instance.ask.assert_called_once()



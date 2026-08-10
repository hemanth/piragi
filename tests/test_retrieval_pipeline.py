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



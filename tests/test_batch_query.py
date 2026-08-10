import pytest
import time
from unittest.mock import Mock, patch

from piragi.pipelines.retrieval import RetrievalPipeline
from piragi.types import Citation

class DummyEmbedder:
    def embed_query(self, query):
        if query == "fail":
            raise ValueError("Embedding failed")
        return [0.1, 0.2, 0.3]

class DummyStore:
    def __init__(self, delay=0.0):
        self.delay = delay
        
    def search(self, query_embedding, top_k, filters=None):
        if self.delay > 0:
            time.sleep(self.delay)
            
        # Return different citations based on a fake query characteristic, or just mock it
        # We can just return a standard citation for testing deduplication
        return [
            Citation(source="doc1", chunk="chunk A", score=0.9),
            Citation(source="doc2", chunk="chunk B", score=0.8),
        ]

class DummyRetriever:
    def expand_query(self, query):
        return [query, query + " 1", query + " 2"]
        
    def generate_answer(self, query, citations, system_prompt=None):
        from piragi.types import Answer
        return Answer(text="answer", citations=citations, query=query)

def test_parallel_search_produces_same_results():
    embedder = DummyEmbedder()
    store = DummyStore()
    retriever = DummyRetriever()
    
    # Run parallel
    pipeline = RetrievalPipeline(embedder, store, retriever, max_parallel_queries=4)
    # mock the expand_query to just use standard 3 queries
    
    pipeline.retriever.expand_query = Mock(return_value=["q1", "q2", "q3"])
    
    # Let store return different results per query
    def mock_search(query_embedding, top_k, filters=None):
        return [Citation(source="doc1", chunk="chunk A", score=0.9)]
        
    store.search = Mock(side_effect=mock_search)
    
    citations = pipeline.retrieve("test query")
    assert len(citations) == 1
    assert citations[0].source == "doc1"

def test_parallel_search_deduplicates():
    embedder = DummyEmbedder()
    store = DummyStore()
    retriever = DummyRetriever()
    
    pipeline = RetrievalPipeline(embedder, store, retriever)
    pipeline.retriever.expand_query = Mock(return_value=["q1", "q2"])
    
    # Store returns exact same citations for both queries
    citations = pipeline.retrieve("test query", top_k=5)
    
    # Should only have 2 citations total, not 4
    assert len(citations) == 2
    
def test_parallel_search_handles_failures():
    embedder = DummyEmbedder()
    store = DummyStore()
    retriever = DummyRetriever()
    
    pipeline = RetrievalPipeline(embedder, store, retriever)
    
    # One query will fail in embedder
    pipeline.retriever.expand_query = Mock(return_value=["q1", "fail", "q3"])
    
    citations = pipeline.retrieve("test query", top_k=5)
    
    # Should still get results from q1 and q3 (deduplicated)
    assert len(citations) == 2

@patch("piragi.pipelines.retrieval.as_completed")
@patch("piragi.pipelines.retrieval.ThreadPoolExecutor")
def test_single_query_no_parallelism(mock_executor, mock_as_completed):
    mock_as_completed.return_value = []
    
    embedder = DummyEmbedder()
    store = DummyStore()
    retriever = DummyRetriever()
    
    pipeline = RetrievalPipeline(embedder, store, retriever)
    # Ensure it only uses 1 worker if 1 query
    pipeline.retriever.expand_query = Mock(return_value=["q1"])
    
    pipeline.retrieve("test query")
    
    mock_executor.assert_called_once_with(max_workers=1)

@patch("piragi.pipelines.retrieval.as_completed")
@patch("piragi.pipelines.retrieval.ThreadPoolExecutor")
def test_max_workers_respected(mock_executor, mock_as_completed):
    mock_as_completed.return_value = []
    
    embedder = DummyEmbedder()
    store = DummyStore()
    retriever = DummyRetriever()
    
    # Set max_parallel_queries=2, but have 5 queries
    pipeline = RetrievalPipeline(embedder, store, retriever, max_parallel_queries=2)
    # ask() uses expand_query which returns 3 variations from DummyRetriever.
    # Since max_parallel_queries=2, it should cap workers at 2.

    pipeline.ask("test query")
    mock_executor.assert_called_once_with(max_workers=2)

def test_parallel_faster_than_sequential():
    embedder = DummyEmbedder()
    store = DummyStore(delay=0.1)  # 100ms delay per search
    retriever = DummyRetriever()
    
    pipeline = RetrievalPipeline(embedder, store, retriever, max_parallel_queries=4)
    pipeline.retriever.expand_query = Mock(return_value=["q1", "q2", "q3", "q4"])
    
    start_time = time.time()
    pipeline.ask("test query")
    duration = time.time() - start_time
    
    # Sequential would take > 0.4s. Parallel should take ~0.1s + overhead.
    assert duration < 0.35, "Parallel search took too long: {}s".format(duration)

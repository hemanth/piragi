"""
piragi - Zero-setup RAG library with auto-chunking, embeddings, and smart citations.

Example:
    >>> from piragi import Ragi
    >>>
    >>> # One-liner setup and query
    >>> kb = Ragi("./docs")
    >>> answer = kb.ask("How do I install this?")
    >>>
    >>> # Access answer and citations
    >>> print(answer.text)
    >>> for citation in answer.citations:
    ...     print(f"Source: {citation.source}")
    ...     print(f"Relevance: {citation.score:.2f}")
    >>>
    >>> # Callable shorthand
    >>> answer = kb("What's the API?")
    >>>
    >>> # Filter by metadata
    >>> answer = kb.filter(type="documentation").ask("How to configure?")
    >>>
    >>> # Advanced retrieval with HyDE, hybrid search, and cross-encoder
    >>> kb = Ragi("./docs", config={
    ...     "retrieval": {
    ...         "use_hyde": True,
    ...         "use_hybrid_search": True,
    ...         "use_cross_encoder": True,
    ...     }
    ... })
"""

from .core import Ragi
from .types import Answer, Citation

# Advanced components (optional imports)
from .reranker import CrossEncoderReranker, TFIDFReranker, HybridReranker
from .hybrid_search import BM25, HybridSearcher
from .query_transform import HyDE, QueryExpander, MultiQueryRetriever, StepBackPrompting
from .semantic_chunking import (
    SemanticChunker,
    ContextualChunker,
    PropositionChunker,
    HierarchicalChunker,
)

__version__ = "0.2.1"
__all__ = [
    # Core
    "Ragi",
    "Answer",
    "Citation",
    # Reranking
    "CrossEncoderReranker",
    "TFIDFReranker",
    "HybridReranker",
    # Hybrid search
    "BM25",
    "HybridSearcher",
    # Query transformation
    "HyDE",
    "QueryExpander",
    "MultiQueryRetriever",
    "StepBackPrompting",
    # Chunking strategies
    "SemanticChunker",
    "ContextualChunker",
    "PropositionChunker",
    "HierarchicalChunker",
]

"""Shared fixtures, capability probes, and test infrastructure for E2E tests."""

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest
import numpy as np

from piragi.config import RagiConfig, RetrievalConfig
from piragi.types import Answer, Citation, Chunk


# ---------------------------------------------------------------------------
# Feature Capability Probes (Progressive Testability)
# ---------------------------------------------------------------------------

def has_u_shaped_support() -> bool:
    """Check if U-shaped context attention ordering is implemented."""
    cfg = RetrievalConfig()
    return hasattr(cfg, "context_ordering")


def has_auto_merge_support() -> bool:
    """Check if hierarchical auto-merging deduplication is implemented."""
    cfg = RetrievalConfig()
    return hasattr(cfg, "auto_merge_parents")


def has_step_back_support() -> bool:
    """Check if StepBackPrompting is integrated into RetrievalConfig."""
    cfg = RetrievalConfig()
    return hasattr(cfg, "use_step_back")


def has_citation_pruning_support() -> bool:
    """Check if inline citation attribution pruning is implemented."""
    cfg = RetrievalConfig()
    return hasattr(cfg, "prune_unreferenced_citations")


def has_bm25_caching_support() -> bool:
    """Check if BM25 corpus index caching is implemented in HybridSearcher."""
    from piragi.hybrid_search import HybridSearcher
    return (
        hasattr(HybridSearcher, "invalidate_bm25_cache")
        or hasattr(HybridSearcher, "_bm25_cached")
        or hasattr(HybridSearcher, "_tokenized_corpus")
    )


def has_gcs_support() -> bool:
    """Check if LanceStore supports gs:// and gcs:// remote storage."""
    from piragi.stores.lance import LanceStore
    import inspect
    src = inspect.getsource(LanceStore.__init__)
    return "gs://" in src or "gcs://" in src


def has_vertex_embeddings_support() -> bool:
    """Check if EmbeddingGenerator supports vertexai backend."""
    from piragi.embeddings import EmbeddingGenerator
    import inspect
    src = inspect.getsource(EmbeddingGenerator.__init__)
    return "vertexai" in src or "vertex" in src


def has_vertex_dimension_support() -> bool:
    """Check if text-embedding-004 is registered in EMBEDDING_DIMENSIONS."""
    from piragi.stores.lance import EMBEDDING_DIMENSIONS
    return "text-embedding-004" in EMBEDDING_DIMENSIONS


def has_adc_token_refresh_support() -> bool:
    """Check if LLMClient supports ADC token refresh."""
    from piragi.llm_client import LLMClient
    import inspect
    src = inspect.getsource(LLMClient.__init__)
    return "vertexai" in src or "refresh" in src or "google.auth" in src


def has_benchmark_harness_support() -> bool:
    """Check if evals/eval_benchmark.py exists."""
    try:
        import evals.eval_benchmark
        return True
    except (ImportError, ModuleNotFoundError):
        benchmark_file = Path(__file__).resolve().parent.parent.parent / "evals" / "eval_benchmark.py"
        return benchmark_file.is_file()


def has_leaderboard_exporters_support() -> bool:
    """Check if evals/leaderboard_exporters.py exists."""
    try:
        import evals.leaderboard_exporters
        return True
    except (ImportError, ModuleNotFoundError):
        exporter_file = Path(__file__).resolve().parent.parent.parent / "evals" / "leaderboard_exporters.py"
        return exporter_file.is_file()


# ---------------------------------------------------------------------------
# Reusable Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_768_embeddings() -> List[float]:
    """Deterministic 768-dimensional normalized embedding vector."""
    vec = [0.05] * 768
    vec[0] = 0.5
    vec[1] = 0.2
    return vec


def make_deterministic_embedding(text: str, dim: int = 768) -> List[float]:
    """Generate a deterministic normalized vector from text content."""
    seed = sum(ord(c) for c in text) % 100000
    rng = np.random.RandomState(seed)
    vec = rng.randn(dim).astype(np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.tolist()


@pytest.fixture
def make_mock_embedder(mock_768_embeddings):
    """Factory for fast in-memory mock EmbeddingGenerators."""
    def _create_embedder(dim: int = 768):
        mock_gen = MagicMock()
        mock_gen.batch_size = 32

        norm = float(np.linalg.norm(mock_768_embeddings))
        norm_vec = [v / norm for v in mock_768_embeddings]

        def embed_chunks_side_effect(chunks, on_progress=None):
            for i, chunk in enumerate(chunks):
                chunk.embedding = list(norm_vec)
                if on_progress:
                    on_progress(f"Embedded {i+1}/{len(chunks)} chunks")
            return chunks

        def embed_query_side_effect(query):
            return list(norm_vec)

        mock_gen.embed_chunks.side_effect = embed_chunks_side_effect
        mock_gen.embed_query.side_effect = embed_query_side_effect
        return mock_gen

    return _create_embedder


@pytest.fixture
def e2e_temp_dir():
    """Create an isolated temporary directory for E2E tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_openai_response():
    """Factory for mocked OpenAI LLM chat completion response."""
    def _make_response(text: str = "According to Source 1, RAG provides grounded facts."):
        mock_completion = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = text
        mock_completion.choices = [mock_choice]
        return mock_completion

    return _make_response


@pytest.fixture
def sample_corpus_dir(e2e_temp_dir):
    """Generate a realistic multi-document sample corpus."""
    corpus_path = os.path.join(e2e_temp_dir, "corpus")
    os.makedirs(corpus_path, exist_ok=True)

    # Document 1: Architecture guide
    with open(os.path.join(corpus_path, "architecture.md"), "w", encoding="utf-8") as f:
        f.write("""# Piragi Architecture Overview

Piragi is designed with a layered pipeline architecture:
- Layer 1: Ingestion and multi-format document chunking.
- Layer 2: Hybrid dense and sparse vector retrieval.
- Layer 3: Neural reranking and context assembly.
- Layer 4: Synthesis and attribution pruning.

## Hierarchical Indexing
Hierarchical indexing chunks documents into large parent sections (1500 chars)
and small child chunks (300 chars) to balance embedding precision with rich context.
""")

    # Document 2: Deployment options
    with open(os.path.join(corpus_path, "deployment.txt"), "w", encoding="utf-8") as f:
        f.write("""Deployment Guide:
1. Local: LanceDB file-based storage in .piragi/ directory.
2. Cloud Storage: Remote LanceDB using gs:// and s3:// object storage buckets.
3. Vertex AI: text-embedding-004 integration with Application Default Credentials (ADC).
""")

    # Document 3: API reference
    with open(os.path.join(corpus_path, "api_reference.md"), "w", encoding="utf-8") as f:
        f.write("""# API Reference

The core facade is Ragi:
- `kb = Ragi(sources=["./docs"])`
- `answer = kb.ask("How do I configure Piragi?")`
- `kb.filter(source="api_reference.md")`
- `kb.clear()`
""")

    return corpus_path

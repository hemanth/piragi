"""Integration tests for advanced RAG retrieval features."""

import os
import pytest
import tempfile
from unittest.mock import MagicMock, patch
import numpy as np

from piragi import Ragi
from piragi.types import Answer, Citation


@pytest.fixture
def temp_dir():
    """Create a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_docs_dir(temp_dir):
    """Create sample documents for testing."""
    docs_dir = os.path.join(temp_dir, "docs")
    os.makedirs(docs_dir)

    # Create sample markdown files
    doc1 = os.path.join(docs_dir, "python.md")
    with open(doc1, "w") as f:
        f.write("""# Python Guide

Python is a high-level programming language known for its clear syntax.
It supports multiple programming paradigms including procedural and object-oriented.

## Installation

To install Python, download the installer from python.org.
Run the installer and follow the prompts.

## Basic Syntax

Python uses indentation for code blocks instead of braces.
Variables are dynamically typed.
""")

    doc2 = os.path.join(docs_dir, "javascript.md")
    with open(doc2, "w") as f:
        f.write("""# JavaScript Guide

JavaScript is a scripting language primarily used for web development.
It runs in web browsers and enables interactive web pages.

## Installation

JavaScript runs in browsers, no installation needed.
For server-side, install Node.js from nodejs.org.

## Basic Syntax

JavaScript uses curly braces for code blocks.
Variables can be declared with var, let, or const.
""")

    return docs_dir


@pytest.fixture
def mock_embeddings():
    """Mock embedding responses (768-dim for all-mpnet-base-v2)."""
    return np.random.rand(768).tolist()


@pytest.fixture
def mock_llm_response():
    """Mock LLM response."""
    return "Python is a high-level programming language known for its clear syntax."


def create_mock_embedding_generator(mock_embeddings):
    """Create a mock EmbeddingGenerator for testing."""
    mock_gen = MagicMock()

    def embed_chunks_side_effect(chunks):
        for chunk in chunks:
            chunk.embedding = mock_embeddings
        return chunks

    mock_gen.embed_chunks.side_effect = embed_chunks_side_effect
    mock_gen.embed_query.return_value = mock_embeddings
    return mock_gen


class TestRagiAdvancedConfig:
    """Tests for Ragi with advanced configuration."""

    @patch("piragi.retrieval.OpenAI")
    @patch("piragi.core.EmbeddingGenerator")
    def test_init_with_hyde(self, mock_embed_gen, mock_openai, temp_dir, mock_embeddings):
        """Test initialization with HyDE enabled."""
        mock_embed_gen.return_value = create_mock_embedding_generator(mock_embeddings)

        kb = Ragi(
            persist_dir=os.path.join(temp_dir, "store"),
            config={
                "retrieval": {"use_hyde": True},
            }
        )

        assert kb._use_hyde is True
        assert kb._hyde is not None

    @patch("piragi.retrieval.OpenAI")
    @patch("piragi.core.EmbeddingGenerator")
    def test_init_with_hybrid_search(self, mock_embed_gen, mock_openai, temp_dir, mock_embeddings):
        """Test initialization with hybrid search enabled."""
        mock_embed_gen.return_value = create_mock_embedding_generator(mock_embeddings)

        kb = Ragi(
            persist_dir=os.path.join(temp_dir, "store"),
            config={
                "retrieval": {"use_hybrid_search": True},
            }
        )

        assert kb._use_hybrid_search is True
        assert kb._hybrid_searcher is not None

    @patch("piragi.retrieval.OpenAI")
    @patch("piragi.core.EmbeddingGenerator")
    def test_init_with_cross_encoder(self, mock_embed_gen, mock_openai, temp_dir, mock_embeddings):
        """Test initialization with cross-encoder reranking enabled."""
        mock_embed_gen.return_value = create_mock_embedding_generator(mock_embeddings)

        kb = Ragi(
            persist_dir=os.path.join(temp_dir, "store"),
            config={
                "retrieval": {"use_cross_encoder": True},
            }
        )

        assert kb._use_cross_encoder is True
        assert kb._cross_encoder is not None

    @patch("piragi.retrieval.OpenAI")
    @patch("piragi.core.EmbeddingGenerator")
    def test_init_all_advanced_features(self, mock_embed_gen, mock_openai, temp_dir, mock_embeddings):
        """Test initialization with all advanced features."""
        mock_embed_gen.return_value = create_mock_embedding_generator(mock_embeddings)

        kb = Ragi(
            persist_dir=os.path.join(temp_dir, "store"),
            config={
                "retrieval": {
                    "use_hyde": True,
                    "use_hybrid_search": True,
                    "use_cross_encoder": True,
                    "vector_weight": 0.6,
                    "bm25_weight": 0.4,
                },
            }
        )

        assert kb._use_hyde is True
        assert kb._use_hybrid_search is True
        assert kb._use_cross_encoder is True


class TestRagiSemanticChunking:
    """Tests for Ragi with different chunking strategies."""

    @patch("sentence_transformers.SentenceTransformer")
    @patch("piragi.retrieval.OpenAI")
    @patch("piragi.core.EmbeddingGenerator")
    def test_semantic_chunking_strategy(
        self, mock_embed_gen, mock_openai, mock_semantic_st, temp_dir, sample_docs_dir, mock_embeddings
    ):
        """Test initialization with semantic chunking."""
        mock_embed_gen.return_value = create_mock_embedding_generator(mock_embeddings)
        mock_instance = MagicMock()
        mock_instance.encode.return_value = np.random.rand(10, 384)
        mock_semantic_st.return_value = mock_instance

        kb = Ragi(
            persist_dir=os.path.join(temp_dir, "store"),
            config={
                "chunk": {"strategy": "semantic"},
            }
        )

        from piragi.semantic_chunking import SemanticChunker
        assert isinstance(kb.chunker, SemanticChunker)

    @patch("piragi.retrieval.OpenAI")
    @patch("piragi.core.EmbeddingGenerator")
    def test_hierarchical_chunking_strategy(
        self, mock_embed_gen, mock_openai, temp_dir, mock_embeddings
    ):
        """Test initialization with hierarchical chunking."""
        mock_embed_gen.return_value = create_mock_embedding_generator(mock_embeddings)

        kb = Ragi(
            persist_dir=os.path.join(temp_dir, "store"),
            config={
                "chunk": {
                    "strategy": "hierarchical",
                    "parent_size": 2000,
                    "child_size": 400,
                },
            }
        )

        assert kb._use_hierarchical is True


class TestRagiQueryValidation:
    """Tests for query validation."""

    @patch("piragi.retrieval.OpenAI")
    @patch("piragi.core.EmbeddingGenerator")
    def test_empty_query(self, mock_embed_gen, mock_openai, temp_dir, mock_embeddings):
        """Test handling of empty query."""
        mock_embed_gen.return_value = create_mock_embedding_generator(mock_embeddings)

        kb = Ragi(persist_dir=os.path.join(temp_dir, "store"))
        answer = kb.ask("")

        assert isinstance(answer, Answer)
        assert "valid question" in answer.text.lower()

    @patch("piragi.retrieval.OpenAI")
    @patch("piragi.core.EmbeddingGenerator")
    def test_whitespace_query(self, mock_embed_gen, mock_openai, temp_dir, mock_embeddings):
        """Test handling of whitespace-only query."""
        mock_embed_gen.return_value = create_mock_embedding_generator(mock_embeddings)

        kb = Ragi(persist_dir=os.path.join(temp_dir, "store"))
        answer = kb.ask("   ")

        assert isinstance(answer, Answer)
        assert "valid question" in answer.text.lower()


class TestRagiHybridSearchIntegration:
    """Integration tests for hybrid search."""

    @patch("piragi.retrieval.OpenAI")
    @patch("piragi.core.EmbeddingGenerator")
    def test_add_indexes_for_hybrid(
        self, mock_embed_gen, mock_openai, temp_dir, sample_docs_dir, mock_embeddings
    ):
        """Test that adding documents indexes for hybrid search."""
        mock_embed_gen.return_value = create_mock_embedding_generator(mock_embeddings)

        kb = Ragi(
            persist_dir=os.path.join(temp_dir, "store"),
            config={
                "retrieval": {"use_hybrid_search": True},
            }
        )
        kb.add(sample_docs_dir)

        # Verify hybrid searcher has indexed chunks
        assert kb._hybrid_searcher is not None
        assert len(kb._hybrid_searcher._chunk_texts) > 0

    @patch("piragi.retrieval.OpenAI")
    @patch("piragi.core.EmbeddingGenerator")
    def test_search_uses_hybrid(
        self, mock_embed_gen, mock_openai, temp_dir, sample_docs_dir, mock_embeddings, mock_llm_response
    ):
        """Test that search uses hybrid when enabled."""
        mock_embed_gen.return_value = create_mock_embedding_generator(mock_embeddings)

        # Mock LLM
        mock_client = MagicMock()
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock(message=MagicMock(content=mock_llm_response))]
        mock_client.chat.completions.create.return_value = mock_completion
        mock_openai.return_value = mock_client

        kb = Ragi(
            persist_dir=os.path.join(temp_dir, "store"),
            config={
                "retrieval": {"use_hybrid_search": True},
                "llm": {"enable_query_expansion": False},
            }
        )
        kb.add(sample_docs_dir)

        answer = kb.ask("What is Python?")

        assert isinstance(answer, Answer)
        assert answer.text


class TestRagiCrossEncoderIntegration:
    """Integration tests for cross-encoder reranking."""

    @patch("sentence_transformers.CrossEncoder")
    @patch("piragi.retrieval.OpenAI")
    @patch("piragi.core.EmbeddingGenerator")
    def test_search_uses_cross_encoder(
        self, mock_embed_gen, mock_openai, mock_ce, temp_dir, sample_docs_dir, mock_embeddings, mock_llm_response
    ):
        """Test that search uses cross-encoder when enabled."""
        mock_embed_gen.return_value = create_mock_embedding_generator(mock_embeddings)

        # Mock cross-encoder
        mock_ce_instance = MagicMock()
        mock_ce_instance.predict.return_value = np.array([0.9, 0.7, 0.5, 0.3])
        mock_ce.return_value = mock_ce_instance

        # Mock LLM
        mock_client = MagicMock()
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock(message=MagicMock(content=mock_llm_response))]
        mock_client.chat.completions.create.return_value = mock_completion
        mock_openai.return_value = mock_client

        kb = Ragi(
            persist_dir=os.path.join(temp_dir, "store"),
            config={
                "retrieval": {"use_cross_encoder": True},
                "llm": {"enable_query_expansion": False},
            }
        )
        kb.add(sample_docs_dir)

        answer = kb.ask("What is Python?")

        assert isinstance(answer, Answer)


class TestRagiHyDEIntegration:
    """Integration tests for HyDE."""

    @patch("piragi.query_transform.OpenAI")
    @patch("piragi.retrieval.OpenAI")
    @patch("piragi.core.EmbeddingGenerator")
    def test_search_uses_hyde(
        self, mock_embed_gen, mock_retrieval_openai, mock_hyde_openai, temp_dir, sample_docs_dir, mock_embeddings, mock_llm_response
    ):
        """Test that search uses HyDE when enabled."""
        mock_embed_gen.return_value = create_mock_embedding_generator(mock_embeddings)

        # Mock HyDE LLM response
        mock_hyde_client = MagicMock()
        mock_hyde_response = MagicMock()
        mock_hyde_response.choices = [MagicMock()]
        mock_hyde_response.choices[0].message.content = (
            "Python is a versatile programming language widely used for various applications "
            "including web development, data science, and automation."
        )
        mock_hyde_client.chat.completions.create.return_value = mock_hyde_response
        mock_hyde_openai.return_value = mock_hyde_client

        # Mock retrieval LLM
        mock_ret_client = MagicMock()
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock(message=MagicMock(content=mock_llm_response))]
        mock_ret_client.chat.completions.create.return_value = mock_completion
        mock_retrieval_openai.return_value = mock_ret_client

        kb = Ragi(
            persist_dir=os.path.join(temp_dir, "store"),
            config={
                "retrieval": {"use_hyde": True},
            }
        )
        kb.add(sample_docs_dir)

        answer = kb.ask("What is Python?")

        assert isinstance(answer, Answer)


class TestRagiVectorDimensions:
    """Tests for correct vector dimension handling."""

    @patch("piragi.retrieval.OpenAI")
    @patch("piragi.core.EmbeddingGenerator")
    def test_default_dimension_matches_model(self, mock_embed_gen, mock_openai, temp_dir, mock_embeddings):
        """Test that store dimension matches default embedding model."""
        mock_embed_gen.return_value = create_mock_embedding_generator(mock_embeddings)

        kb = Ragi(persist_dir=os.path.join(temp_dir, "store"))

        # Default model is all-mpnet-base-v2 (768 dim)
        assert kb.store.vector_dimension == 768

    @patch("piragi.retrieval.OpenAI")
    @patch("piragi.core.EmbeddingGenerator")
    def test_custom_model_dimension(self, mock_embed_gen, mock_openai, temp_dir, mock_embeddings):
        """Test that store dimension matches custom embedding model."""
        mock_embed_gen.return_value = create_mock_embedding_generator(mock_embeddings)

        kb = Ragi(
            persist_dir=os.path.join(temp_dir, "store"),
            config={
                "embedding": {"model": "all-MiniLM-L6-v2"},
            }
        )

        # MiniLM is 384 dim
        assert kb.store.vector_dimension == 384

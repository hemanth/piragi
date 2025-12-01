"""Tests for core Ragi class."""

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from piragi import Ragi
from piragi.types import Answer, Citation


@pytest.fixture
def mock_embeddings():
    """Mock embedding responses (768-dim for all-mpnet-base-v2)."""
    return [0.1] * 768


@pytest.fixture
def mock_llm_response():
    """Mock LLM response."""
    return "This is a test answer based on the provided context."


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


class TestRagiInit:
    """Tests for Ragi initialization."""

    def test_init_without_sources(self, temp_dir):
        """Test initialization without sources."""
        persist_dir = os.path.join(temp_dir, "test_ragi")
        kb = Ragi(persist_dir=persist_dir)

        assert kb.store.count() == 0

    @patch("piragi.retrieval.OpenAI")
    def test_init_with_config(self, mock_openai, temp_dir):
        """Test initialization with custom config."""
        persist_dir = os.path.join(temp_dir, "test_ragi")
        kb = Ragi(
            persist_dir=persist_dir,
            config={"llm": {"model": "gpt-4", "api_key": "custom-key"}}
        )

        # Verify OpenAI was initialized
        mock_openai.assert_called()


class TestRagiAdd:
    """Tests for adding documents."""

    @patch("piragi.core.EmbeddingGenerator")
    def test_add_single_file(self, mock_embed_gen, temp_dir, sample_text_file, mock_embeddings):
        """Test adding a single file."""
        mock_embed_gen.return_value = create_mock_embedding_generator(mock_embeddings)

        persist_dir = os.path.join(temp_dir, "test_ragi")
        kb = Ragi(persist_dir=persist_dir)
        kb.add(sample_text_file)

        assert kb.count() > 0

    @patch("piragi.core.EmbeddingGenerator")
    def test_add_multiple_files(
        self, mock_embed_gen, temp_dir, sample_text_file, sample_markdown_file, mock_embeddings
    ):
        """Test adding multiple files."""
        mock_embed_gen.return_value = create_mock_embedding_generator(mock_embeddings)

        persist_dir = os.path.join(temp_dir, "test_ragi")
        kb = Ragi(persist_dir=persist_dir)
        kb.add([sample_text_file, sample_markdown_file])

        assert kb.count() > 0

    @patch("piragi.core.EmbeddingGenerator")
    def test_add_returns_self(self, mock_embed_gen, temp_dir, sample_text_file, mock_embeddings):
        """Test that add() returns self for chaining."""
        mock_embed_gen.return_value = create_mock_embedding_generator(mock_embeddings)

        persist_dir = os.path.join(temp_dir, "test_ragi")
        kb = Ragi(persist_dir=persist_dir)
        result = kb.add(sample_text_file)

        assert result is kb


class TestRagiQuery:
    """Tests for querying."""

    @patch("piragi.retrieval.OpenAI")
    @patch("piragi.core.EmbeddingGenerator")
    def test_ask_question(
        self,
        mock_embed_gen,
        mock_openai,
        temp_dir,
        sample_text_file,
        mock_embeddings,
        mock_llm_response,
    ):
        """Test asking a question."""
        mock_embed_gen.return_value = create_mock_embedding_generator(mock_embeddings)

        # Mock OpenAI response
        mock_client = MagicMock()
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock(message=MagicMock(content=mock_llm_response))]
        mock_client.chat.completions.create.return_value = mock_completion
        mock_openai.return_value = mock_client

        persist_dir = os.path.join(temp_dir, "test_ragi")
        kb = Ragi(persist_dir=persist_dir)
        kb.add(sample_text_file)

        answer = kb.ask("What is this document about?")

        assert isinstance(answer, Answer)
        assert answer.text
        assert answer.query == "What is this document about?"

    @patch("piragi.retrieval.OpenAI")
    @patch("piragi.core.EmbeddingGenerator")
    def test_callable_interface(
        self,
        mock_embed_gen,
        mock_openai,
        temp_dir,
        sample_text_file,
        mock_embeddings,
        mock_llm_response,
    ):
        """Test using Ragi as callable."""
        mock_embed_gen.return_value = create_mock_embedding_generator(mock_embeddings)

        # Mock LLM response
        mock_client = MagicMock()
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock(message=MagicMock(content=mock_llm_response))]
        mock_client.chat.completions.create.return_value = mock_completion
        mock_openai.return_value = mock_client

        persist_dir = os.path.join(temp_dir, "test_ragi")
        kb = Ragi(persist_dir=persist_dir)
        kb.add(sample_text_file)

        answer = kb("What is this?")

        assert isinstance(answer, Answer)


class TestRagiFilter:
    """Tests for metadata filtering."""

    def test_filter_returns_self(self, temp_dir):
        """Test that filter() returns self for chaining."""
        persist_dir = os.path.join(temp_dir, "test_ragi")
        kb = Ragi(persist_dir=persist_dir)
        result = kb.filter(type="test")

        assert result is kb

    @patch("piragi.retrieval.OpenAI")
    @patch("piragi.core.EmbeddingGenerator")
    def test_filter_chaining(
        self,
        mock_embed_gen,
        mock_openai,
        temp_dir,
        sample_text_file,
        mock_embeddings,
        mock_llm_response,
    ):
        """Test filter chaining with ask."""
        mock_embed_gen.return_value = create_mock_embedding_generator(mock_embeddings)

        # Mock LLM response
        mock_client = MagicMock()
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock(message=MagicMock(content=mock_llm_response))]
        mock_client.chat.completions.create.return_value = mock_completion
        mock_openai.return_value = mock_client

        persist_dir = os.path.join(temp_dir, "test_ragi")
        kb = Ragi(persist_dir=persist_dir)
        kb.add(sample_text_file)

        answer = kb.filter(type="test").ask("What is this?")

        assert isinstance(answer, Answer)


class TestRagiUtility:
    """Tests for utility methods."""

    def test_count_empty(self, temp_dir):
        """Test count on empty store."""
        persist_dir = os.path.join(temp_dir, "test_ragi")
        kb = Ragi(persist_dir=persist_dir)

        assert kb.count() == 0

    @patch("piragi.core.EmbeddingGenerator")
    def test_clear(self, mock_embed_gen, temp_dir, sample_text_file, mock_embeddings):
        """Test clearing the knowledge base."""
        mock_embed_gen.return_value = create_mock_embedding_generator(mock_embeddings)

        persist_dir = os.path.join(temp_dir, "test_ragi")
        kb = Ragi(persist_dir=persist_dir)
        kb.add(sample_text_file)

        assert kb.count() > 0

        kb.clear()
        assert kb.count() == 0

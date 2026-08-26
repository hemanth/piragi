"""Tests for core Ragi class."""

import os
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from piragi import Ragi
from piragi.core import FilteredRagi
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

    def embed_chunks_side_effect(chunks, on_progress=None):
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

    @patch("openai.OpenAI")
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

    @patch("openai.OpenAI")
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

    @patch("openai.OpenAI")
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

    def test_filter_returns_filtered_ragi(self, temp_dir):
        """Test that filter() returns FilteredRagi for chaining."""
        persist_dir = os.path.join(temp_dir, "test_ragi")
        kb = Ragi(persist_dir=persist_dir)
        result = kb.filter(type="test")

        assert isinstance(result, FilteredRagi)
        assert result is not kb

    @patch("openai.OpenAI")
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


class TestRagiEmbedderInjection:
    """Tests for custom embedder injection."""

    def test_init_with_custom_embedder(self, temp_dir, mock_embeddings):
        """Test initialization with custom embedder."""
        mock_embedder = create_mock_embedding_generator(mock_embeddings)

        persist_dir = os.path.join(temp_dir, "test_ragi")
        kb = Ragi(persist_dir=persist_dir, embedder=mock_embedder)

        # Verify custom embedder is used
        assert kb.embedder is mock_embedder

    def test_custom_embedder_used_for_add(
        self, temp_dir, sample_text_file, mock_embeddings
    ):
        """Test that custom embedder is used when adding documents."""
        mock_embedder = create_mock_embedding_generator(mock_embeddings)

        persist_dir = os.path.join(temp_dir, "test_ragi")
        kb = Ragi(persist_dir=persist_dir, embedder=mock_embedder)
        kb.add(sample_text_file)

        # Verify embedder's embed_chunks was called
        mock_embedder.embed_chunks.assert_called()
        assert kb.count() > 0

    def test_shared_embedder_across_instances(self, temp_dir, mock_embeddings):
        """Test that same embedder can be shared across multiple Ragi instances."""
        mock_embedder = create_mock_embedding_generator(mock_embeddings)

        persist_dir1 = os.path.join(temp_dir, "test_ragi1")
        persist_dir2 = os.path.join(temp_dir, "test_ragi2")

        kb1 = Ragi(persist_dir=persist_dir1, embedder=mock_embedder)
        kb2 = Ragi(persist_dir=persist_dir2, embedder=mock_embedder)

        # Both instances should share the same embedder
        assert kb1.embedder is kb2.embedder
        assert kb1.embedder is mock_embedder

    def test_custom_embedder_overrides_config(self, temp_dir, mock_embeddings):
        """Test that custom embedder takes precedence over config."""
        mock_embedder = create_mock_embedding_generator(mock_embeddings)
        mock_embedder.model_name = "custom-model"

        persist_dir = os.path.join(temp_dir, "test_ragi")
        kb = Ragi(
            persist_dir=persist_dir,
            embedder=mock_embedder,
            config={"embedding": {"model": "different-model"}}
        )

        # Custom embedder should be used, not the one from config
        assert kb.embedder is mock_embedder

    @patch("openai.OpenAI")
    def test_custom_embedder_used_for_query(
        self,
        mock_openai,
        temp_dir,
        sample_text_file,
        mock_embeddings,
        mock_llm_response,
    ):
        """Test that custom embedder is used for query embedding."""
        mock_embedder = create_mock_embedding_generator(mock_embeddings)

        # Mock LLM response
        mock_client = MagicMock()
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock(message=MagicMock(content=mock_llm_response))]
        mock_client.chat.completions.create.return_value = mock_completion
        mock_openai.return_value = mock_client

        persist_dir = os.path.join(temp_dir, "test_ragi")
        kb = Ragi(persist_dir=persist_dir, embedder=mock_embedder)
        kb.add(sample_text_file)

        kb.ask("What is this?")

        # Verify embed_query was called for the search
        mock_embedder.embed_query.assert_called()


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


class TestRagiRefresh:
    """Tests for refresh() across retrieval modes.

    AutoTokenizer.from_pretrained is patched because Chunker() always loads a
    tokenizer at construction time regardless of embedding config - unrelated
    to what refresh() itself does, but required to construct a Ragi() at all
    in an offline/sandboxed test environment.
    """

    @patch("piragi.chunking.AutoTokenizer")
    @patch("piragi.core.EmbeddingGenerator")
    def test_refresh_dense_mode_deletes_and_reingests(
        self, mock_embed_gen, mock_tokenizer, temp_dir, sample_text_file, mock_embeddings
    ):
        mock_embed_gen.return_value = create_mock_embedding_generator(mock_embeddings)

        persist_dir = os.path.join(temp_dir, "test_ragi")
        kb = Ragi(persist_dir=persist_dir)

        doc = MagicMock()
        doc.source = sample_text_file
        kb.loader = MagicMock()
        kb.loader.load.return_value = [doc]
        kb.store = MagicMock()

        with patch("piragi.pipelines.ingestion.IngestionPipeline") as MockPipeline:
            pipeline_instance = MockPipeline.return_value
            result = kb.refresh(sample_text_file)

        kb.loader.load.assert_called_once_with(sample_text_file)
        kb.store.delete_by_source.assert_called_once_with(sample_text_file)
        MockPipeline.assert_called_once_with(
            loader=kb.loader,
            chunker=kb.chunker,
            embedder=kb.embedder,
            store=kb.store,
            post_chunk_hook=kb._post_chunk_hook,
            post_embed_hook=kb._post_embed_hook,
            use_hierarchical=kb._use_hierarchical,
            mode=kb._mode,
            bm25_index=kb._bm25_index,
            on_the_fly_index=kb._on_the_fly_index,
        )
        pipeline_instance.ingest.assert_called_once_with(sample_text_file)
        assert result is kb

    @patch("piragi.chunking.AutoTokenizer")
    @patch("piragi.core.EmbeddingGenerator")
    def test_refresh_bm25_only_mode_deletes_from_bm25_index_not_store(
        self, mock_embed_gen, mock_tokenizer, temp_dir, sample_text_file, mock_embeddings
    ):
        mock_embed_gen.return_value = create_mock_embedding_generator(mock_embeddings)

        persist_dir = os.path.join(temp_dir, "test_ragi")
        kb = Ragi(persist_dir=persist_dir, config={"retrieval": {"mode": "bm25_only"}})
        assert kb.store is None  # bm25_only mode has no vector store

        doc = MagicMock()
        doc.source = sample_text_file
        kb.loader = MagicMock()
        kb.loader.load.return_value = [doc]
        kb._bm25_index = MagicMock()

        with patch("piragi.pipelines.ingestion.IngestionPipeline") as MockPipeline:
            pipeline_instance = MockPipeline.return_value
            kb.refresh(sample_text_file)

        kb._bm25_index.delete_by_source.assert_called_once_with(sample_text_file)
        pipeline_instance.ingest.assert_called_once_with(sample_text_file)

    @patch("piragi.chunking.AutoTokenizer")
    @patch("piragi.core.EmbeddingGenerator")
    def test_refresh_deletes_per_source_for_multiple_documents(
        self, mock_embed_gen, mock_tokenizer, temp_dir, sample_text_file, mock_embeddings
    ):
        mock_embed_gen.return_value = create_mock_embedding_generator(mock_embeddings)

        persist_dir = os.path.join(temp_dir, "test_ragi")
        kb = Ragi(persist_dir=persist_dir)

        doc1, doc2 = MagicMock(), MagicMock()
        doc1.source, doc2.source = "a.txt", "b.txt"
        kb.loader = MagicMock()
        kb.loader.load.return_value = [doc1, doc2]
        kb.store = MagicMock()

        with patch("piragi.pipelines.ingestion.IngestionPipeline"):
            kb.refresh(["a.txt", "b.txt"])

        assert kb.store.delete_by_source.call_args_list == [call("a.txt"), call("b.txt")]

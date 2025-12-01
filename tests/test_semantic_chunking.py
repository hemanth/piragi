"""Tests for semantic chunking strategies."""

import pytest
from unittest.mock import MagicMock, patch
import numpy as np

from piragi.semantic_chunking import (
    SemanticChunker,
    ContextualChunker,
    PropositionChunker,
    HierarchicalChunker,
)
from piragi.types import Document


@pytest.fixture
def sample_document():
    """Create a sample document for testing."""
    return Document(
        content="""# Introduction

This is the introduction section. It provides an overview of the topic.
The introduction explains the main concepts.

# Methods

This section describes the methodology used. We employed various techniques
to analyze the data. The analysis involved statistical methods.

# Results

The results show significant improvements. We observed a 50% increase in
performance. The data clearly demonstrates the effectiveness of our approach.

# Conclusion

In conclusion, our approach works well. Future work should explore additional
applications and optimizations.""",
        source="test_doc.md",
        metadata={"type": "research"},
    )


@pytest.fixture
def short_document():
    """Create a short document for testing."""
    return Document(
        content="This is a short document with just one sentence.",
        source="short.txt",
        metadata={},
    )


class TestSemanticChunker:
    """Tests for semantic-based chunking."""

    def test_init_defaults(self):
        """Test initialization with defaults."""
        chunker = SemanticChunker()
        assert chunker.similarity_threshold == 0.5
        assert chunker.min_chunk_size == 100
        assert chunker.max_chunk_size == 2000

    def test_init_custom(self):
        """Test initialization with custom parameters."""
        chunker = SemanticChunker(
            similarity_threshold=0.7,
            min_chunk_size=50,
            max_chunk_size=1000,
        )
        assert chunker.similarity_threshold == 0.7
        assert chunker.min_chunk_size == 50
        assert chunker.max_chunk_size == 1000

    def test_split_sentences(self):
        """Test sentence splitting."""
        chunker = SemanticChunker()
        text = "First sentence. Second sentence! Third sentence?"
        sentences = chunker._split_sentences(text)

        assert len(sentences) >= 2

    def test_split_sentences_with_paragraphs(self):
        """Test sentence splitting respects paragraphs."""
        chunker = SemanticChunker()
        text = "Paragraph one.\n\nParagraph two."
        sentences = chunker._split_sentences(text)

        assert len(sentences) == 2

    def test_compute_similarities(self):
        """Test computing similarities between sentences."""
        mock_model = MagicMock()
        # Return 3 embeddings of dimension 4
        mock_model.encode.return_value = np.array([
            [1, 0, 0, 0],
            [0.9, 0.1, 0, 0],  # Similar to first
            [0, 0, 1, 0],      # Different
        ])

        chunker = SemanticChunker()
        # Inject mock model directly to bypass lazy loading
        chunker._model = mock_model

        similarities = chunker._compute_similarities([
            "Sentence 1", "Sentence 2", "Sentence 3"
        ])

        assert len(similarities) == 2  # n-1 similarities
        assert similarities[0] > similarities[1]  # First pair more similar

    def test_chunk_document(self, sample_document):
        """Test chunking a full document."""
        mock_model = MagicMock()
        # Return embeddings that will create natural splits
        mock_model.encode.return_value = np.random.rand(20, 384)

        chunker = SemanticChunker(
            min_chunk_size=50,
            max_chunk_size=500,
            similarity_threshold=0.5,
        )
        # Inject mock model directly to bypass lazy loading
        chunker._model = mock_model

        chunks = chunker.chunk_document(sample_document)

        assert len(chunks) >= 1
        assert all(chunk.source == "test_doc.md" for chunk in chunks)
        assert all(chunk.metadata.get("type") == "research" for chunk in chunks)

    def test_chunk_short_document(self, short_document):
        """Test chunking a very short document."""
        chunker = SemanticChunker(min_chunk_size=10)

        # Inject mock model directly to bypass lazy loading
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.rand(1, 384)
        chunker._model = mock_model

        chunks = chunker.chunk_document(short_document)

        assert len(chunks) == 1

    def test_chunk_empty_document(self):
        """Test chunking an empty document."""
        chunker = SemanticChunker()
        empty_doc = Document(content="", source="empty.txt", metadata={})

        chunks = chunker.chunk_document(empty_doc)
        assert chunks == []


class TestContextualChunker:
    """Tests for contextual chunking with LLM-generated context."""

    def test_init(self):
        """Test initialization."""
        with patch("openai.OpenAI"):
            chunker = ContextualChunker()
            assert chunker.model == "llama3.2"

    def test_init_custom_template(self):
        """Test initialization with custom template."""
        with patch("openai.OpenAI"):
            template = "Context: {context}\nContent: {chunk}"
            chunker = ContextualChunker(context_template=template)
            assert chunker.context_template == template

    @patch("openai.OpenAI")
    def test_generate_context(self, mock_openai, sample_document):
        """Test context generation for a chunk."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = (
            "This chunk is from the Introduction section discussing main concepts."
        )
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        from piragi.types import Chunk
        chunk = Chunk(
            text="This is the introduction section.",
            source="test.md",
            chunk_index=0,
            metadata={},
        )

        chunker = ContextualChunker()
        context = chunker._generate_context(sample_document, chunk)

        assert len(context) > 0
        assert "Introduction" in context or "section" in context

    @patch("openai.OpenAI")
    def test_generate_context_fallback(self, mock_openai, sample_document):
        """Test context generation fallback on error."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API error")
        mock_openai.return_value = mock_client

        from piragi.types import Chunk
        chunk = Chunk(
            text="Test chunk",
            source="test.md",
            chunk_index=0,
            metadata={},
        )

        chunker = ContextualChunker()
        context = chunker._generate_context(sample_document, chunk)

        # Should fallback to source-based context
        assert "test.md" in context or "From" in context

    @patch("openai.OpenAI")
    @patch("piragi.chunking.Chunker")
    def test_chunk_document(self, mock_base_chunker, mock_openai, sample_document):
        """Test full contextual chunking."""
        # Mock base chunker
        from piragi.types import Chunk
        mock_chunker_instance = MagicMock()
        mock_chunker_instance.chunk_document.return_value = [
            Chunk(text="Chunk 1", source="test.md", chunk_index=0, metadata={}),
            Chunk(text="Chunk 2", source="test.md", chunk_index=1, metadata={}),
        ]
        mock_base_chunker.return_value = mock_chunker_instance

        # Mock LLM
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Context for chunk"
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        chunker = ContextualChunker()
        chunks = chunker.chunk_document(sample_document)

        assert len(chunks) == 2
        # Each chunk should have context prepended
        for chunk in chunks:
            assert "document_context" in chunk.text or "Context" in chunk.text
            assert "original_text" in chunk.metadata
            assert "context" in chunk.metadata


class TestPropositionChunker:
    """Tests for proposition-based chunking."""

    def test_init(self):
        """Test initialization."""
        with patch("openai.OpenAI"):
            chunker = PropositionChunker()
            assert chunker.model == "llama3.2"
            assert chunker.max_propositions_per_call == 20

    @patch("openai.OpenAI")
    def test_extract_propositions(self, mock_openai):
        """Test extracting propositions from text."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = (
            "Python is a programming language.\n"
            "Python was created by Guido van Rossum.\n"
            "Python is widely used for data science."
        )
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        chunker = PropositionChunker()
        text = "Python is a popular language created by Guido, used for data science."
        propositions = chunker._extract_propositions(text)

        assert len(propositions) == 3
        assert all("Python" in p for p in propositions)

    @patch("openai.OpenAI")
    def test_extract_propositions_fallback(self, mock_openai):
        """Test fallback on extraction error."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API error")
        mock_openai.return_value = mock_client

        chunker = PropositionChunker()
        text = "Original text"
        propositions = chunker._extract_propositions(text)

        assert propositions == [text]

    @patch("openai.OpenAI")
    def test_chunk_document(self, mock_openai, sample_document):
        """Test proposition-based chunking of a document."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = (
            "This is proposition one about the topic.\n"
            "This is proposition two with more details."
        )
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        chunker = PropositionChunker()
        chunks = chunker.chunk_document(sample_document)

        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.metadata.get("chunk_type") == "proposition"


class TestHierarchicalChunker:
    """Tests for hierarchical parent-child chunking."""

    def test_init_defaults(self):
        """Test initialization with defaults."""
        chunker = HierarchicalChunker()
        assert chunker.parent_chunk_size == 2000
        assert chunker.child_chunk_size == 400
        assert chunker.overlap == 50

    def test_init_custom(self):
        """Test initialization with custom sizes."""
        chunker = HierarchicalChunker(
            parent_chunk_size=3000,
            child_chunk_size=500,
            overlap=100,
        )
        assert chunker.parent_chunk_size == 3000
        assert chunker.child_chunk_size == 500
        assert chunker.overlap == 100

    def test_create_parent_chunks(self, sample_document):
        """Test creating parent-level chunks."""
        chunker = HierarchicalChunker(parent_chunk_size=200)
        parents = chunker._create_parent_chunks(
            sample_document.content,
            sample_document.source,
        )

        assert len(parents) >= 1
        # Each parent should be a (text, index) tuple
        for text, idx in parents:
            assert len(text) > 0
            assert isinstance(idx, int)

    def test_create_child_chunks(self, sample_document):
        """Test creating child chunks from parent."""
        chunker = HierarchicalChunker(child_chunk_size=100, overlap=20)
        parent_text = sample_document.content[:500]

        children = chunker._create_child_chunks(
            parent_text,
            parent_idx=0,
            source="test.md",
            metadata={"type": "test"},
        )

        assert len(children) >= 1
        for child in children:
            assert child.metadata.get("parent_index") == 0
            assert child.metadata.get("parent_text") == parent_text
            assert child.metadata.get("chunk_type") == "child"

    def test_chunk_document(self, sample_document):
        """Test full hierarchical chunking."""
        chunker = HierarchicalChunker(
            parent_chunk_size=300,
            child_chunk_size=100,
        )
        parent_chunks, child_chunks = chunker.chunk_document(sample_document)

        assert len(parent_chunks) >= 1
        assert len(child_chunks) >= len(parent_chunks)

        # Parents should have chunk_type "parent"
        for parent in parent_chunks:
            assert parent.metadata.get("chunk_type") == "parent"

        # Children should reference parents
        for child in child_chunks:
            assert child.metadata.get("chunk_type") == "child"
            assert "parent_index" in child.metadata
            assert "parent_text" in child.metadata

    def test_chunk_document_preserves_metadata(self, sample_document):
        """Test that original metadata is preserved."""
        chunker = HierarchicalChunker()
        parent_chunks, child_chunks = chunker.chunk_document(sample_document)

        for parent in parent_chunks:
            assert parent.metadata.get("type") == "research"

        for child in child_chunks:
            assert child.metadata.get("type") == "research"

    def test_chunk_short_document(self, short_document):
        """Test hierarchical chunking of short document."""
        chunker = HierarchicalChunker(
            parent_chunk_size=1000,
            child_chunk_size=100,
        )
        parent_chunks, child_chunks = chunker.chunk_document(short_document)

        # Short doc should produce 1 parent with 1 child
        assert len(parent_chunks) == 1
        assert len(child_chunks) >= 1

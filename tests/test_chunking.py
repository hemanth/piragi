"""Tests for document chunking."""

import pytest

from piragi.chunking import Chunker
from piragi.types import Document


def test_chunk_small_document():
    """Test chunking a small document that fits in one chunk."""
    chunker = Chunker(chunk_size=512)
    doc = Document(
        content="This is a small document.",
        source="test.txt",
        metadata={"type": "test"},
    )

    chunks = chunker.chunk_document(doc)

    assert len(chunks) == 1
    assert chunks[0].text == "This is a small document."
    assert chunks[0].source == "test.txt"
    assert chunks[0].chunk_index == 0
    assert chunks[0].metadata["type"] == "test"


def test_chunk_document_with_headers():
    """Test chunking respects markdown headers."""
    chunker = Chunker(chunk_size=100)
    doc = Document(
        content="""# Header 1
Some content here.

## Header 2
More content here.

### Header 3
Even more content.""",
        source="test.md",
        metadata={},
    )

    chunks = chunker.chunk_document(doc)

    # Should split by headers
    assert len(chunks) >= 3
    assert any("Header 1" in chunk.text for chunk in chunks)
    assert any("Header 2" in chunk.text for chunk in chunks)


def test_chunk_with_overlap():
    """Test that chunks have proper overlap."""
    chunker = Chunker(chunk_size=50, chunk_overlap=10)

    # Create a long document
    doc = Document(
        content=" ".join([f"word{i}" for i in range(200)]),
        source="long.txt",
        metadata={},
    )

    chunks = chunker.chunk_document(doc)

    # Should have multiple chunks due to length
    assert len(chunks) > 1

    # All chunks should have source and increasing indices
    for i, chunk in enumerate(chunks):
        assert chunk.chunk_index == i
        assert chunk.source == "long.txt"


def test_metadata_propagation():
    """Test that document metadata is propagated to chunks."""
    chunker = Chunker(chunk_size=512)
    doc = Document(
        content="Test content.",
        source="test.txt",
        metadata={"author": "Test Author", "category": "test"},
    )

    chunks = chunker.chunk_document(doc)

    assert len(chunks) == 1
    assert chunks[0].metadata["author"] == "Test Author"
    assert chunks[0].metadata["category"] == "test"

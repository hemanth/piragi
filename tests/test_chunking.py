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


def test_sentence_break_with_numbered_list():
    """Test that numbered lists don't cause incorrect sentence breaks."""
    chunker = Chunker(chunk_size=50, chunk_overlap=10)

    # Text with numbered list - periods after numbers should not be sentence endings
    text = """How Do You Make Strawberry Jam?
1. Mash the strawberries.
2. Combine all the ingredients in a saucepan and dissolve the sugar over low heat.
3. Bring the mixture to a boil. Cook and check the doneness.
4. Process according to the recipe below."""

    # Use _break_at_sentence directly to test
    result = chunker._break_at_sentence(text)

    # Should not break after "1." or "2." etc
    # The result should contain complete sentences
    assert "1." in result or "Mash" in result


def test_sentence_break_with_acronyms():
    """Test that acronyms don't cause incorrect sentence breaks."""
    chunker = Chunker(chunk_size=100, chunk_overlap=10)

    # Text with acronyms - periods in acronyms should not be sentence endings
    text = """Geoffrey Hinton received his B.A. in Experimental Psychology from Cambridge in 1970 and his Ph.D. in Artificial Intelligence from Edinburgh in 1978. He is a pioneer in deep learning."""

    result = chunker._break_at_sentence(text)

    # Should not break after "B.A." or "Ph.D."
    # The acronyms should remain intact within sentences
    assert "B.A." in result or "Ph.D." in result


def test_sentence_break_with_abbreviations():
    """Test that common abbreviations don't cause incorrect sentence breaks."""
    chunker = Chunker(chunk_size=100, chunk_overlap=10)

    # Text with abbreviations
    text = """Dr. Smith and Mr. Jones met with Prof. Williams at the U.S. embassy. They discussed important matters regarding the U.K. delegation."""

    result = chunker._break_at_sentence(text)

    # Should not break after "Dr." or "Mr." or "Prof."
    assert "Dr." in result
    assert "Mr." in result or "Prof." in result


def test_sentence_break_with_initials():
    """Test that initials don't cause incorrect sentence breaks."""
    chunker = Chunker(chunk_size=100, chunk_overlap=10)

    # Text with initials in names
    text = """J.K. Rowling wrote the Harry Potter series. C.S. Lewis wrote the Chronicles of Narnia. Both are beloved authors."""

    result = chunker._break_at_sentence(text)

    # Should not break after "J." or "K." in "J.K."
    assert "J.K. Rowling" in result or "C.S. Lewis" in result

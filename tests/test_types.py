"""Tests for type definitions."""

from piragi.types import Answer, Citation, Chunk, Document


def test_citation_creation():
    """Test creating a citation."""
    citation = Citation(
        source="test.txt",
        chunk="This is a test chunk.",
        score=0.95,
        metadata={"type": "test"},
    )

    assert citation.source == "test.txt"
    assert citation.chunk == "This is a test chunk."
    assert citation.score == 0.95
    assert citation.metadata["type"] == "test"


def test_citation_preview():
    """Test citation preview for long chunks."""
    long_text = "a" * 150
    citation = Citation(source="test.txt", chunk=long_text, score=0.9)

    preview = citation.preview
    assert len(preview) <= 103  # 100 chars + "..."
    assert preview.endswith("...")


def test_citation_preview_short():
    """Test citation preview for short chunks."""
    short_text = "Short text"
    citation = Citation(source="test.txt", chunk=short_text, score=0.9)

    preview = citation.preview
    assert preview == short_text
    assert not preview.endswith("...")


def test_answer_creation():
    """Test creating an answer."""
    citations = [
        Citation(source="test1.txt", chunk="Chunk 1", score=0.9),
        Citation(source="test2.txt", chunk="Chunk 2", score=0.8),
    ]

    answer = Answer(
        text="This is the answer.",
        citations=citations,
        query="What is the question?",
    )

    assert answer.text == "This is the answer."
    assert len(answer.citations) == 2
    assert answer.query == "What is the question?"


def test_answer_str():
    """Test Answer string representation."""
    answer = Answer(text="Test answer", citations=[], query="Test query")
    assert str(answer) == "Test answer"


def test_answer_repr():
    """Test Answer repr."""
    citations = [Citation(source="test.txt", chunk="Test", score=0.9)]
    answer = Answer(text="Long answer text here", citations=citations, query="Test")

    repr_str = repr(answer)
    assert "Answer" in repr_str
    assert "citations=1" in repr_str


def test_document_creation():
    """Test creating a document."""
    doc = Document(
        content="# Test\nContent here.",
        source="test.md",
        metadata={"type": "markdown"},
    )

    assert doc.content == "# Test\nContent here."
    assert doc.source == "test.md"
    assert doc.metadata["type"] == "markdown"


def test_chunk_creation():
    """Test creating a chunk."""
    chunk = Chunk(
        text="Chunk text",
        source="test.txt",
        chunk_index=0,
        metadata={"key": "value"},
    )

    assert chunk.text == "Chunk text"
    assert chunk.source == "test.txt"
    assert chunk.chunk_index == 0
    assert chunk.embedding is None


def test_chunk_with_embedding():
    """Test chunk with embedding."""
    embedding = [0.1] * 1536
    chunk = Chunk(
        text="Test",
        source="test.txt",
        chunk_index=0,
        embedding=embedding,
    )

    assert chunk.embedding == embedding
    assert len(chunk.embedding) == 1536

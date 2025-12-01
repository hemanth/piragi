"""Tests for document loader."""

import os

import pytest

from piragi.loader import DocumentLoader


def test_load_single_file(sample_text_file):
    """Test loading a single file."""
    loader = DocumentLoader()
    documents = loader.load(sample_text_file)

    assert len(documents) == 1
    assert documents[0].source == sample_text_file
    assert "Sample Document" in documents[0].content
    assert documents[0].metadata["filename"] == "sample.txt"


def test_load_multiple_files(sample_text_file, sample_markdown_file):
    """Test loading multiple files."""
    loader = DocumentLoader()
    documents = loader.load([sample_text_file, sample_markdown_file])

    assert len(documents) == 2
    sources = {doc.source for doc in documents}
    assert sample_text_file in sources
    assert sample_markdown_file in sources


def test_load_glob_pattern(temp_dir, sample_text_file, sample_markdown_file):
    """Test loading files with glob pattern."""
    loader = DocumentLoader()
    pattern = os.path.join(temp_dir, "*.md")
    documents = loader.load(pattern)

    assert len(documents) >= 1
    assert any("README" in doc.metadata.get("filename", "") for doc in documents)


def test_load_directory(temp_dir, sample_text_file, sample_markdown_file, sample_code_file):
    """Test loading entire directory."""
    loader = DocumentLoader()
    documents = loader.load(temp_dir)

    assert len(documents) >= 3


def test_load_invalid_source():
    """Test loading from invalid source."""
    loader = DocumentLoader()

    with pytest.raises(ValueError, match="Invalid source"):
        loader.load("/nonexistent/file.txt")


def test_metadata_extraction(sample_text_file):
    """Test metadata extraction."""
    loader = DocumentLoader()
    documents = loader.load(sample_text_file)
    doc = documents[0]

    assert "filename" in doc.metadata
    assert "file_type" in doc.metadata
    assert "file_path" in doc.metadata
    assert doc.metadata["file_type"] == "txt"

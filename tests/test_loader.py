"""Tests for document loader."""

import os
from unittest.mock import MagicMock, patch

import pytest

from piragi.loader import DocumentLoader, REMOTE_SCHEMES


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


# Remote filesystem tests


def test_is_remote_uri():
    """Test detection of remote filesystem URIs."""
    loader = DocumentLoader()

    # Remote URIs should be detected
    assert loader._is_remote_uri("s3://bucket/path/file.pdf")
    assert loader._is_remote_uri("gs://bucket/docs/*.md")
    assert loader._is_remote_uri("az://container/file.txt")
    assert loader._is_remote_uri("gcs://bucket/path")
    assert loader._is_remote_uri("abfs://container/path")

    # These should NOT be detected as remote URIs
    assert not loader._is_remote_uri("./local/file.txt")
    assert not loader._is_remote_uri("/absolute/path/file.md")
    assert not loader._is_remote_uri("https://example.com/file.pdf")
    assert not loader._is_remote_uri("http://example.com/doc.md")


def test_is_url():
    """Test detection of HTTP/HTTPS URLs."""
    loader = DocumentLoader()

    # HTTP URLs should be detected
    assert loader._is_url("https://example.com/file.pdf")
    assert loader._is_url("http://example.com/doc.md")

    # These should NOT be URLs
    assert not loader._is_url("s3://bucket/path/file.pdf")
    assert not loader._is_url("./local/file.txt")
    assert not loader._is_url("/absolute/path/file.md")


def test_remote_schemes_defined():
    """Test that expected remote schemes are defined."""
    assert "s3" in REMOTE_SCHEMES
    assert "gs" in REMOTE_SCHEMES
    assert "gcs" in REMOTE_SCHEMES
    assert "az" in REMOTE_SCHEMES
    assert "abfs" in REMOTE_SCHEMES


@patch("piragi.loader.fsspec")
def test_load_remote_single_file(mock_fsspec, tmp_path):
    """Test loading a single file from remote filesystem."""
    # Create a mock filesystem
    mock_fs = MagicMock()
    mock_fsspec.filesystem.return_value = mock_fs

    # Mock filesystem methods
    mock_fs.isdir.return_value = False
    mock_fs.glob.return_value = []

    # Create a temp file to simulate downloaded content
    test_file = tmp_path / "test.txt"
    test_file.write_text("Remote file content")

    def mock_get(remote_path, local_path):
        # Copy test content to the temp file location
        with open(local_path, "w") as f:
            f.write("Remote file content")

    mock_fs.get.side_effect = mock_get

    loader = DocumentLoader()
    documents = loader._load_remote("s3://bucket/docs/test.txt")

    assert len(documents) == 1
    assert documents[0].source == "s3://bucket/docs/test.txt"
    assert documents[0].metadata["remote_scheme"] == "s3"
    assert documents[0].metadata["filename"] == "test.txt"
    mock_fsspec.filesystem.assert_called_with("s3")


@patch("piragi.loader.fsspec")
def test_load_remote_glob_pattern(mock_fsspec, tmp_path):
    """Test loading files with glob pattern from remote filesystem."""
    mock_fs = MagicMock()
    mock_fsspec.filesystem.return_value = mock_fs

    # Mock glob returning multiple files
    mock_fs.glob.return_value = ["bucket/docs/file1.txt", "bucket/docs/file2.txt"]
    mock_fs.isdir.return_value = False

    def mock_get(remote_path, local_path):
        with open(local_path, "w") as f:
            f.write(f"Content of {remote_path}")

    mock_fs.get.side_effect = mock_get

    loader = DocumentLoader()
    documents = loader._load_remote("s3://bucket/docs/*.txt")

    assert len(documents) == 2
    assert mock_fs.glob.called
    mock_fs.glob.assert_called_with("bucket/docs/*.txt")


@patch("piragi.loader.fsspec")
def test_load_remote_missing_dependency(mock_fsspec):
    """Test helpful error message when remote FS dependency is missing."""
    mock_fsspec.filesystem.side_effect = ImportError("No module named 's3fs'")

    loader = DocumentLoader()

    with pytest.raises(ImportError, match="pip install piragi"):
        loader._load_remote("s3://bucket/file.txt")


@patch("piragi.loader.fsspec")
def test_load_remote_no_files_found(mock_fsspec):
    """Test error when no files match remote glob pattern."""
    mock_fs = MagicMock()
    mock_fsspec.filesystem.return_value = mock_fs
    mock_fs.glob.return_value = []

    loader = DocumentLoader()

    with pytest.raises(ValueError, match="No files found"):
        loader._load_remote("s3://bucket/nonexistent/*.pdf")


# Crawl URL tests


def test_is_crawl_url():
    """Test detection of crawl URLs."""
    loader = DocumentLoader()

    # Crawl URLs should be detected
    assert loader._is_crawl_url("https://example.com/**")
    assert loader._is_crawl_url("https://docs.example.com/api/**")
    assert loader._is_crawl_url("http://localhost:8000/**")

    # These should NOT be crawl URLs
    assert not loader._is_crawl_url("https://example.com")
    assert not loader._is_crawl_url("https://example.com/")
    assert not loader._is_crawl_url("https://example.com/*.html")
    assert not loader._is_crawl_url("s3://bucket/**")
    assert not loader._is_crawl_url("./docs/**")


@patch("piragi.loader._get_crawl4ai")
def test_crawl_url_missing_dependency(mock_get_crawl4ai):
    """Test helpful error message when crawl4ai is not installed."""
    mock_get_crawl4ai.side_effect = ImportError(
        "crawl4ai is required for recursive URL crawling. "
        "Install it with: pip install piragi[crawler] or pip install crawl4ai"
    )

    loader = DocumentLoader()

    with pytest.raises(ImportError, match="pip install piragi"):
        loader._crawl_url("https://example.com/**")

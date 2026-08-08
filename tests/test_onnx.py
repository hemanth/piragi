import pytest
from unittest.mock import patch, MagicMock

from piragi.embeddings import EmbeddingGenerator


def test_backend_parameter_stored():
    """Test that the backend parameter is stored properly."""
    generator = EmbeddingGenerator(backend="onnx", base_url="http://mock")
    assert generator.backend == "onnx"


def test_backend_default_none():
    """Test that the backend parameter defaults to None."""
    generator = EmbeddingGenerator(base_url="http://mock")
    assert generator.backend is None


@patch("sentence_transformers.SentenceTransformer")
def test_backend_passed_to_sentence_transformer(mock_st):
    """Test that backend='onnx' is passed to SentenceTransformer when set."""
    generator = EmbeddingGenerator(backend="onnx")
    mock_st.assert_called_once()
    assert "backend" in mock_st.call_args.kwargs
    assert mock_st.call_args.kwargs["backend"] == "onnx"


@patch("sentence_transformers.SentenceTransformer")
def test_backend_none_not_passed(mock_st):
    """Test that backend=None does not pass backend kwarg to SentenceTransformer."""
    generator = EmbeddingGenerator(backend=None)
    mock_st.assert_called_once()
    assert "backend" not in mock_st.call_args.kwargs


def test_remote_mode_ignores_backend():
    """Test that remote mode stores backend but doesn't use SentenceTransformer."""
    generator = EmbeddingGenerator(base_url="http://mock", api_key="test", backend="onnx")
    assert generator.backend == "onnx"
    assert generator.model is None  # SentenceTransformer not used

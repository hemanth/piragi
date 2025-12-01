"""Tests for query transformation techniques (HyDE, expansion, etc.)."""

import pytest
from unittest.mock import MagicMock, patch

from piragi.query_transform import (
    HyDE,
    QueryExpander,
    MultiQueryRetriever,
    StepBackPrompting,
)


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Mock response"
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


class TestHyDE:
    """Tests for Hypothetical Document Embeddings."""

    def test_init_defaults(self):
        """Test HyDE initialization with defaults."""
        with patch("piragi.query_transform.OpenAI"):
            hyde = HyDE()
            assert hyde.model == "llama3.2"
            assert hyde.num_hypothetical == 1
            assert hyde.temperature == 0.7

    def test_init_custom(self):
        """Test HyDE with custom configuration."""
        with patch("piragi.query_transform.OpenAI"):
            hyde = HyDE(
                model="gpt-4",
                num_hypothetical=3,
                temperature=0.5,
                max_tokens=512,
            )
            assert hyde.model == "gpt-4"
            assert hyde.num_hypothetical == 3
            assert hyde.temperature == 0.5
            assert hyde.max_tokens == 512

    @patch("piragi.query_transform.OpenAI")
    def test_generate_hypothetical_document(self, mock_openai):
        """Test generating a hypothetical document."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = (
            "Python is a versatile programming language widely used for web development, "
            "data science, and automation. It features clean syntax and extensive libraries."
        )
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        hyde = HyDE()
        result = hyde.generate_hypothetical_document("What is Python?")

        assert len(result) > 0
        assert "Python" in result
        mock_client.chat.completions.create.assert_called_once()

    @patch("piragi.query_transform.OpenAI")
    def test_generate_hypothetical_fallback_on_error(self, mock_openai):
        """Test fallback to original query on error."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API error")
        mock_openai.return_value = mock_client

        hyde = HyDE()
        query = "What is Python?"
        result = hyde.generate_hypothetical_document(query)

        # Should return original query on failure
        assert result == query

    @patch("piragi.query_transform.OpenAI")
    def test_generate_multiple(self, mock_openai):
        """Test generating multiple hypothetical documents."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hypothetical content"
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        hyde = HyDE(num_hypothetical=3)
        results = hyde.generate_multiple("What is Python?")

        assert len(results) == 3
        assert mock_client.chat.completions.create.call_count == 3

    @patch("piragi.query_transform.OpenAI")
    def test_transform_query(self, mock_openai):
        """Test transform_query method."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Transformed hypothetical"
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        hyde = HyDE()
        result = hyde.transform_query("Original query")

        assert result == "Transformed hypothetical"


class TestQueryExpander:
    """Tests for query expansion."""

    def test_init_defaults(self):
        """Test QueryExpander initialization."""
        with patch("piragi.query_transform.OpenAI"):
            expander = QueryExpander()
            assert expander.model == "llama3.2"
            assert expander.num_expansions == 2

    @patch("piragi.query_transform.OpenAI")
    def test_expand_returns_original_plus_variations(self, mock_openai):
        """Test that expand returns original query plus variations."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = (
            "How does Python work?\nWhat are Python's main features?"
        )
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        expander = QueryExpander()
        results = expander.expand("What is Python?")

        assert len(results) >= 1
        assert "What is Python?" in results  # Original should be first

    @patch("piragi.query_transform.OpenAI")
    def test_expand_fallback_on_error(self, mock_openai):
        """Test fallback to original query on error."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API error")
        mock_openai.return_value = mock_client

        expander = QueryExpander()
        query = "What is Python?"
        results = expander.expand(query)

        assert results == [query]

    @patch("piragi.query_transform.OpenAI")
    def test_expand_filters_short_lines(self, mock_openai):
        """Test that very short variations are filtered out."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "ok\n\nWhat about Python?\nhi"
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        expander = QueryExpander()
        results = expander.expand("What is Python?")

        # Short lines ("ok", "hi") should be filtered
        for result in results:
            assert len(result) > 5 or result == "What is Python?"

    @patch("piragi.query_transform.OpenAI")
    def test_expand_removes_numbering(self, mock_openai):
        """Test that numbered lists are cleaned up."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = (
            "1. How does Python work?\n2) What are Python features?"
        )
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        expander = QueryExpander()
        results = expander.expand("What is Python?")

        # Numbering should be removed
        for result in results[1:]:  # Skip original
            assert not result.startswith("1.") and not result.startswith("2)")


class TestMultiQueryRetriever:
    """Tests for multi-query retriever combining HyDE and expansion."""

    def test_init_both_enabled(self):
        """Test initialization with both methods enabled."""
        with patch("piragi.query_transform.OpenAI"):
            retriever = MultiQueryRetriever(use_hyde=True, use_expansion=True)
            assert retriever.hyde is not None
            assert retriever.expander is not None

    def test_init_hyde_only(self):
        """Test initialization with HyDE only."""
        with patch("piragi.query_transform.OpenAI"):
            retriever = MultiQueryRetriever(use_hyde=True, use_expansion=False)
            assert retriever.hyde is not None
            assert retriever.expander is None

    def test_init_expansion_only(self):
        """Test initialization with expansion only."""
        with patch("piragi.query_transform.OpenAI"):
            retriever = MultiQueryRetriever(use_hyde=False, use_expansion=True)
            assert retriever.hyde is None
            assert retriever.expander is not None

    @patch("piragi.query_transform.OpenAI")
    def test_get_queries_with_both(self, mock_openai):
        """Test getting queries with both methods enabled."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hypothetical or expanded"
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        retriever = MultiQueryRetriever(use_hyde=True, use_expansion=True)
        results = retriever.get_queries("What is Python?")

        # Should have original + hyde + expansions
        assert len(results) >= 2
        assert "What is Python?" in results

    @patch("piragi.query_transform.OpenAI")
    def test_get_queries_handles_hyde_failure(self, mock_openai):
        """Test graceful handling of HyDE failure."""
        mock_client = MagicMock()
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("HyDE failed")
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Expansion result"
            return mock_response

        mock_client.chat.completions.create.side_effect = side_effect
        mock_openai.return_value = mock_client

        retriever = MultiQueryRetriever(use_hyde=True, use_expansion=True)
        results = retriever.get_queries("What is Python?")

        # Should still have original query
        assert "What is Python?" in results


class TestStepBackPrompting:
    """Tests for step-back prompting."""

    def test_init(self):
        """Test initialization."""
        with patch("piragi.query_transform.OpenAI"):
            stepback = StepBackPrompting()
            assert stepback.model == "llama3.2"

    @patch("piragi.query_transform.OpenAI")
    def test_generate_stepback_query(self, mock_openai):
        """Test generating a step-back query."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = (
            "How does the authentication system work?"
        )
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        stepback = StepBackPrompting()
        result = stepback.generate_stepback_query(
            "What is the error code for invalid API key?"
        )

        assert "authentication" in result.lower() or len(result) > 10

    @patch("piragi.query_transform.OpenAI")
    def test_generate_stepback_fallback(self, mock_openai):
        """Test fallback on error."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API error")
        mock_openai.return_value = mock_client

        stepback = StepBackPrompting()
        query = "Specific query"
        result = stepback.generate_stepback_query(query)

        assert result == query

    @patch("piragi.query_transform.OpenAI")
    def test_get_queries(self, mock_openai):
        """Test getting both original and step-back queries."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "General question"
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        stepback = StepBackPrompting()
        results = stepback.get_queries("Specific query")

        assert len(results) == 2
        assert results[0] == "Specific query"  # Original first
        assert results[1] == "General question"  # Step-back second

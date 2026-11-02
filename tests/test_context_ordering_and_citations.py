"""Tests for U-shaped context attention ordering and inline citation attribution pruning."""

import re
from unittest.mock import MagicMock, patch

import pytest

from piragi import Ragi
from piragi.config import RagiConfig, RetrievalConfig
from piragi.retrieval import (
    Retriever,
    parse_citation_references,
    prune_citations,
    reorder_u_shaped,
)
from piragi.types import Citation


class TestUShapedContextOrdering:
    """Tests for reorder_u_shaped."""

    def test_reorder_empty_and_single(self):
        assert reorder_u_shaped([]) == []
        c1 = Citation(source="s1", chunk="c1", score=0.9)
        assert reorder_u_shaped([c1]) == [c1]

    def test_reorder_two_citations(self):
        c1 = Citation(source="s1", chunk="c1", score=0.9)
        c2 = Citation(source="s2", chunk="c2", score=0.8)
        assert reorder_u_shaped([c1, c2]) == [c1, c2]

    def test_reorder_odd_citations(self):
        # 5 citations: [c0, c1, c2, c3, c4] -> [c0, c2, c4, c3, c1]
        citations = [Citation(source=f"s{i}", chunk=f"c{i}", score=1.0 - i * 0.1) for i in range(5)]
        reordered = reorder_u_shaped(citations)
        sources = [c.source for c in reordered]
        assert sources == ["s0", "s2", "s4", "s3", "s1"]

    def test_reorder_even_citations(self):
        # 4 citations: [c0, c1, c2, c3] -> [c0, c2, c3, c1]
        citations = [Citation(source=f"s{i}", chunk=f"c{i}", score=1.0 - i * 0.1) for i in range(4)]
        reordered = reorder_u_shaped(citations)
        sources = [c.source for c in reordered]
        assert sources == ["s0", "s2", "s3", "s1"]

    def test_extreme_relevance_placement(self):
        citations = [Citation(source=f"s{i}", chunk=f"c{i}", score=1.0 - i * 0.1) for i in range(5)]
        reordered = reorder_u_shaped(citations)
        # Top 2 most relevant must be at the outer ends
        assert reordered[0].source == "s0"
        assert reordered[-1].source == "s1"
        # Lowest relevance must be in the middle
        assert reordered[2].source == "s4"


class TestCitationReferenceParsing:
    """Tests for parse_citation_references."""

    def test_parse_single_bracket(self):
        text = "This is verified by [1] and later by [2]."
        assert parse_citation_references(text, max_citations=5) == [1, 2]

    def test_parse_multi_bracket(self):
        text = "Multiple sources support this: [1, 3] and [2, 4]."
        assert parse_citation_references(text, max_citations=5) == [1, 3, 2, 4]

    def test_parse_range_bracket(self):
        text = "Comprehensive evidence in [1-3]."
        assert parse_citation_references(text, max_citations=5) == [1, 2, 3]

    def test_parse_footnote_syntax(self):
        text = "Footnote citations [^1] and [^3]."
        assert parse_citation_references(text, max_citations=5) == [1, 3]

    def test_parse_textual_source_syntax(self):
        text = "According to Source 1, this works. Source 3 confirms."
        assert parse_citation_references(text, max_citations=5) == [1, 3]

    def test_parse_out_of_bounds_and_years(self):
        text = "In [1995], HTTP [404] error was returned [1]."
        # 1995 and 404 exceed max_citations=5, only 1 is retained
        assert parse_citation_references(text, max_citations=5) == [1]

    def test_parse_deduplication(self):
        text = "Source [1] is great. In fact, [1] is the only one."
        assert parse_citation_references(text, max_citations=5) == [1]

    def test_parse_empty_and_whitespace(self):
        assert parse_citation_references("", 5) == []
        assert parse_citation_references("   ", 5) == []
        assert parse_citation_references("No numbers here.", 5) == []


class TestCitationPruning:
    """Tests for prune_citations."""

    def test_prune_filters_unreferenced(self):
        citations = [Citation(source=f"s{i}", chunk=f"c{i}", score=0.9) for i in range(1, 4)]
        text = "Only source [2] was used."
        active, unref = prune_citations(text, citations)
        assert len(active) == 1
        assert active[0].source == "s2"
        assert active[0].metadata["citation_index"] == 2
        assert len(unref) == 2
        assert [c.source for c in unref] == ["s1", "s3"]

    def test_prune_all_referenced(self):
        citations = [Citation(source=f"s{i}", chunk=f"c{i}", score=0.9) for i in range(1, 4)]
        text = "Used [1], [2], and [3]."
        active, unref = prune_citations(text, citations)
        assert len(active) == 3
        assert len(unref) == 0

    def test_prune_no_references_fallback(self):
        citations = [Citation(source=f"s{i}", chunk=f"c{i}", score=0.9) for i in range(1, 4)]
        text = "The answer without citations."
        active, unref = prune_citations(text, citations)
        # Clean fallback: all returned
        assert len(active) == 3
        assert len(unref) == 0


class TestRetrieverIntegration:
    """Tests for Retriever end-to-end with mock LLM."""

    def test_retriever_u_shaped_and_pruning(self):
        mock_llm = MagicMock()
        mock_completion = MagicMock()
        mock_completion.choices = [
            MagicMock(message=MagicMock(content="Answer using [1] and [5]."))
        ]
        mock_llm.complete.return_value = mock_completion

        retriever = Retriever(
            llm_client=mock_llm,
            enable_reranking=False,
            enable_query_expansion=False,
            context_ordering="u_shaped",
            prune_unreferenced_citations=True,
        )

        citations = [
            Citation(source=f"doc_{i}.txt", chunk=f"chunk {i}", score=1.0 - i * 0.1)
            for i in range(5)
        ]
        answer = retriever.generate_answer("query", citations)

        # Citations in context were [doc_0, doc_2, doc_4, doc_3, doc_1]
        # [1] is doc_0.txt, [5] is doc_1.txt
        assert len(answer.citations) == 2
        assert answer.citations[0].source == "doc_0.txt"
        assert answer.citations[0].metadata["citation_index"] == 1
        assert answer.citations[1].source == "doc_1.txt"
        assert answer.citations[1].metadata["citation_index"] == 5
        assert len(answer.unreferenced_citations) == 3

    def test_retriever_descending_no_pruning(self):
        mock_llm = MagicMock()
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock(message=MagicMock(content="Answer text."))]
        mock_llm.complete.return_value = mock_completion

        retriever = Retriever(
            llm_client=mock_llm,
            enable_reranking=False,
            enable_query_expansion=False,
            context_ordering="descending",
            prune_unreferenced_citations=False,
        )

        citations = [
            Citation(source=f"doc_{i}.txt", chunk=f"chunk {i}", score=1.0 - i * 0.1)
            for i in range(3)
        ]
        answer = retriever.generate_answer("query", citations)

        assert len(answer.citations) == 3
        assert [c.source for c in answer.citations] == ["doc_0.txt", "doc_1.txt", "doc_2.txt"]
        assert len(answer.unreferenced_citations) == 0

    def test_retriever_reranks_first_then_applies_u_shaped_ordering(self):
        """Verify generate_answer runs rerank first, then applies U-shaped ordering to context."""
        mock_llm = MagicMock()
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock(message=MagicMock(content="Answer text [1]."))]
        mock_llm.complete.return_value = mock_completion

        retriever = Retriever(
            llm_client=mock_llm,
            enable_reranking=True,
            enable_query_expansion=False,
            context_ordering="u_shaped",
            prune_unreferenced_citations=False,
        )

        query = "quantum computing"
        citations = [
            Citation(source="doc_0.txt", chunk="quantum computing basics", score=0.9),
            Citation(source="doc_1.txt", chunk="computing systems architecture", score=0.8),
            Citation(source="doc_2.txt", chunk="hardware memory principles", score=0.7),
            Citation(source="doc_3.txt", chunk="network routing protocols", score=0.6),
            Citation(source="doc_4.txt", chunk="database storage engines", score=0.5),
        ]
        answer = retriever.generate_answer(query, citations)
        assert answer.text == "Answer text [1]."

        mock_llm.complete.assert_called_once()
        call_messages = mock_llm.complete.call_args[1]["messages"]
        user_prompt = next(m["content"] for m in call_messages if m["role"] == "user")

        # After rerank descending [doc_0, doc_1, doc_2, doc_3, doc_4],
        # U-shaped reorder produces: [doc_0, doc_2, doc_4, doc_3, doc_1]
        source_order = re.findall(r"Source \d+ \((doc_\d\.txt)\):", user_prompt)
        assert source_order == ["doc_0.txt", "doc_2.txt", "doc_4.txt", "doc_3.txt", "doc_1.txt"]


class TestRagiConfigValidation:
    """Tests for Pydantic configuration validation."""

    def test_default_config_values(self):
        config = RagiConfig()
        assert config.retrieval.context_ordering == "u_shaped"
        assert config.retrieval.prune_unreferenced_citations is True

    def test_valid_context_ordering_options(self):
        c1 = RagiConfig(retrieval=RetrievalConfig(context_ordering="u_shaped"))
        assert c1.retrieval.context_ordering == "u_shaped"
        c2 = RagiConfig(retrieval=RetrievalConfig(context_ordering="descending"))
        assert c2.retrieval.context_ordering == "descending"

    def test_invalid_context_ordering_raises_validation_error(self):
        with pytest.raises(Exception):
            RagiConfig(retrieval={"context_ordering": "random_order"})


class TestRagiDefaultAskContextOrdering:
    """Regression tests for default Ragi().ask() context ordering (Reviewer 2 Finding 1)."""

    @patch("openai.OpenAI")
    def test_default_ragi_ask_formats_context_in_u_shaped_order(self, mock_openai, tmp_path):
        """Default Ragi().ask() (enable_reranking=True) formats U-shaped context in prompt."""
        mock_client = MagicMock()
        mock_completion = MagicMock()
        mock_completion.choices = [
            MagicMock(message=MagicMock(content="Answer grounded in [1] and [5]."))
        ]
        mock_client.chat.completions.create.return_value = mock_completion
        mock_openai.return_value = mock_client

        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = [0.1, 0.2]
        mock_embedder.model_name = "mock-model"

        # Instantiate default Ragi (default: enable_reranking=True, context_ordering="u_shaped")
        kb = Ragi(persist_dir=str(tmp_path), embedder=mock_embedder)
        assert kb.retriever.enable_reranking is True
        assert kb._retrieval_pipeline.context_ordering == "u_shaped"

        # 5 citations returned by store search in descending order
        citations = [
            Citation(
                source=f"source_{i}.txt",
                chunk=f"chunk content for source {i}",
                score=1.0 - (i * 0.1),
            )
            for i in range(5)
        ]
        kb.store.search = MagicMock(return_value=citations)

        query = "How is context ordered in prompt?"
        answer = kb.ask(query, top_k=5)
        assert answer.text

        # Verify LLM generation was invoked (after optional query expansion)
        assert mock_client.chat.completions.create.call_count >= 1
        gen_call = mock_client.chat.completions.create.call_args_list[-1]
        messages = gen_call[1]["messages"]
        user_prompt = next(m["content"] for m in messages if m["role"] == "user")

        # In U-shaped layout:
        # [s0, s1, s2, s3, s4] -> [s0, s2, s4, s3, s1]
        sources_in_prompt = re.findall(r"Source \d+ \((source_\d\.txt)\):", user_prompt)
        assert sources_in_prompt == [
            "source_0.txt",
            "source_2.txt",
            "source_4.txt",
            "source_3.txt",
            "source_1.txt",
        ]
        # Outer ends have top relevance (source_0.txt, source_1.txt)
        assert sources_in_prompt[0] == "source_0.txt"
        assert sources_in_prompt[-1] == "source_1.txt"
        # Middle trough has lowest relevance (source_4.txt)
        assert sources_in_prompt[2] == "source_4.txt"

    @patch("openai.OpenAI")
    def test_ragi_ask_with_descending_ordering_preserves_descending_order(
        self, mock_openai, tmp_path
    ):
        """Verify context_ordering='descending' presents citations to LLM in descending score."""
        mock_client = MagicMock()
        mock_completion = MagicMock()
        mock_completion.choices = [
            MagicMock(message=MagicMock(content="Answer grounded in [1]."))
        ]
        mock_client.chat.completions.create.return_value = mock_completion
        mock_openai.return_value = mock_client

        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = [0.1, 0.2]
        mock_embedder.model_name = "mock-model"

        kb = Ragi(
            persist_dir=str(tmp_path),
            embedder=mock_embedder,
            config={"retrieval": {"context_ordering": "descending"}},
        )
        assert kb._retrieval_pipeline.context_ordering == "descending"

        citations = [
            Citation(
                source=f"source_{i}.txt",
                chunk=f"chunk content for source {i}",
                score=1.0 - (i * 0.1),
            )
            for i in range(5)
        ]
        kb.store.search = MagicMock(return_value=citations)

        answer = kb.ask("How is context ordered in prompt?", top_k=5)
        assert answer.text

        assert mock_client.chat.completions.create.call_count >= 1
        gen_call = mock_client.chat.completions.create.call_args_list[-1]
        messages = gen_call[1]["messages"]
        user_prompt = next(m["content"] for m in messages if m["role"] == "user")

        sources_in_prompt = re.findall(r"Source \d+ \((source_\d\.txt)\):", user_prompt)
        assert sources_in_prompt == [
            "source_0.txt",
            "source_1.txt",
            "source_2.txt",
            "source_3.txt",
            "source_4.txt",
        ]

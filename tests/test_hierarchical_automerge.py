"""Tests for hierarchical auto-merging context deduplication."""

from unittest.mock import MagicMock
import pytest
from piragi.config import RagiConfig, RetrievalConfig
from piragi.pipelines.retrieval import RetrievalPipeline
from piragi.types import Citation


class TestHierarchicalAutoMerging:
    """Unit tests for _expand_to_parent_context and auto-merging deduplication."""

    @pytest.fixture
    def pipeline(self):
        return RetrievalPipeline(
            embedder=MagicMock(),
            store=MagicMock(),
            retriever=MagicMock(),
            use_hierarchical=True,
            auto_merge_parents=True,
        )

    def test_deduplicates_multiple_sibling_children(self, pipeline):
        """Three sibling child chunks must merge into exactly one parent citation."""
        parent_text = "This is the full parent section containing comprehensive context."
        citations = [
            Citation(
                source="guide.md",
                chunk="child snippet 0",
                score=0.85,
                metadata={
                    "parent_index": 0,
                    "parent_text": parent_text,
                    "chunk_index": 0,
                    "doc_id": "G1",
                },
            ),
            Citation(
                source="guide.md",
                chunk="child snippet 1",
                score=0.92,
                metadata={
                    "parent_index": 0,
                    "parent_text": parent_text,
                    "chunk_index": 1,
                    "doc_id": "G1",
                },
            ),
            Citation(
                source="guide.md",
                chunk="child snippet 2",
                score=0.78,
                metadata={
                    "parent_index": 0,
                    "parent_text": parent_text,
                    "chunk_index": 2,
                    "doc_id": "G1",
                },
            ),
        ]

        result = pipeline._expand_to_parent_context(citations)

        assert len(result) == 1
        merged = result[0]
        assert merged.source == "guide.md"
        assert merged.chunk == parent_text
        assert merged.score == 0.92  # Max score
        assert merged.metadata["chunk_type"] == "parent"
        assert merged.metadata["merged_children_count"] == 3
        assert merged.metadata["child_indices"] == [0, 1, 2]
        assert merged.metadata["doc_id"] == "G1"
        assert "parent_text" not in merged.metadata

    def test_single_child_expands_to_parent(self, pipeline):
        """A single child chunk expands to parent text with merged_children_count=1."""
        parent_text = "Single child parent text."
        citations = [
            Citation(
                source="notes.txt",
                chunk="single child",
                score=0.88,
                metadata={"parent_index": 1, "parent_text": parent_text, "chunk_index": 4},
            )
        ]

        result = pipeline._expand_to_parent_context(citations)

        assert len(result) == 1
        assert result[0].chunk == parent_text
        assert result[0].score == 0.88
        assert result[0].metadata["merged_children_count"] == 1
        assert result[0].metadata["child_indices"] == [4]

    def test_preserves_non_hierarchical_citations(self, pipeline):
        """Citations without parent_text metadata must pass through untouched."""
        regular = Citation(
            source="standard.md",
            chunk="Regular non-hierarchical chunk.",
            score=0.80,
            metadata={"category": "api"},
        )

        result = pipeline._expand_to_parent_context([regular])

        assert len(result) == 1
        assert result[0].chunk == "Regular non-hierarchical chunk."
        assert result[0].score == 0.80
        assert result[0].metadata == {"category": "api"}

    def test_mixed_hierarchical_and_regular_citations(self, pipeline):
        """Mixed pool sorts correctly and consolidates only hierarchical siblings."""
        p_text = "Parent context for doc A."
        c1 = Citation(
            source="doc_a.md",
            chunk="c1",
            score=0.90,
            metadata={"parent_index": 0, "parent_text": p_text, "chunk_index": 0},
        )
        c2 = Citation(
            source="doc_b.md",
            chunk="Independent chunk B",
            score=0.85,
            metadata={"tag": "regular"},
        )
        c3 = Citation(
            source="doc_a.md",
            chunk="c2",
            score=0.75,
            metadata={"parent_index": 0, "parent_text": p_text, "chunk_index": 1},
        )

        result = pipeline._expand_to_parent_context([c1, c2, c3])

        assert len(result) == 2
        # Doc A merged has max score 0.90, Doc B has 0.85
        assert result[0].source == "doc_a.md"
        assert result[0].chunk == p_text
        assert result[0].score == 0.90
        assert result[0].metadata["merged_children_count"] == 2

        assert result[1].source == "doc_b.md"
        assert result[1].chunk == "Independent chunk B"
        assert result[1].score == 0.85

    def test_different_documents_same_parent_index_isolated(self, pipeline):
        """Chunks from different documents with parent_index=0 are never cross-merged."""
        c_a = Citation(
            source="doc_1.md",
            chunk="child 1",
            score=0.95,
            metadata={"parent_index": 0, "parent_text": "Doc 1 Parent", "chunk_index": 0},
        )
        c_b = Citation(
            source="doc_2.md",
            chunk="child 2",
            score=0.91,
            metadata={"parent_index": 0, "parent_text": "Doc 2 Parent", "chunk_index": 0},
        )

        result = pipeline._expand_to_parent_context([c_a, c_b])

        assert len(result) == 2
        assert result[0].source == "doc_1.md"
        assert result[0].chunk == "Doc 1 Parent"
        assert result[1].source == "doc_2.md"
        assert result[1].chunk == "Doc 2 Parent"

    def test_different_parents_same_text_distinguished_by_index(self, pipeline):
        """Repeated boilerplate text with different parent_index produces distinct citations."""
        boilerplate = "Copyright (c) 2026. All rights reserved."
        c1 = Citation(
            source="terms.md",
            chunk="c1",
            score=0.90,
            metadata={"parent_index": 0, "parent_text": boilerplate, "chunk_index": 0},
        )
        c2 = Citation(
            source="terms.md",
            chunk="c2",
            score=0.80,
            metadata={"parent_index": 5, "parent_text": boilerplate, "chunk_index": 12},
        )

        result = pipeline._expand_to_parent_context([c1, c2])

        assert len(result) == 2
        assert result[0].metadata["parent_index"] == 0
        assert result[1].metadata["parent_index"] == 5

    def test_missing_or_empty_parent_text_handled_gracefully(self, pipeline):
        """Missing or whitespace parent_text does not throw and leaves citation untouched."""
        c1 = Citation(
            source="bad_meta.md",
            chunk="Raw child text 1",
            score=0.70,
            metadata={"parent_text": None, "chunk_type": "child"},
        )
        c2 = Citation(
            source="bad_meta.md",
            chunk="Raw child text 2",
            score=0.65,
            metadata={"parent_text": "   ", "chunk_type": "child"},
        )

        result = pipeline._expand_to_parent_context([c1, c2])

        assert len(result) == 2
        assert result[0].chunk == "Raw child text 1"
        assert result[1].chunk == "Raw child text 2"

    def test_empty_citations_list(self, pipeline):
        """Empty input returns empty list."""
        assert pipeline._expand_to_parent_context([]) == []

    def test_auto_merge_disabled_falls_back_to_legacy(self, pipeline):
        """When auto_merge_parents=False, legacy 1-to-1 expansion is preserved."""
        pipeline.auto_merge_parents = False
        p_text = "Shared parent text"
        c1 = Citation(
            source="test.md",
            chunk="child 1",
            score=0.90,
            metadata={"parent_index": 0, "parent_text": p_text},
        )
        c2 = Citation(
            source="test.md",
            chunk="child 2",
            score=0.85,
            metadata={"parent_index": 0, "parent_text": p_text},
        )

        result = pipeline._expand_to_parent_context([c1, c2])

        assert len(result) == 2  # Duplication preserved when disabled
        assert result[0].chunk == p_text
        assert result[1].chunk == p_text

    def test_retrieval_config_validation(self):
        """RetrievalConfig accepts and validates auto_merge_parents."""
        cfg_default = RetrievalConfig()
        assert cfg_default.auto_merge_parents is True

        cfg_disabled = RetrievalConfig(auto_merge_parents=False)
        assert cfg_disabled.auto_merge_parents is False

        ragi_cfg = RagiConfig(retrieval=RetrievalConfig(auto_merge_parents=False))
        assert ragi_cfg.retrieval.auto_merge_parents is False

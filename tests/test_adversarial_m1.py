"""Adversarial and empirical stress tests for U-shaped context attention ordering

and inline citation attribution parsing and pruning (Milestone 1 Challenger suite).
"""

import math
import re
import time
from typing import List
import pytest

from piragi.types import Citation, Answer
from piragi.retrieval import (
    reorder_u_shaped,
    parse_citation_references,
    prune_citations,
)


# =====================================================================
# 1. EMPIRICAL STRESS TESTS: U-SHAPED CONTEXT ATTENTION ORDERING
# =====================================================================

class TestUShapedAttentionStress:
    """Stress tests and mathematical invariant verification for reorder_u_shaped."""

    @pytest.mark.parametrize("n", [0, 1, 2, 3, 4, 5, 10, 15, 50, 100, 500, 1000])
    def test_reorder_sizes_invariant(self, n: int):
        """Verify bijection, parity, and length preservation for all list sizes up to 1000."""
        original = [
            Citation(source=f"doc_{i}", chunk=f"chunk_{i}", score=1.0 - (i / max(1, n)))
            for i in range(n)
        ]
        reordered = reorder_u_shaped(original)

        # Invariant 1: Length preservation
        assert len(reordered) == n

        # Invariant 2: Bijection (exact set and multiset preservation, no elements dropped or duplicated)
        orig_sources = [c.source for c in original]
        reordered_sources = [c.source for c in reordered]
        assert set(reordered_sources) == set(orig_sources)
        assert len(set(reordered_sources)) == n

        # Invariant 3: Immutability / stability of input
        assert [c.source for c in original] == orig_sources

        # Invariant 4: Positional invariants for N >= 1, N >= 2, N >= 3
        if n >= 1:
            assert reordered[0].source == original[0].source  # Top-1 highest relevance at start
        if n >= 2:
            assert reordered[-1].source == original[1].source  # Top-2 highest relevance at end
        if n >= 3:
            # The lowest relevance item (original[n-1]) must be in the interior (the trough of the U)
            lowest_item = original[n - 1]
            if n % 2 == 1:
                # Odd n: middle is at index (n-1)//2
                assert reordered[(n - 1) // 2].source == lowest_item.source
            else:
                # Even n: middle is at index n // 2
                assert reordered[n // 2].source == lowest_item.source

    def test_all_sizes_from_0_to_1000_exhaustively(self):
        """Exhaustive check of every single integer size from 0 to 1,000."""
        for n in range(1001):
            original = [
                Citation(source=f"s_{i}", chunk=f"c_{i}", score=1.0 - (i / max(1, n)))
                for i in range(n)
            ]
            reordered = reorder_u_shaped(original)
            assert len(reordered) == n
            assert set(c.source for c in reordered) == set(c.source for c in original)
            if n >= 1:
                assert reordered[0].source == "s_0"
            if n >= 2:
                assert reordered[-1].source == "s_1"

    def test_u_shape_monotonicity_profile(self):
        """Verify that scores form a strict U-curve: descending on left half, ascending on right half."""
        n = 100
        # Distinct scores strictly decreasing
        original = [
            Citation(source=f"s_{i}", chunk=f"c_{i}", score=float(1000 - i))
            for i in range(n)
        ]
        reordered = reorder_u_shaped(original)
        scores = [c.score for c in reordered]

        # Left half: indices 0, 2, 4, ... -> scores strictly decreasing
        half_point = math.ceil(n / 2)
        left_scores = scores[:half_point]
        for i in range(len(left_scores) - 1):
            assert left_scores[i] > left_scores[i + 1], f"Left side not decreasing at {i}"

        # Right half: scores strictly increasing towards the end
        right_scores = scores[half_point:]
        for i in range(len(right_scores) - 1):
            assert right_scores[i] < right_scores[i + 1], f"Right side not increasing at {i}"

    def test_reorder_latency_benchmark(self):
        """Verify O(N) performance on large N=10,000 list."""
        citations = [
            Citation(source=f"s_{i}", chunk=f"c_{i}", score=1.0)
            for i in range(10000)
        ]
        start_time = time.perf_counter()
        reordered = reorder_u_shaped(citations)
        elapsed = time.perf_counter() - start_time
        assert len(reordered) == 10000
        assert elapsed < 0.05, f"Reordering 10,000 items took {elapsed:.4f}s (exceeds 50ms budget)"


# =====================================================================
# 2. ADVERSARIAL STRESS TESTS: CITATION ATTRIBUTION PARSING
# =====================================================================

class TestCitationParsingAdversarial:
    """Adversarial stress testing of parse_citation_references."""

    def test_negative_and_zero_citations(self):
        """Verify that negative numbers and 0 are rejected and do not raise exceptions."""
        cases = [
            ("Text with [-1] and [0].", 5, []),
            ("Range with negative [-5-3].", 5, []),
            ("Source 0 and Source -1.", 5, []),
            ("Mixed: [-1], [0], [1], [2].", 5, [1, 2]),
            ("Zero in multi: [0, 1, 2].", 5, [1, 2]),
        ]
        for text, max_c, expected in cases:
            res = parse_citation_references(text, max_c)
            assert res == expected, f"Failed for text '{text}': got {res}, expected {expected}"

    def test_out_of_bounds_and_large_numbers(self):
        """Verify out of bounds numbers are dropped."""
        cases = [
            ("Referencing [999] when max is 5.", 5, []),
            ("Referencing [6] when max is 5.", 5, []),
            ("Referencing [1000000000] huge int.", 10, []),
            ("Year citations like [1999] and [2024] but [2] is valid.", 5, [2]),
            ("HTTP [404] and [500] status codes with [3].", 5, [3]),
        ]
        for text, max_c, expected in cases:
            res = parse_citation_references(text, max_c)
            assert res == expected, f"Failed for text '{text}': got {res}, expected {expected}"

    def test_malformed_bracket_sequences(self):
        """Verify non-numeric brackets and malformed syntax do not crash."""
        malformed = [
            "[abc]",
            "[NaN]",
            "[None]",
            "[null]",
            "[]",
            "[ ]",
            "[\t\n]",
            "[-]",
            "[,]",
            "[--]",
            "[1, , 2]",
            "[1,,2]",
            "[[1]]",
            "[[1, 2]]",
            "[ [ 1 ] ]",
            "[1 [2] 3]",
            "[#1]",
            "[$1]",
        ]
        for sample in malformed:
            # Must not raise exception
            res = parse_citation_references(sample, max_citations=5)
            assert isinstance(res, list)

    def test_nested_brackets_extraction(self):
        """Test extraction behavior with nested brackets."""
        # '[[1]]' contains '[1]'
        assert 1 in parse_citation_references("[[1]]", max_citations=5)
        # '[[1], [2]]'
        res = parse_citation_references("[[1], [2]]", max_citations=5)
        assert res == [1, 2]

    def test_range_adversarial_syntax(self):
        """Test range edge cases: inverted ranges, excessive spans, unicode dashes."""
        cases = [
            ("Inverted range [5-1].", 5, []),
            ("Equal range [2-2].", 5, [2]),
            ("Excessive range [1-100].", 100, []),  # Exceeds max span 20
            ("Valid span 20 [1-21].", 50, list(range(1, 22))),
            ("Span 21 [1-22].", 50, []),  # Exceeds 20
            ("En-dash [1–3].", 5, [1, 2, 3]),
            ("Em-dash [1—3].", 5, [1, 2, 3]),
            ("Triple hyphen [1-2-3].", 5, []),
        ]
        for text, max_c, expected in cases:
            res = parse_citation_references(text, max_c)
            assert res == expected, f"Range test failed for '{text}': got {res}, expected {expected}"

    def test_textual_sources_adversarial(self):
        """Test textual references like 'Source 1', 'Sources 1, 2 and 3'."""
        cases = [
            ("As seen in Source 1 and Source 2.", 5, [1, 2]),
            ("Sources 1, 2 and 3 state...", 5, [1, 2, 3]),
            ("According to Source 1-3.", 5, [1, 2, 3]),
            ("Source 99 (out of bounds).", 5, []),
            ("Source -5 (negative).", 5, []),
            ("Source abc (not a number).", 5, []),
            ("[Source 1] should not be double-counted by Source pattern.", 5, [1]),
        ]
        for text, max_c, expected in cases:
            res = parse_citation_references(text, max_c)
            assert res == expected, f"Failed for '{text}': got {res}, expected {expected}"

    def test_unicode_and_multilingual_contexts(self):
        """Verify citations in non-English, multilingual, and emoji contexts."""
        cases = [
            ("根据文献 [1] 和 [3] 的分析，结果有效。", 5, [1, 3]),
            ("参考文献 [1] を参照してください。また [2] も確認。", 5, [1, 2]),
            ("وفقاً للمصدر [1] والمصدر [4] في التقرير", 5, [1, 4]),
            ("Источники [1] и [2] подтверждают гипотезу.", 5, [1, 2]),
            ("Emoji test 🚀 [1] 💡 [2] 🔍 [3]", 5, [1, 2, 3]),
            ("Combining characters: école [1] café [2]", 5, [1, 2]),
            ("RTL text with brackets: [1] אבג [2]", 5, [1, 2]),
        ]
        for text, max_c, expected in cases:
            res = parse_citation_references(text, max_c)
            assert res == expected, f"Unicode test failed for '{text}': got {res}, expected {expected}"

    def test_deduplication_and_order_of_first_appearance(self):
        """Verify citations are returned in the exact order of first appearance without duplicate."""
        text = "First [3], then [1], again [3], then [2], then [1], finally [4]."
        res = parse_citation_references(text, max_citations=5)
        assert res == [3, 1, 2, 4]

    def test_boundary_max_citations(self):
        """Verify behavior with max_citations <= 0 or extreme."""
        assert parse_citation_references("Citation [1].", max_citations=0) == []
        assert parse_citation_references("Citation [1].", max_citations=-5) == []
        assert parse_citation_references("", max_citations=5) == []
        assert parse_citation_references(None, max_citations=5) == []  # type: ignore

    def test_redos_and_pathological_inputs(self):
        """Verify regex does not exhibit exponential backtracking (ReDoS) on adversarial input."""
        # 10,000 opening brackets
        pathological_1 = "[" * 5000 + "1" + "]" * 5000
        start = time.perf_counter()
        parse_citation_references(pathological_1, max_citations=5)
        elapsed_1 = time.perf_counter() - start
        assert elapsed_1 < 0.1, f"Pathological brackets took {elapsed_1}s"

        # 10,000 commas inside bracket
        pathological_2 = "[" + "," * 10000 + "]"
        start = time.perf_counter()
        parse_citation_references(pathological_2, max_citations=5)
        elapsed_2 = time.perf_counter() - start
        assert elapsed_2 < 0.1, f"Pathological commas took {elapsed_2}s"

        # 10,000 repetitions of 'Sources and and and'
        pathological_3 = "Sources " + "and " * 5000 + "1"
        start = time.perf_counter()
        parse_citation_references(pathological_3, max_citations=5)
        elapsed_3 = time.perf_counter() - start
        assert elapsed_3 < 0.1, f"Pathological textual repetition took {elapsed_3}s"


# =====================================================================
# 3. EMPIRICAL STRESS TESTS: CITATION PRUNING
# =====================================================================

class TestCitationPruningStress:
    """Stress tests for prune_citations."""

    def test_empty_candidates_list(self):
        """prune_citations with empty candidates should return ([], [])."""
        active, unref = prune_citations("Used [1].", [])
        assert active == []
        assert unref == []

    def test_zero_referenced_fallback(self):
        """When text contains 0 citations, fallback should return all citations as active."""
        candidates = [
            Citation(source=f"s{i}", chunk=f"c{i}", score=0.8)
            for i in range(10)
        ]
        active, unref = prune_citations("No citations mentioned.", candidates)
        assert len(active) == 10
        assert len(unref) == 0
        assert [c.source for c in active] == [c.source for c in candidates]

    def test_all_referenced(self):
        """When text references all citations, all should be active."""
        candidates = [
            Citation(source=f"s{i}", chunk=f"c{i}", score=0.8)
            for i in range(1, 6)
        ]
        text = "Referencing [1], [2], [3], [4], and [5]."
        active, unref = prune_citations(text, candidates)
        assert len(active) == 5
        assert len(unref) == 0
        assert [c.source for c in active] == [f"s{i}" for i in range(1, 6)]
        for idx, c in enumerate(active, 1):
            assert c.metadata["citation_index"] == idx

    def test_partial_referenced(self):
        """Check correct segregation into active and unreferenced."""
        candidates = [
            Citation(source=f"s{i}", chunk=f"c{i}", score=0.8)
            for i in range(1, 6)
        ]
        text = "Only [2] and [4] were cited."
        active, unref = prune_citations(text, candidates)
        assert len(active) == 2
        assert [c.source for c in active] == ["s2", "s4"]
        assert [c.metadata["citation_index"] for c in active] == [2, 4]

        assert len(unref) == 3
        assert [c.source for c in unref] == ["s1", "s3", "s5"]
        assert [c.metadata["citation_index"] for c in unref] == [1, 3, 5]

    def test_only_out_of_bounds_citations_fallback(self):
        """When text references citations that are out of bounds (e.g. [99]), treat as 0 citations cited."""
        candidates = [
            Citation(source=f"s{i}", chunk=f"c{i}", score=0.8)
            for i in range(3)
        ]
        text = "Referencing invalid [99] and [100]."
        active, unref = prune_citations(text, candidates)
        # Should cleanly fall back to all candidates active
        assert len(active) == 3
        assert len(unref) == 0

    def test_deep_copy_metadata_isolation(self):
        """Verify that prune_citations does NOT mutate the original Citation objects or their metadata dicts."""
        orig_metadata = {"existing_key": "original_value"}
        c1 = Citation(source="s1", chunk="c1", score=0.9, metadata=orig_metadata.copy())
        c2 = Citation(source="s2", chunk="c2", score=0.8, metadata=orig_metadata.copy())
        candidates = [c1, c2]

        active, unref = prune_citations("Using [1].", candidates)

        # Active citation must have citation_index in metadata
        assert active[0].metadata["citation_index"] == 1

        # Original citation must NOT be mutated!
        assert "citation_index" not in c1.metadata
        assert c1.metadata == orig_metadata
        assert "citation_index" not in c2.metadata
        assert c2.metadata == orig_metadata

    def test_large_candidate_list_scalability(self):
        """Verify pruning works cleanly with 1,000 candidate citations."""
        candidates = [
            Citation(source=f"s_{i}", chunk=f"c_{i}", score=1.0 - (i / 1000.0))
            for i in range(1, 1001)
        ]
        text = "Based on [1], [500], [1000], and [1-3]."
        start = time.perf_counter()
        active, unref = prune_citations(text, candidates)
        elapsed = time.perf_counter() - start

        # Referenced: [1, 500, 1000, 1, 2, 3] -> unique: 1, 500, 1000, 2, 3 (5 active)
        assert len(active) == 5
        assert len(unref) == 995
        assert set(c.metadata["citation_index"] for c in active) == {1, 2, 3, 500, 1000}
        assert elapsed < 0.1, f"Pruning 1,000 citations took {elapsed:.4f}s"

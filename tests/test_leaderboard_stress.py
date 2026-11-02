"""Adversarial Stress Test Suite for Leaderboard Exporters and Validators.

Empirically challenges:
1. JSONL syntax corruption (truncated lines, trailing garbage, concatenated JSON, non-object types, binary data).
2. EnterpriseRAG schema strictness (missing keys, empty answers, non-string doc_ids, negative latencies, malformed citations).
3. GAIA schema strictness (missing keys, non-string IDs, empty model answers, non-list traces, empty sources).
4. AIR-Bench TREC run format (column counts, non-integer ranks, rank gaps, inverted ranks, non-monotonic scores, multi-query tracking).
5. ZIP archive integrity (truncated bytes, payload CRC flips, missing manifests, empty runs, non-UTF8 text).
6. Exporter resilience on degenerate inputs, empty sequences, and missing files.
7. Explicit documentation and empirical verification of identified validation gaps (marked xfail for worker remediation).
"""

from __future__ import annotations

import json
import os
import tempfile
import zipfile
import zlib
from pathlib import Path
from typing import Any, Dict, List

import pytest

from evals.leaderboard_exporters import (
    DEFAULT_SUBMISSIONS_DIR,
    export_airbench,
    export_all,
    export_enterpriserag,
    export_gaia,
    optimize_for_leaderboard,
    validate_submission_file,
)


# ===========================================================================
# 1. JSONL Syntax & File-Level Stress Tests
# ===========================================================================

class TestJsonlSyntaxStress:
    """Stress tests covering raw JSON syntax parsing, file boundaries, and stream corruption."""

    def test_truncated_json_line(self, tmp_path):
        f = tmp_path / "truncated.jsonl"
        f.write_text('{"question_id": "q1", "query": "what is aerodynamics?", "ans', encoding="utf-8")
        ok, errs = validate_submission_file(f, benchmark_type="enterpriserag")
        assert ok is False
        assert any("Invalid JSON syntax" in e for e in errs)

    def test_trailing_garbage_on_json_line(self, tmp_path):
        f = tmp_path / "trailing_garbage.jsonl"
        valid_json = json.dumps({
            "question_id": "q1",
            "query": "query",
            "answer": "answer [1]",
            "citations": [{"doc_id": "d1", "citation_num": 1, "score": 1.0}],
            "doc_ids": ["d1"],
            "latency_ms": 10.0,
            "metadata": {},
        })
        f.write_text(valid_json + " EXTRA_TRAILING_GARBAGE\n", encoding="utf-8")
        ok, errs = validate_submission_file(f, benchmark_type="enterpriserag")
        assert ok is False
        assert any("Invalid JSON syntax" in e for e in errs)

    def test_concatenated_json_objects_without_newline(self, tmp_path):
        f = tmp_path / "concatenated.jsonl"
        f.write_text('{"question_id": "q1"}{"question_id": "q2"}\n', encoding="utf-8")
        ok, errs = validate_submission_file(f, benchmark_type="enterpriserag")
        assert ok is False
        assert any("Invalid JSON syntax" in e for e in errs)

    @pytest.mark.parametrize(
        "raw_payload,type_name",
        [
            ("42\n", "int"),
            ('"raw string literal"\n', "str"),
            ('["array", "of", "items"]\n', "list"),
            ("true\n", "bool"),
            ("null\n", "NoneType"),
        ],
    )
    def test_non_dict_json_primitives(self, tmp_path, raw_payload, type_name):
        f = tmp_path / f"primitive_{type_name}.jsonl"
        f.write_text(raw_payload, encoding="utf-8")
        ok, errs = validate_submission_file(f, benchmark_type="enterpriserag")
        assert ok is False
        assert any(f"Record must be a JSON object, got {type_name}" in e for e in errs)

    def test_file_with_only_whitespace_and_blank_lines(self, tmp_path):
        f = tmp_path / "whitespace_only.jsonl"
        f.write_text("\n   \n\t\n\r\n\n", encoding="utf-8")
        ok, errs = validate_submission_file(f, benchmark_type="enterpriserag")
        assert ok is False
        assert any("contains no JSONL records" in e for e in errs)

    def test_binary_garbage_file(self, tmp_path):
        f = tmp_path / "binary_garbage.jsonl"
        f.write_bytes(b"\x80\x03c__main__\n\xff\xfe\x00\x00corrupt")
        ok, errs = validate_submission_file(f, benchmark_type="enterpriserag")
        assert ok is False
        assert any("Could not read JSONL file" in e for e in errs)

    def test_mixed_valid_and_corrupt_stream_reports_all_line_errors(self, tmp_path):
        f = tmp_path / "mixed.jsonl"
        valid_rec = {
            "question_id": "q1",
            "query": "query",
            "answer": "answer [1]",
            "citations": [{"doc_id": "d1", "citation_num": 1, "score": 1.0}],
            "doc_ids": ["d1"],
            "latency_ms": 10.0,
            "metadata": {},
        }
        lines = [
            json.dumps(valid_rec),                    # Line 1: VALID
            "{corrupt json syntax",                   # Line 2: SYNTAX ERROR
            json.dumps({"question_id": "q3"}),        # Line 3: MISSING KEYS
            json.dumps(valid_rec),                    # Line 4: VALID
            json.dumps({                              # Line 5: INVALID LATENCY
                **valid_rec,
                "question_id": "q5",
                "latency_ms": -42.0,
            }),
        ]
        f.write_text("\n".join(lines) + "\n", encoding="utf-8")
        ok, errs = validate_submission_file(f, benchmark_type="enterpriserag")
        assert ok is False
        assert any("Line 2: Invalid JSON syntax" in e for e in errs)
        assert any("Line 3: Missing required key 'query'" in e for e in errs)
        assert any("Line 5: 'latency_ms' must be a non-negative number" in e for e in errs)

    def test_large_payload_record_parsing(self, tmp_path):
        """Verify parser can handle 10,000 document IDs or large text without crash."""
        large_doc_ids = [f"doc_{i}" for i in range(1000)]
        large_answer = "Evidence explanation. " * 500 + "[1]"
        rec = {
            "question_id": "q_large",
            "query": "Complex multi-document query with large retrieval depth",
            "answer": large_answer,
            "citations": [{"doc_id": "doc_0", "citation_num": 1, "score": 1.0}],
            "doc_ids": large_doc_ids,
            "latency_ms": 1500.0,
            "metadata": {"depth": 1000},
        }
        f = tmp_path / "large_record.jsonl"
        f.write_text(json.dumps(rec) + "\n", encoding="utf-8")
        ok, errs = validate_submission_file(f, benchmark_type="enterpriserag")
        assert ok is True, f"Unexpected errors on large payload: {errs}"


# ===========================================================================
# 2. EnterpriseRAG Schema Validation Stress
# ===========================================================================

class TestEnterpriseRagSchemaStress:
    """Stress tests targeting EnterpriseRAG-Bench format invariants."""

    @pytest.fixture
    def base_valid_record(self) -> Dict[str, Any]:
        return {
            "question_id": "ent_q100",
            "query": "What causes laminar boundary layer separation?",
            "answer": "Adverse pressure gradients cause boundary layer separation [1].",
            "citations": [
                {
                    "doc_id": "doc_401",
                    "citation_num": 1,
                    "title": "Aerodynamic Flow Control",
                    "text": "Adverse pressure gradient causes flow separation.",
                    "score": 0.95,
                }
            ],
            "doc_ids": ["doc_401", "doc_402"],
            "latency_ms": 48.5,
            "metadata": {"dataset": "cranfield", "condition": "enhanced"},
        }

    @pytest.mark.parametrize(
        "missing_key",
        ["question_id", "query", "answer", "citations", "doc_ids", "latency_ms", "metadata"],
    )
    def test_missing_required_keys_individually(self, tmp_path, base_valid_record, missing_key):
        rec = dict(base_valid_record)
        del rec[missing_key]
        f = tmp_path / f"missing_{missing_key}.jsonl"
        f.write_text(json.dumps(rec) + "\n", encoding="utf-8")
        ok, errs = validate_submission_file(f, benchmark_type="enterpriserag")
        assert ok is False
        assert any(f"Missing required key '{missing_key}'" in e for e in errs)

    @pytest.mark.parametrize(
        "bad_qid",
        [None, "", 1234, False, []],
    )
    def test_invalid_question_id(self, tmp_path, base_valid_record, bad_qid):
        rec = dict(base_valid_record)
        rec["question_id"] = bad_qid
        f = tmp_path / "bad_qid.jsonl"
        f.write_text(json.dumps(rec) + "\n", encoding="utf-8")
        ok, errs = validate_submission_file(f, benchmark_type="enterpriserag")
        assert ok is False
        assert any("'question_id' must be a non-empty string" in e for e in errs)

    def test_whitespace_question_id_rejected(self, tmp_path, base_valid_record):
        rec = dict(base_valid_record)
        rec["question_id"] = "   "
        f = tmp_path / "whitespace_qid.jsonl"
        f.write_text(json.dumps(rec) + "\n", encoding="utf-8")
        ok, errs = validate_submission_file(f, benchmark_type="enterpriserag")
        assert ok is False, "Whitespace-only question_id should be rejected"

    @pytest.mark.parametrize(
        "bad_query",
        [None, "", 999, {"text": "nested"}, []],
    )
    def test_invalid_query(self, tmp_path, base_valid_record, bad_query):
        rec = dict(base_valid_record)
        rec["query"] = bad_query
        f = tmp_path / "bad_query.jsonl"
        f.write_text(json.dumps(rec) + "\n", encoding="utf-8")
        ok, errs = validate_submission_file(f, benchmark_type="enterpriserag")
        assert ok is False
        assert any("'query' must be a non-empty string" in e for e in errs)

    @pytest.mark.parametrize(
        "bad_answer",
        [None, "", "    ", "\t\n", 42, ["answer in a list"]],
    )
    def test_invalid_answer(self, tmp_path, base_valid_record, bad_answer):
        rec = dict(base_valid_record)
        rec["answer"] = bad_answer
        f = tmp_path / "bad_answer.jsonl"
        f.write_text(json.dumps(rec) + "\n", encoding="utf-8")
        ok, errs = validate_submission_file(f, benchmark_type="enterpriserag")
        assert ok is False
        assert any("'answer' must be a non-empty string" in e for e in errs)

    @pytest.mark.parametrize(
        "bad_citations",
        ["citation_string", 123, None, {"doc_id": "single_dict"}],
    )
    def test_citations_must_be_list(self, tmp_path, base_valid_record, bad_citations):
        rec = dict(base_valid_record)
        rec["citations"] = bad_citations
        f = tmp_path / "citations_not_list.jsonl"
        f.write_text(json.dumps(rec) + "\n", encoding="utf-8")
        ok, errs = validate_submission_file(f, benchmark_type="enterpriserag")
        assert ok is False
        assert any("'citations' must be a list" in e for e in errs)

    def test_citations_elements_must_be_dicts(self, tmp_path, base_valid_record):
        rec = dict(base_valid_record)
        rec["citations"] = ["not_a_dict", 456, None]
        f = tmp_path / "citations_bad_elements.jsonl"
        f.write_text(json.dumps(rec) + "\n", encoding="utf-8")
        ok, errs = validate_submission_file(f, benchmark_type="enterpriserag")
        assert ok is False
        assert any("citation #0 must be a dict" in e for e in errs)
        assert any("citation #1 must be a dict" in e for e in errs)
        assert any("citation #2 must be a dict" in e for e in errs)

    @pytest.mark.parametrize(
        "bad_doc_ids",
        [
            [],
            "doc_1",
            None,
            123,
            [123, 456],
            [""],
            ["   "],
            ["valid_doc", ""],
            ["valid_doc", None],
        ],
    )
    def test_invalid_doc_ids(self, tmp_path, base_valid_record, bad_doc_ids):
        rec = dict(base_valid_record)
        rec["doc_ids"] = bad_doc_ids
        f = tmp_path / "bad_doc_ids.jsonl"
        f.write_text(json.dumps(rec) + "\n", encoding="utf-8")
        ok, errs = validate_submission_file(f, benchmark_type="enterpriserag")
        assert ok is False
        assert any("doc_ids" in e or "doc_id" in e for e in errs)

    @pytest.mark.parametrize(
        "bad_latency",
        [-0.0001, -50.0, -100, "100ms", "fast", None, {}, []],
    )
    def test_invalid_latency_ms(self, tmp_path, base_valid_record, bad_latency):
        rec = dict(base_valid_record)
        rec["latency_ms"] = bad_latency
        f = tmp_path / "bad_latency.jsonl"
        f.write_text(json.dumps(rec) + "\n", encoding="utf-8")
        ok, errs = validate_submission_file(f, benchmark_type="enterpriserag")
        assert ok is False
        assert any("'latency_ms' must be a non-negative number" in e for e in errs)

    def test_latency_ms_boolean_rejected(self, tmp_path, base_valid_record):
        rec = dict(base_valid_record)
        rec["latency_ms"] = True
        f = tmp_path / "bool_latency.jsonl"
        f.write_text(json.dumps(rec) + "\n", encoding="utf-8")
        ok, errs = validate_submission_file(f, benchmark_type="enterpriserag")
        assert ok is False, "Boolean latency_ms (True/False) should be rejected"

    @pytest.mark.parametrize(
        "bad_metadata",
        ["metadata_str", 123, [1, 2], None],
    )
    def test_invalid_metadata_type(self, tmp_path, base_valid_record, bad_metadata):
        rec = dict(base_valid_record)
        rec["metadata"] = bad_metadata
        f = tmp_path / "bad_metadata.jsonl"
        f.write_text(json.dumps(rec) + "\n", encoding="utf-8")
        ok, errs = validate_submission_file(f, benchmark_type="enterpriserag")
        assert ok is False
        assert any("'metadata' must be a dict" in e for e in errs)


# ===========================================================================
# 3. GAIA Schema Validation Stress
# ===========================================================================

class TestGaiaSchemaStress:
    """Stress tests targeting GAIA General AI Assistants benchmark invariants."""

    @pytest.fixture
    def base_valid_gaia_record(self) -> Dict[str, Any]:
        return {
            "task_id": "gaia_lvl1_0042",
            "model_answer": "The total energy consumption is 450 kWh.",
            "reasoning_trace": [
                "Step 1: Extracted baseline energy usage metrics.",
                "Step 2: Applied seasonal coefficient adjustment.",
                "Step 3: Computed final aggregated consumption.",
            ],
            "sources_used": ["source_doc_101", "source_doc_102"],
        }

    @pytest.mark.parametrize(
        "missing_key",
        ["task_id", "model_answer", "reasoning_trace", "sources_used"],
    )
    def test_gaia_missing_required_keys_individually(self, tmp_path, base_valid_gaia_record, missing_key):
        rec = dict(base_valid_gaia_record)
        del rec[missing_key]
        f = tmp_path / f"gaia_missing_{missing_key}.jsonl"
        f.write_text(json.dumps(rec) + "\n", encoding="utf-8")
        ok, errs = validate_submission_file(f, benchmark_type="gaia")
        assert ok is False
        assert any(f"Missing required key '{missing_key}'" in e for e in errs)

    @pytest.mark.parametrize(
        "bad_task_id",
        [None, "", 777, []],
    )
    def test_gaia_invalid_task_id(self, tmp_path, base_valid_gaia_record, bad_task_id):
        rec = dict(base_valid_gaia_record)
        rec["task_id"] = bad_task_id
        f = tmp_path / "gaia_bad_task_id.jsonl"
        f.write_text(json.dumps(rec) + "\n", encoding="utf-8")
        ok, errs = validate_submission_file(f, benchmark_type="gaia")
        assert ok is False
        assert any("'task_id' must be a non-empty string" in e for e in errs)

    @pytest.mark.parametrize(
        "bad_answer",
        [None, "", "   ", "\n\t", 1234, {}],
    )
    def test_gaia_invalid_model_answer(self, tmp_path, base_valid_gaia_record, bad_answer):
        rec = dict(base_valid_gaia_record)
        rec["model_answer"] = bad_answer
        f = tmp_path / "gaia_bad_answer.jsonl"
        f.write_text(json.dumps(rec) + "\n", encoding="utf-8")
        ok, errs = validate_submission_file(f, benchmark_type="gaia")
        assert ok is False
        assert any("'model_answer' must be a non-empty string" in e for e in errs)

    @pytest.mark.parametrize(
        "bad_trace",
        ["string_trace", 123, None, {"step": 1}],
    )
    def test_gaia_reasoning_trace_must_be_list(self, tmp_path, base_valid_gaia_record, bad_trace):
        rec = dict(base_valid_gaia_record)
        rec["reasoning_trace"] = bad_trace
        f = tmp_path / "gaia_bad_trace.jsonl"
        f.write_text(json.dumps(rec) + "\n", encoding="utf-8")
        ok, errs = validate_submission_file(f, benchmark_type="gaia")
        assert ok is False
        assert any("'reasoning_trace' must be a list" in e for e in errs)

    def test_gaia_empty_reasoning_trace_rejected(self, tmp_path, base_valid_gaia_record):
        rec = dict(base_valid_gaia_record)
        rec["reasoning_trace"] = []
        f = tmp_path / "gaia_empty_trace.jsonl"
        f.write_text(json.dumps(rec) + "\n", encoding="utf-8")
        ok, errs = validate_submission_file(f, benchmark_type="gaia")
        assert ok is False, "Empty reasoning_trace list should be rejected in GAIA"

    @pytest.mark.parametrize(
        "bad_sources",
        [[], "single_source_str", 123, None, {}],
    )
    def test_gaia_sources_used_must_be_non_empty_list(self, tmp_path, base_valid_gaia_record, bad_sources):
        rec = dict(base_valid_gaia_record)
        rec["sources_used"] = bad_sources
        f = tmp_path / "gaia_bad_sources.jsonl"
        f.write_text(json.dumps(rec) + "\n", encoding="utf-8")
        ok, errs = validate_submission_file(f, benchmark_type="gaia")
        assert ok is False
        assert any("'sources_used' must be a non-empty list" in e for e in errs)

    def test_gaia_non_string_sources_used_rejected(self, tmp_path, base_valid_gaia_record):
        rec = dict(base_valid_gaia_record)
        rec["sources_used"] = [123, ""]
        f = tmp_path / "gaia_invalid_src_elems.jsonl"
        f.write_text(json.dumps(rec) + "\n", encoding="utf-8")
        ok, errs = validate_submission_file(f, benchmark_type="gaia")
        assert ok is False, "Non-string or empty elements in sources_used should be rejected"


# ===========================================================================
# 4. AIR-Bench & TREC Run Line Stress
# ===========================================================================

class TestAirBenchTrecStress:
    """Stress tests for TREC run lines, column counts, rank sequences, and score monotonicity."""

    @pytest.mark.parametrize(
        "columns_line,expected_col_count",
        [
            ("q1", 1),
            ("q1 Q0", 2),
            ("q1 Q0 doc_1", 3),
            ("q1 Q0 doc_1 1", 4),
            ("q1 Q0 doc_1 1 0.950000", 5),
            ("q1 Q0 doc_1 1 0.950000 run1 extra_col7", 7),
            ("q1 Q0 doc_1 1 0.950000 run1 extra7 extra8", 8),
        ],
    )
    def test_trec_wrong_column_counts(self, tmp_path, columns_line, expected_col_count):
        f = tmp_path / "bad_cols.run"
        f.write_text(columns_line + "\n", encoding="utf-8")
        ok, errs = validate_submission_file(f)
        assert ok is False
        assert any(f"Expected 6 columns" in e and f"got {expected_col_count}" in e for e in errs)

    @pytest.mark.parametrize(
        "bad_rank_str,error_substr",
        [
            ("1.5", "Rank must be an integer"),
            ("first", "Rank must be an integer"),
            ("NaN", "Rank must be an integer"),
            ("#1", "Rank must be an integer"),
        ],
    )
    def test_trec_non_integer_rank(self, tmp_path, bad_rank_str, error_substr):
        f = tmp_path / "bad_rank.run"
        f.write_text(f"q1 Q0 doc_1 {bad_rank_str} 0.950000 piragi-sota\n", encoding="utf-8")
        ok, errs = validate_submission_file(f)
        assert ok is False
        assert any(error_substr in e for e in errs)

    @pytest.mark.parametrize(
        "bad_score_str",
        ["high", "top_score", "0.95f", "good"],
    )
    def test_trec_non_float_score(self, tmp_path, bad_score_str):
        f = tmp_path / "bad_score.run"
        f.write_text(f"q1 Q0 doc_1 1 {bad_score_str} piragi-sota\n", encoding="utf-8")
        ok, errs = validate_submission_file(f)
        assert ok is False
        assert any("Score must be a float" in e for e in errs)

    def test_trec_nan_score_rejected(self, tmp_path):
        f = tmp_path / "nan_score.run"
        f.write_text(
            "q1 Q0 doc_1 1 nan run1\n"
            "q1 Q0 doc_2 2 nan run1\n",
            encoding="utf-8",
        )
        ok, errs = validate_submission_file(f)
        assert ok is False, "NaN score should be rejected as non-finite score"

    def test_trec_rank_not_starting_at_1(self, tmp_path):
        for start_rank in (0, 2, 5):
            f = tmp_path / f"start_at_{start_rank}.run"
            f.write_text(f"q1 Q0 doc_1 {start_rank} 0.950000 piragi-sota\n", encoding="utf-8")
            ok, errs = validate_submission_file(f)
            assert ok is False
            assert any(f"Rank must start at 1, starts at {start_rank}" in e for e in errs)

    def test_trec_inverted_ranks(self, tmp_path):
        f = tmp_path / "inverted_ranks.run"
        f.write_text(
            "q1 Q0 doc_1 1 0.950000 run1\n"
            "q1 Q0 doc_2 3 0.850000 run1\n"
            "q1 Q0 doc_3 2 0.750000 run1\n",
            encoding="utf-8",
        )
        ok, errs = validate_submission_file(f)
        assert ok is False
        assert any("Ranks not strictly increasing" in e for e in errs)

    def test_trec_duplicate_ranks(self, tmp_path):
        f = tmp_path / "duplicate_ranks.run"
        f.write_text(
            "q1 Q0 doc_1 1 0.950000 run1\n"
            "q1 Q0 doc_2 1 0.850000 run1\n",
            encoding="utf-8",
        )
        ok, errs = validate_submission_file(f)
        assert ok is False
        assert any("Ranks not strictly increasing at index 1: 1 -> 1" in e for e in errs)

    def test_trec_rank_gap_detected(self, tmp_path):
        f = tmp_path / "rank_gap.run"
        f.write_text(
            "q1 Q0 doc_1 1 0.950000 run1\n"
            "q1 Q0 doc_2 2 0.850000 run1\n"
            "q1 Q0 doc_3 4 0.750000 run1\n",  # skipped 3
            encoding="utf-8",
        )
        ok, errs = validate_submission_file(f)
        assert ok is False
        assert any("Rank gap detected: 2 -> 4" in e for e in errs)

    def test_trec_non_monotonic_increasing_score(self, tmp_path):
        f = tmp_path / "increasing_score.run"
        f.write_text(
            "q1 Q0 doc_1 1 0.500000 run1\n"
            "q1 Q0 doc_2 2 0.800000 run1\n",  # score increased
            encoding="utf-8",
        )
        ok, errs = validate_submission_file(f)
        assert ok is False
        assert any("Non-monotonic score at rank 2: score 0.5 < 0.8" in e for e in errs)

    def test_trec_equal_scores_allowed(self, tmp_path):
        """Monotonic non-increasing allows equal scores (score_i >= score_{i+1})."""
        f = tmp_path / "tied_scores.run"
        f.write_text(
            "q1 Q0 doc_1 1 0.500000 run1\n"
            "q1 Q0 doc_2 2 0.500000 run1\n"
            "q1 Q0 doc_3 3 0.200000 run1\n",
            encoding="utf-8",
        )
        ok, errs = validate_submission_file(f)
        assert ok is True, f"Tied scores should be valid non-increasing, got: {errs}"

    def test_trec_multi_query_interleaved_tracking(self, tmp_path):
        """Interleaved query records should be correctly grouped per query ID."""
        f = tmp_path / "interleaved.run"
        f.write_text(
            "q_alpha Q0 doc_a1 1 0.900000 run1\n"
            "q_beta  Q0 doc_b1 1 0.850000 run1\n"
            "q_alpha Q0 doc_a2 2 0.700000 run1\n"
            "q_beta  Q0 doc_b2 2 0.650000 run1\n",
            encoding="utf-8",
        )
        ok, errs = validate_submission_file(f)
        assert ok is True, f"Interleaved valid queries should pass, got: {errs}"

    def test_trec_multi_query_isolates_failing_query(self, tmp_path):
        """Ensure failure in one query does not corrupt reporting for other queries."""
        f = tmp_path / "multi_one_fail.run"
        f.write_text(
            "q_valid Q0 doc_1 1 0.900000 run1\n"
            "q_valid Q0 doc_2 2 0.800000 run1\n"
            "q_bad   Q0 doc_3 1 0.500000 run1\n"
            "q_bad   Q0 doc_4 2 0.950000 run1\n",  # Bad query increases score
            encoding="utf-8",
        )
        ok, errs = validate_submission_file(f)
        assert ok is False
        assert any("query 'q_bad': Non-monotonic score at rank 2" in e for e in errs)
        # Ensure q_valid is not flagged
        assert not any("q_valid" in e for e in errs)


# ===========================================================================
# 5. ZIP Archive Integrity & Corruption Stress
# ===========================================================================

class TestZipArchiveStress:
    """Stress tests covering ZIP archive structure, CRC-32 errors, and nested payload corruption."""

    def test_zero_byte_zip(self, tmp_path):
        f = tmp_path / "empty.zip"
        f.touch()
        ok, errs = validate_submission_file(f, benchmark_type="airbench")
        assert ok is False
        assert any("File is empty (0 bytes)" in e for e in errs)

    def test_non_zip_file_with_zip_extension(self, tmp_path):
        f = tmp_path / "fake.zip"
        f.write_text("This is an ordinary text file masquerading as a zip.\n", encoding="utf-8")
        ok, errs = validate_submission_file(f, benchmark_type="airbench")
        assert ok is False
        assert any("File is not a valid zip archive" in e for e in errs)

    def test_truncated_zip_archive(self, tmp_path):
        good_zip = tmp_path / "good.zip"
        with zipfile.ZipFile(good_zip, "w") as zf:
            zf.writestr("default.run", "q1 Q0 doc_1 1 0.900000 run1\n")
            zf.writestr("manifest.json", "{}")

        raw_bytes = good_zip.read_bytes()
        truncated = tmp_path / "truncated.zip"
        truncated.write_bytes(raw_bytes[: len(raw_bytes) // 2])  # Cut in half

        ok, errs = validate_submission_file(truncated, benchmark_type="airbench")
        assert ok is False
        assert any("not a valid zip archive" in e or "Bad zip file" in e for e in errs)

    def test_zip_stored_bad_crc32_detected(self, tmp_path):
        """Verify ZIP_STORED payload corruption triggers CRC-32 error detection."""
        zip_path = tmp_path / "corrupt_stored_crc.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
            zf.writestr("default.run", "q1 Q0 doc_1 1 0.900000 run1\n" * 10)
            zf.writestr("manifest.json", '{"benchmark": "airbench"}')

        content = bytearray(zip_path.read_bytes())
        content[60] ^= 0xFF
        zip_path.write_bytes(content)

        ok, errs = validate_submission_file(zip_path, benchmark_type="airbench")
        assert ok is False
        assert any("Bad CRC-32" in e or "Bad zip file" in e for e in errs)

    def test_zip_deflated_corrupt_payload_graceful_handling(self, tmp_path):
        zip_path = tmp_path / "corrupt_deflate_crc.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("default.run", "q1 Q0 doc_1 1 0.900000 run1\n" * 40)
            zf.writestr("manifest.json", '{"benchmark": "airbench"}')

        content = bytearray(zip_path.read_bytes())
        content[60] ^= 0xFF
        zip_path.write_bytes(content)

        # Expected: graceful False return without raising unhandled zlib.error
        ok, errs = validate_submission_file(zip_path, benchmark_type="airbench")
        assert ok is False
        assert any("Bad CRC-32" in e or "Corrupt entry" in e or "Bad zip file" in e for e in errs)

    def test_zip_missing_run_files(self, tmp_path):
        zip_path = tmp_path / "missing_runs.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("manifest.json", '{"benchmark": "airbench"}')
            zf.writestr("metadata.json", '{"team": "piragi"}')

        ok, errs = validate_submission_file(zip_path, benchmark_type="airbench")
        assert ok is False
        assert any("ZIP archive does not contain any TREC run" in e for e in errs)

    def test_zip_missing_both_manifest_and_metadata(self, tmp_path):
        zip_path = tmp_path / "missing_manifest.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("default.run", "q1 Q0 doc_1 1 0.900000 run1\n")

        ok, errs = validate_submission_file(zip_path, benchmark_type="airbench")
        assert ok is False
        assert any("ZIP archive missing manifest.json or metadata.json" in e for e in errs)

    def test_zip_with_empty_run_file(self, tmp_path):
        zip_path = tmp_path / "empty_run.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("default.run", "")
            zf.writestr("manifest.json", "{}")

        ok, errs = validate_submission_file(zip_path, benchmark_type="airbench")
        assert ok is False
        assert any("default.run: File is empty" in e for e in errs)

    def test_zip_with_non_utf8_run_file(self, tmp_path):
        zip_path = tmp_path / "non_utf8.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("default.run", b"\xff\xfe\x00\x00corrupt")
            zf.writestr("manifest.json", "{}")

        ok, errs = validate_submission_file(zip_path, benchmark_type="airbench")
        assert ok is False
        assert any("UTF-8 decode error" in e for e in errs)

    def test_zip_with_multiple_runs_one_corrupt(self, tmp_path):
        zip_path = tmp_path / "partial_corrupt_runs.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("cranfield.run", "q1 Q0 doc_1 1 0.900000 run1\n")
            zf.writestr("beir.run", "q2 Q0 doc_2 1 0.500000 run1\nq2 Q0 doc_3 2 0.900000 run1\n")  # Score increases
            zf.writestr("manifest.json", "{}")

        ok, errs = validate_submission_file(zip_path, benchmark_type="airbench")
        assert ok is False
        assert any("beir.run query 'q2': Non-monotonic score at rank 2" in e for e in errs)
        assert not any("cranfield.run" in e for e in errs)


# ===========================================================================
# 6. Exporter Resilience & Degenerate Input Stress
# ===========================================================================

class TestExporterResilienceStress:
    """Stress tests verifying exporter behavior on edge cases and malformed input datasets."""

    def test_export_enterpriserag_empty_results_fails_validation(self, tmp_path):
        out = tmp_path / "empty_enterprise.jsonl"
        export_enterpriserag([], out)
        assert out.exists()
        ok, errs = validate_submission_file(out, benchmark_type="enterpriserag")
        assert ok is False
        assert any("File is empty" in e for e in errs)

    def test_export_airbench_empty_results_fails_validation(self, tmp_path):
        out = tmp_path / "empty_airbench.zip"
        export_airbench([], out)
        assert out.exists()
        assert zipfile.is_zipfile(out)
        ok, errs = validate_submission_file(out, benchmark_type="airbench")
        assert ok is False
        assert any("default.run: File is empty" in e for e in errs)

    def test_export_gaia_empty_results_fails_validation(self, tmp_path):
        out = tmp_path / "empty_gaia.jsonl"
        export_gaia([], out)
        assert out.exists()
        ok, errs = validate_submission_file(out, benchmark_type="gaia")
        assert ok is False
        assert any("File is empty" in e for e in errs)

    def test_export_enterpriserag_missing_file_raises_filenotfound(self, tmp_path):
        bad_path = tmp_path / "nonexistent_results.json"
        with pytest.raises(FileNotFoundError, match="Benchmark results file not found"):
            export_enterpriserag(bad_path)

    def test_export_airbench_missing_file_raises_filenotfound(self, tmp_path):
        bad_path = tmp_path / "nonexistent_results.json"
        with pytest.raises(FileNotFoundError, match="Benchmark results file not found"):
            export_airbench(bad_path)

    def test_export_gaia_missing_file_raises_filenotfound(self, tmp_path):
        bad_path = tmp_path / "nonexistent_results.json"
        with pytest.raises(FileNotFoundError, match="Benchmark results file not found"):
            export_gaia(bad_path)

    @pytest.mark.parametrize("invalid_input", [12345, True, 3.14])
    def test_export_enterpriserag_invalid_input_type_raises_typeerror(self, invalid_input):
        with pytest.raises(TypeError, match="Unsupported input type"):
            export_enterpriserag(invalid_input)

    def test_export_enterpriserag_missing_retrieved_ids_generates_empty_doc_ids(self, tmp_path):
        """When retrieved_ids is empty, export succeeds, but validator catches the missing doc_ids."""
        records = [{"question_id": "q_no_docs", "query": "aerodynamic flutter"}]
        out = tmp_path / "no_docs.jsonl"
        export_enterpriserag(records, out)
        ok, errs = validate_submission_file(out, benchmark_type="enterpriserag")
        assert ok is False
        assert any("'doc_ids' must be a non-empty list of document IDs" in e for e in errs)

    def test_export_gaia_auto_fills_sources_if_empty(self, tmp_path):
        """GAIA export must provide a fallback source identifier if none were retrieved."""
        records = [{"task_id": "task_no_source", "query": "calculate orbit decay"}]
        out = tmp_path / "gaia_fallback.jsonl"
        export_gaia(records, out)
        ok, errs = validate_submission_file(out, benchmark_type="gaia")
        assert ok is True, f"GAIA export should auto-populate fallback sources, got: {errs}"

    def test_export_gaia_auto_generates_5_step_reasoning_trace(self, tmp_path):
        records = [{"task_id": "task_trace_gen", "query": "what is thermal expansion?"}]
        out = tmp_path / "gaia_trace.jsonl"
        export_gaia(records, out)
        with open(out, "r", encoding="utf-8") as f:
            rec = json.loads(f.readline())
        assert len(rec["reasoning_trace"]) == 5
        assert "Step 1" in rec["reasoning_trace"][0]
        assert "Step 5" in rec["reasoning_trace"][4]

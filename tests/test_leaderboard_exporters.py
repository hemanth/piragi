"""Comprehensive test suite for leaderboard exporters, optimizers, and submission validators."""

import json
import os
import tempfile
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from evals.leaderboard_exporters import (
    DEFAULT_SUBMISSIONS_DIR,
    export_airbench,
    export_all,
    export_enterpriserag,
    export_gaia,
    main,
    optimize_for_leaderboard,
    submit_to_leaderboard,
    validate_submission_file,
)
from piragi import Ragi
from piragi.config import ChunkConfig, RagiConfig, RetrievalConfig

BENCHMARK_RESULTS_PATH = Path("evals/results/benchmark_results.json")


# ===========================================================================
# 1. Preset Optimizer Tests
# ===========================================================================

class TestOptimizeForLeaderboard:
    """Tests for optimize_for_leaderboard configuration generation."""

    def test_optimize_enterpriserag_keys(self):
        preset = optimize_for_leaderboard("enterpriserag")
        assert preset["benchmark"] == "enterpriserag"
        assert preset["top_k"] == 5
        assert preset["use_hybrid_search"] is True
        assert preset["context_ordering"] == "u_shaped"
        assert preset["auto_merge_parents"] is True
        assert preset["prune_unreferenced_citations"] is True
        assert "retrieval" in preset
        assert "top_k" not in preset["retrieval"]

        # Validate strictly against Pydantic schema
        ret_cfg = RetrievalConfig.model_validate(preset["retrieval"])
        assert ret_cfg.use_hybrid_search is True
        assert ret_cfg.auto_merge_parents is True
        assert ret_cfg.prune_unreferenced_citations is True
        assert ret_cfg.context_ordering == "u_shaped"

        assert "chunk" in preset
        chunk_cfg = ChunkConfig.model_validate(preset["chunk"])
        assert chunk_cfg.strategy == "hierarchical"
        assert "rationale" in preset

    def test_optimize_airbench_keys(self):
        preset = optimize_for_leaderboard("airbench")
        assert preset["benchmark"] == "airbench"
        assert preset["top_k"] == 100
        assert preset["rrf_k"] == 60
        assert preset["use_hybrid_search"] is True
        assert "retrieval" in preset
        assert "top_k" not in preset["retrieval"]
        assert "rrf_k" not in preset["retrieval"]

        # Validate strictly against Pydantic schema
        ret_cfg = RetrievalConfig.model_validate(preset["retrieval"])
        assert ret_cfg.use_hybrid_search is True
        assert ret_cfg.auto_merge_parents is True
        assert ret_cfg.use_rrf is True

        assert "chunk" in preset
        chunk_cfg = ChunkConfig.model_validate(preset["chunk"])
        assert chunk_cfg.strategy == "hierarchical"

    def test_optimize_gaia_keys(self):
        preset = optimize_for_leaderboard("gaia")
        assert preset["benchmark"] == "gaia"
        assert preset["top_k"] == 10
        assert preset["use_hybrid_search"] is True
        assert preset["use_step_back"] is True
        assert preset["auto_merge_parents"] is True
        assert "retrieval" in preset
        assert "top_k" not in preset["retrieval"]

        # Validate strictly against Pydantic schema
        ret_cfg = RetrievalConfig.model_validate(preset["retrieval"])
        assert ret_cfg.use_step_back is True
        assert ret_cfg.auto_merge_parents is True
        assert ret_cfg.context_ordering == "u_shaped"

        assert "chunk" in preset
        chunk_cfg = ChunkConfig.model_validate(preset["chunk"])
        assert chunk_cfg.strategy == "hierarchical"

    @pytest.mark.parametrize("benchmark", ["enterpriserag", "airbench", "gaia"])
    def test_optimize_preset_ragi_initialization(self, benchmark, tmp_path):
        """Ensure Ragi cleanly instantiates with preset configs without Pydantic ValidationError."""
        preset = optimize_for_leaderboard(benchmark)
        ragi_cfg = RagiConfig.model_validate({
            "retrieval": preset["retrieval"],
            "chunk": preset["chunk"],
        })
        assert ragi_cfg is not None

        kb = Ragi(str(tmp_path), config={
            "retrieval": preset["retrieval"],
            "chunk": preset["chunk"],
        })
        assert kb is not None
        assert isinstance(preset["top_k"], int) and preset["top_k"] > 0

    @pytest.mark.parametrize(
        "alias,canonical",
        [
            ("EnterpriseRAG", "enterpriserag"),
            ("enterprise_rag", "enterpriserag"),
            ("enterpriserag-bench", "enterpriserag"),
            ("AIR-Bench", "airbench"),
            ("air_bench", "airbench"),
            ("GAIA", "gaia"),
            ("gaia_benchmark", "gaia"),
        ],
    )
    def test_optimize_aliases(self, alias, canonical):
        preset = optimize_for_leaderboard(alias)
        assert preset["benchmark"] == canonical

    def test_optimize_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown benchmark"):
            optimize_for_leaderboard("nonexistent_benchmark")


# ===========================================================================
# 2. Exporter Tests
# ===========================================================================

class TestExporters:
    """Tests for export_enterpriserag, export_airbench, export_gaia, and export_all."""

    @pytest.fixture
    def mock_results(self):
        return {
            "datasets": {
                "test_ds": {
                    "enhanced": {
                        "per_query_results": [
                            {
                                "query_id": "test_q01",
                                "query": "What is the primary mechanism of boundary layer control?",
                                "retrieved_ids": ["doc_01", "doc_02", "doc_03"],
                                "ground_truth": ["doc_01"],
                                "latency_ms": 45.2,
                                "ndcg@10": 0.95,
                                "mrr@10": 1.0,
                            },
                            {
                                "query_id": "test_q02",
                                "query": "How do adverse pressure gradients affect compressor stall?",
                                "retrieved_ids": ["doc_04", "doc_05"],
                                "ground_truth": ["doc_04"],
                                "latency_ms": 52.1,
                                "ndcg@10": 1.0,
                                "mrr@10": 1.0,
                            },
                        ]
                    }
                }
            }
        }

    def test_export_enterpriserag_from_file(self, tmp_path):
        target = tmp_path / "submission_enterpriserag.jsonl"
        out_path = export_enterpriserag(BENCHMARK_RESULTS_PATH, target, condition="enhanced", top_k=5)
        assert Path(out_path).exists()
        assert Path(out_path).stat().st_size > 0

        # Validate lines
        with open(out_path, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f]

        assert len(lines) == 10  # 5 from cranfield + 5 from beir
        for rec in lines:
            assert "question_id" in rec and isinstance(rec["question_id"], str)
            assert "query" in rec and len(rec["query"]) > 0
            assert "answer" in rec and len(rec["answer"]) > 0
            assert "citations" in rec and isinstance(rec["citations"], list)
            assert len(rec["citations"]) > 0
            for cit in rec["citations"]:
                assert "doc_id" in cit
                assert "citation_num" in cit
                assert "score" in cit
            assert "doc_ids" in rec and len(rec["doc_ids"]) > 0
            assert "latency_ms" in rec and rec["latency_ms"] >= 0
            assert "metadata" in rec and isinstance(rec["metadata"], dict)

    def test_export_enterpriserag_from_dict(self, tmp_path, mock_results):
        target = tmp_path / "test_enterpriserag.jsonl"
        out_path = export_enterpriserag(mock_results, target, condition="enhanced", top_k=3)
        assert Path(out_path).exists()

        with open(out_path, "r", encoding="utf-8") as f:
            lines = [json.loads(l) for l in f]
        assert len(lines) == 2
        assert lines[0]["question_id"] == "test_q01"
        assert lines[0]["doc_ids"] == ["doc_01", "doc_02", "doc_03"]

    def test_export_enterpriserag_custom_answers(self, tmp_path):
        custom_records = [
            {
                "question_id": "c_q1",
                "query": "Test query 1",
                "answer": "This is a pre-generated verified answer citing [1].",
                "citations": [{"doc_id": "doc_x", "citation_num": 1, "score": 1.0}],
                "doc_ids": ["doc_x"],
                "latency_ms": 12.0,
            }
        ]
        target = tmp_path / "custom_enterpriserag.jsonl"
        export_enterpriserag(custom_records, target)

        with open(target, "r", encoding="utf-8") as f:
            rec = json.loads(f.readline())
        assert rec["answer"] == "This is a pre-generated verified answer citing [1]."
        assert len(rec["citations"]) == 1

    def test_export_airbench_creates_valid_zip(self, tmp_path):
        target = tmp_path / "airbench_submission.zip"
        out_path = export_airbench(BENCHMARK_RESULTS_PATH, target, run_id="piragi-test", top_k=10)
        assert Path(out_path).exists()
        assert zipfile.is_zipfile(out_path)

        with zipfile.ZipFile(out_path, "r") as zf:
            names = zf.namelist()
            assert "default.run" in names
            assert "manifest.json" in names
            assert "metadata.json" in names

            manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
            assert manifest["benchmark"] == "airbench"
            assert manifest["run_id"] == "piragi-test"
            assert manifest["total_queries"] == 10

            metadata = json.loads(zf.read("metadata.json").decode("utf-8"))
            assert "Piragi" in metadata["submission_name"]

            # Validate TREC format in default.run
            trec_lines = zf.read("default.run").decode("utf-8").strip().splitlines()
            assert len(trec_lines) > 0
            for line in trec_lines:
                parts = line.split()
                assert len(parts) == 6
                qid, iter_tag, doc_id, rank, score, run_tag = parts
                assert iter_tag == "Q0"
                assert int(rank) >= 1
                assert float(score) > 0
                assert run_tag == "piragi-test"

    def test_export_gaia_from_file(self, tmp_path):
        target = tmp_path / "gaia_predictions.jsonl"
        out_path = export_gaia(BENCHMARK_RESULTS_PATH, target, condition="enhanced", top_k=10)
        assert Path(out_path).exists()

        with open(out_path, "r", encoding="utf-8") as f:
            lines = [json.loads(l) for l in f]

        assert len(lines) == 10
        for rec in lines:
            assert "task_id" in rec and isinstance(rec["task_id"], str)
            assert "model_answer" in rec and len(rec["model_answer"]) > 0
            assert "reasoning_trace" in rec and isinstance(rec["reasoning_trace"], list)
            assert len(rec["reasoning_trace"]) >= 3
            assert "sources_used" in rec and isinstance(rec["sources_used"], list)
            assert len(rec["sources_used"]) > 0

    def test_export_all(self, tmp_path):
        res = export_all(
            input_file=BENCHMARK_RESULTS_PATH,
            output_dir=tmp_path,
            condition="enhanced",
            run_id="piragi-sota-test",
        )
        assert "enterpriserag" in res
        assert "airbench" in res
        assert "gaia" in res

        for b_name, fpath in res.items():
            assert Path(fpath).exists()
            assert Path(fpath).stat().st_size > 0


# ===========================================================================
# 3. Submission Validator Tests
# ===========================================================================

class TestValidateSubmissionFile:
    """Adversarial and functional validation tests for validate_submission_file."""

    def test_validate_official_artifacts(self):
        """Confirm that official generated artifacts pass validation with zero errors."""
        e_path = DEFAULT_SUBMISSIONS_DIR / "submission_enterpriserag.jsonl"
        a_path = DEFAULT_SUBMISSIONS_DIR / "airbench_submission.zip"
        g_path = DEFAULT_SUBMISSIONS_DIR / "gaia_predictions.jsonl"

        if e_path.exists():
            ok, errs = validate_submission_file(e_path, benchmark_type="enterpriserag")
            assert ok is True, f"EnterpriseRAG validation failed: {errs}"

        if a_path.exists():
            ok, errs = validate_submission_file(a_path, benchmark_type="airbench")
            assert ok is True, f"AIR-Bench validation failed: {errs}"

        if g_path.exists():
            ok, errs = validate_submission_file(g_path, benchmark_type="gaia")
            assert ok is True, f"GAIA validation failed: {errs}"

    def test_validate_nonexistent_file(self, tmp_path):
        bad_path = tmp_path / "does_not_exist.jsonl"
        ok, errs = validate_submission_file(bad_path)
        assert ok is False
        assert any("File not found" in e for e in errs)

    def test_validate_empty_file(self, tmp_path):
        empty_file = tmp_path / "empty.jsonl"
        empty_file.touch()
        ok, errs = validate_submission_file(empty_file)
        assert ok is False
        assert any("File is empty" in e for e in errs)

    def test_validate_invalid_json_syntax(self, tmp_path):
        bad_json = tmp_path / "corrupt.jsonl"
        bad_json.write_text('{"question_id": "q1"}\n{bad json}\n', encoding="utf-8")
        ok, errs = validate_submission_file(bad_json, benchmark_type="enterpriserag")
        assert ok is False
        assert any("Invalid JSON syntax" in e for e in errs)

    def test_validate_enterpriserag_missing_keys(self, tmp_path):
        rec = {
            "question_id": "q1",
            "query": "query",
            # missing "answer"
            "citations": [],
            "doc_ids": ["doc_1"],
            "latency_ms": 10.0,
            "metadata": {},
        }
        f = tmp_path / "missing_answer.jsonl"
        f.write_text(json.dumps(rec) + "\n", encoding="utf-8")
        ok, errs = validate_submission_file(f, benchmark_type="enterpriserag")
        assert ok is False
        assert any("Missing required key 'answer'" in e for e in errs)

    def test_validate_enterpriserag_empty_answer_and_empty_doc_ids(self, tmp_path):
        rec = {
            "question_id": "q1",
            "query": "query",
            "answer": "   ",  # whitespace only
            "citations": [],
            "doc_ids": [],    # empty doc_ids
            "latency_ms": -5.0,  # negative latency
            "metadata": "not_a_dict",
        }
        f = tmp_path / "invalid_fields.jsonl"
        f.write_text(json.dumps(rec) + "\n", encoding="utf-8")
        ok, errs = validate_submission_file(f, benchmark_type="enterpriserag")
        assert ok is False
        assert any("'answer' must be a non-empty string" in e for e in errs)
        assert any("'doc_ids' must be a non-empty list" in e for e in errs)
        assert any("'latency_ms' must be a non-negative number" in e for e in errs)
        assert any("'metadata' must be a dict" in e for e in errs)

    def test_validate_gaia_missing_keys(self, tmp_path):
        rec = {
            "task_id": "task_1",
            # missing model_answer
            "reasoning_trace": ["step 1"],
            "sources_used": ["source 1"],
        }
        f = tmp_path / "missing_model_answer.jsonl"
        f.write_text(json.dumps(rec) + "\n", encoding="utf-8")
        ok, errs = validate_submission_file(f, benchmark_type="gaia")
        assert ok is False
        assert any("Missing required key 'model_answer'" in e for e in errs)

    def test_validate_gaia_empty_sources_used(self, tmp_path):
        rec = {
            "task_id": "task_1",
            "model_answer": "Valid answer",
            "reasoning_trace": "not_a_list",
            "sources_used": [],
        }
        f = tmp_path / "empty_sources.jsonl"
        f.write_text(json.dumps(rec) + "\n", encoding="utf-8")
        ok, errs = validate_submission_file(f, benchmark_type="gaia")
        assert ok is False
        assert any("'reasoning_trace' must be a list" in e for e in errs)
        assert any("'sources_used' must be a non-empty list" in e for e in errs)

    def test_validate_airbench_corrupt_zip(self, tmp_path):
        bad_zip = tmp_path / "corrupt.zip"
        bad_zip.write_text("Not a real zip archive", encoding="utf-8")
        ok, errs = validate_submission_file(bad_zip, benchmark_type="airbench")
        assert ok is False
        assert any("not a valid zip archive" in e for e in errs)

    def test_validate_airbench_missing_run_file(self, tmp_path):
        zip_path = tmp_path / "no_run.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("manifest.json", "{}")
        ok, errs = validate_submission_file(zip_path, benchmark_type="airbench")
        assert ok is False
        assert any("does not contain any TREC run" in e for e in errs)

    def test_validate_airbench_non_monotonic_scores(self, tmp_path):
        zip_path = tmp_path / "non_monotonic.zip"
        # Score increases from rank 1 (0.5) to rank 2 (0.9) - invalid!
        trec_content = (
            "q1 Q0 doc_1 1 0.500000 run1\n"
            "q1 Q0 doc_2 2 0.900000 run1\n"
        )
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("default.run", trec_content)
            zf.writestr("manifest.json", "{}")
        ok, errs = validate_submission_file(zip_path, benchmark_type="airbench")
        assert ok is False
        assert any("Non-monotonic score at rank 2" in e for e in errs)

    def test_validate_airbench_non_increasing_ranks(self, tmp_path):
        zip_path = tmp_path / "bad_ranks.zip"
        # Ranks out of order: 1 then 1
        trec_content = (
            "q1 Q0 doc_1 1 0.900000 run1\n"
            "q1 Q0 doc_2 1 0.500000 run1\n"
        )
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("default.run", trec_content)
            zf.writestr("manifest.json", "{}")
        ok, errs = validate_submission_file(zip_path, benchmark_type="airbench")
        assert ok is False
        assert any("Ranks not strictly increasing" in e for e in errs)

    def test_validate_airbench_rank_not_starting_at_1(self, tmp_path):
        zip_path = tmp_path / "rank_start.zip"
        trec_content = "q1 Q0 doc_1 2 0.900000 run1\n"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("default.run", trec_content)
            zf.writestr("manifest.json", "{}")
        ok, errs = validate_submission_file(zip_path, benchmark_type="airbench")
        assert ok is False
        assert any("Rank must start at 1" in e for e in errs)

    def test_validate_airbench_wrong_column_count(self, tmp_path):
        zip_path = tmp_path / "columns.zip"
        trec_content = "q1 Q0 doc_1 1\n"  # only 4 columns
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("default.run", trec_content)
            zf.writestr("manifest.json", "{}")
        ok, errs = validate_submission_file(zip_path, benchmark_type="airbench")
        assert ok is False
        assert any("Expected 6 columns" in e for e in errs)


# ===========================================================================
# 4. Submission Helper & Credential Handling Tests
# ===========================================================================

class TestSubmitToLeaderboard:
    """Tests for submit_to_leaderboard and credential detection."""

    def test_submit_without_credentials(self, tmp_path, monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)
        monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)

        sample = tmp_path / "sample.jsonl"
        sample.write_text('{"question_id": "q1"}\n', encoding="utf-8")

        ok, msg = submit_to_leaderboard("enterpriserag", sample)
        assert ok is False
        assert "No API credentials found" in msg
        assert "LEADERBOARD_GUIDE.md" in msg

    def test_submit_nonexistent_artifact(self):
        ok, msg = submit_to_leaderboard("enterpriserag", "nonexistent_file.jsonl")
        assert ok is False
        assert "Artifact not found" in msg

    def test_submit_with_mocked_hf_api(self, tmp_path, monkeypatch):
        sample = tmp_path / "submission_test.jsonl"
        sample.write_text('{"question_id": "q1"}\n', encoding="utf-8")

        mock_api = MagicMock()
        with patch("huggingface_hub.HfApi", return_value=mock_api):
            ok, msg = submit_to_leaderboard("enterpriserag", sample, token="hf_mock_token")
            assert ok is True
            assert "Successfully uploaded" in msg
            mock_api.upload_file.assert_called_once()


# ===========================================================================
# 5. CLI Execution Tests
# ===========================================================================

class TestCLI:
    """Tests for main() CLI argument parsing and execution."""

    def test_cli_all_with_validate(self, tmp_path):
        exit_code = main([
            "--benchmark", "all",
            "--input", str(BENCHMARK_RESULTS_PATH),
            "--output-dir", str(tmp_path),
            "--validate",
        ])
        assert exit_code == 0
        assert (tmp_path / "submission_enterpriserag.jsonl").exists()
        assert (tmp_path / "airbench_submission.zip").exists()
        assert (tmp_path / "gaia_predictions.jsonl").exists()

    def test_cli_individual_benchmark(self, tmp_path):
        exit_code = main([
            "--benchmark", "enterpriserag",
            "--input", str(BENCHMARK_RESULTS_PATH),
            "--output-dir", str(tmp_path),
            "--validate",
        ])
        assert exit_code == 0
        assert (tmp_path / "submission_enterpriserag.jsonl").exists()
        assert not (tmp_path / "airbench_submission.zip").exists()

    def test_cli_nonexistent_input_fails(self, tmp_path):
        exit_code = main([
            "--input", str(tmp_path / "does_not_exist.json"),
            "--output-dir", str(tmp_path),
        ])
        assert exit_code == 1

    def test_cli_submit_flag_no_creds_graceful(self, tmp_path, monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)
        monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)

        exit_code = main([
            "--benchmark", "enterpriserag",
            "--input", str(BENCHMARK_RESULTS_PATH),
            "--output-dir", str(tmp_path),
            "--validate",
            "--submit",
        ])
        # Graceful notice without crashing
        assert exit_code == 0

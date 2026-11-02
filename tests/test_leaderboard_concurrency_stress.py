"""Empirical stress test harness for leaderboard packaging, CLI variations,

filesystem boundaries, byte-level zip verification, concurrency, and credential safety.
Milestone 4 - Challenger 2 Test Suite.
"""

from __future__ import annotations

import concurrent.futures
import io
import json
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List
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

BENCHMARK_RESULTS_PATH = Path("evals/results/benchmark_results.json")


# ===========================================================================
# 1. CLI Interface & Argument Variations
# ===========================================================================

class TestCLIArgumentVariations:
    """Stress-test CLI across all benchmark options, conditions, and flags."""

    def test_cli_benchmark_enterpriserag_isolated(self, tmp_path: Path):
        out_dir = tmp_path / "cli_enterpriserag"
        ret = main([
            "--benchmark", "enterpriserag",
            "--input", str(BENCHMARK_RESULTS_PATH),
            "--output-dir", str(out_dir),
            "--validate",
        ])
        assert ret == 0, f"CLI exited with non-zero code {ret}"
        erag_file = out_dir / "submission_enterpriserag.jsonl"
        assert erag_file.exists(), "submission_enterpriserag.jsonl was not created"
        assert not (out_dir / "airbench_submission.zip").exists(), "airbench zip unexpectedly created"
        assert not (out_dir / "gaia_predictions.jsonl").exists(), "gaia jsonl unexpectedly created"
        
        valid, errors = validate_submission_file(erag_file, benchmark_type="enterpriserag")
        assert valid is True, f"Validation failed: {errors}"

    def test_cli_benchmark_airbench_isolated(self, tmp_path: Path):
        out_dir = tmp_path / "cli_airbench"
        ret = main([
            "--benchmark", "airbench",
            "--input", str(BENCHMARK_RESULTS_PATH),
            "--output-dir", str(out_dir),
            "--validate",
        ])
        assert ret == 0, f"CLI exited with non-zero code {ret}"
        zip_file = out_dir / "airbench_submission.zip"
        assert zip_file.exists(), "airbench_submission.zip was not created"
        assert not (out_dir / "submission_enterpriserag.jsonl").exists(), "enterpriserag unexpectedly created"
        assert not (out_dir / "gaia_predictions.jsonl").exists(), "gaia jsonl unexpectedly created"

        valid, errors = validate_submission_file(zip_file, benchmark_type="airbench")
        assert valid is True, f"Validation failed: {errors}"

    def test_cli_benchmark_gaia_isolated(self, tmp_path: Path):
        out_dir = tmp_path / "cli_gaia"
        ret = main([
            "--benchmark", "gaia",
            "--input", str(BENCHMARK_RESULTS_PATH),
            "--output-dir", str(out_dir),
            "--validate",
        ])
        assert ret == 0, f"CLI exited with non-zero code {ret}"
        gaia_file = out_dir / "gaia_predictions.jsonl"
        assert gaia_file.exists(), "gaia_predictions.jsonl was not created"
        assert not (out_dir / "submission_enterpriserag.jsonl").exists(), "enterpriserag unexpectedly created"
        assert not (out_dir / "airbench_submission.zip").exists(), "airbench zip unexpectedly created"

        valid, errors = validate_submission_file(gaia_file, benchmark_type="gaia")
        assert valid is True, f"Validation failed: {errors}"

    def test_cli_benchmark_all(self, tmp_path: Path):
        out_dir = tmp_path / "cli_all"
        ret = main([
            "--benchmark", "all",
            "--input", str(BENCHMARK_RESULTS_PATH),
            "--output-dir", str(out_dir),
            "--validate",
        ])
        assert ret == 0, f"CLI exited with non-zero code {ret}"
        for fname in ["submission_enterpriserag.jsonl", "airbench_submission.zip", "gaia_predictions.jsonl"]:
            assert (out_dir / fname).exists(), f"File {fname} missing from --benchmark all output"

    def test_cli_condition_switch_baseline_vs_enhanced(self, tmp_path: Path):
        out_enhanced = tmp_path / "enhanced"
        out_baseline = tmp_path / "baseline"

        ret_enh = main([
            "--benchmark", "all",
            "--condition", "enhanced",
            "--output-dir", str(out_enhanced),
            "--validate",
        ])
        assert ret_enh == 0

        ret_base = main([
            "--benchmark", "all",
            "--condition", "baseline",
            "--output-dir", str(out_baseline),
            "--validate",
        ])
        assert ret_base == 0

        # Check manifest.json inside airbench zip reflects condition
        with zipfile.ZipFile(out_enhanced / "airbench_submission.zip", "r") as zf:
            man_enh = json.loads(zf.read("manifest.json").decode("utf-8"))
            assert man_enh["condition"] == "enhanced"

        with zipfile.ZipFile(out_baseline / "airbench_submission.zip", "r") as zf:
            man_base = json.loads(zf.read("manifest.json").decode("utf-8"))
            assert man_base["condition"] == "baseline"

    def test_cli_custom_run_id_propagation(self, tmp_path: Path):
        out_dir = tmp_path / "cli_run_id"
        custom_id = "test-custom-run-piragi-42"
        ret = main([
            "--benchmark", "airbench",
            "--run-id", custom_id,
            "--output-dir", str(out_dir),
        ])
        assert ret == 0

        zip_path = out_dir / "airbench_submission.zip"
        with zipfile.ZipFile(zip_path, "r") as zf:
            manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
            assert manifest["run_id"] == custom_id

            metadata = json.loads(zf.read("metadata.json").decode("utf-8"))
            assert metadata["run_id"] == custom_id

            lines = zf.read("default.run").decode("utf-8").strip().splitlines()
            for line in lines:
                parts = line.split()
                assert parts[-1] == custom_id, f"Line does not end with custom run_id: {line}"

    def test_cli_invalid_benchmark_choice(self):
        with pytest.raises(SystemExit):
            main(["--benchmark", "invalid_name"])


# ===========================================================================
# 2. Filesystem Boundaries & Input Error Handling
# ===========================================================================

class TestFilesystemBoundariesAndInputErrors:
    """Stress-test filesystem boundary conditions and error resilience."""

    def test_cli_missing_input_file_returns_code_1(self, tmp_path: Path, capsys):
        bad_path = tmp_path / "does_not_exist.json"
        ret = main(["--input", str(bad_path)])
        assert ret == 1
        captured = capsys.readouterr()
        assert "Error: Input file does not exist" in captured.err

    def test_direct_exporters_raise_filenotfound_on_missing_input(self, tmp_path: Path):
        bad_path = tmp_path / "nonexistent.json"
        with pytest.raises(FileNotFoundError):
            export_enterpriserag(bad_path, tmp_path / "out.jsonl")
        with pytest.raises(FileNotFoundError):
            export_airbench(bad_path, tmp_path / "out.zip")
        with pytest.raises(FileNotFoundError):
            export_gaia(bad_path, tmp_path / "out.jsonl")
        with pytest.raises(FileNotFoundError):
            export_all(input_file=bad_path, output_dir=tmp_path)

    def test_corrupt_json_input_raises_decode_error(self, tmp_path: Path):
        corrupt_file = tmp_path / "corrupt.json"
        corrupt_file.write_text("{ unclosed json: [1, 2, }", encoding="utf-8")

        with pytest.raises(json.JSONDecodeError):
            export_enterpriserag(corrupt_file, tmp_path / "out.jsonl")

    def test_auto_creation_of_deeply_nested_output_directory(self, tmp_path: Path):
        deep_dir = tmp_path / "level1" / "level2" / "level3" / "deep_submissions"
        assert not deep_dir.exists()

        res = export_all(output_dir=deep_dir)
        assert deep_dir.exists()
        assert Path(res["enterpriserag"]).exists()
        assert Path(res["airbench"]).exists()
        assert Path(res["gaia"]).exists()

    def test_safe_overwrite_of_existing_files(self, tmp_path: Path):
        out_dir = tmp_path / "overwrite_test"
        out_dir.mkdir(parents=True)

        target = out_dir / "submission_enterpriserag.jsonl"
        target.write_text("OLD STALE DATA\n", encoding="utf-8")

        export_enterpriserag(BENCHMARK_RESULTS_PATH, target)
        content = target.read_text(encoding="utf-8")
        assert "OLD STALE DATA" not in content
        valid, errors = validate_submission_file(target, benchmark_type="enterpriserag")
        assert valid is True, f"Overwritten file invalid: {errors}"

    def test_paths_with_spaces_and_special_characters(self, tmp_path: Path):
        special_dir = tmp_path / "dir with spaces #1" / "sub_dir (v2.0)"
        res = export_all(output_dir=special_dir)
        for b_name, fpath in res.items():
            assert Path(fpath).exists()
            valid, errors = validate_submission_file(fpath, benchmark_type=b_name)
            assert valid is True, f"{b_name} failed in special path: {errors}"


# ===========================================================================
# 3. Zip Extraction and Byte-Level Inspection of airbench_submission.zip
# ===========================================================================

class TestAirbenchZipByteLevelInspection:
    """Byte-level inspection, extraction, and format validation of AIR-Bench archive."""

    @pytest.fixture
    def fresh_airbench_zip(self, tmp_path: Path) -> Path:
        zip_path = tmp_path / "airbench_submission.zip"
        export_airbench(BENCHMARK_RESULTS_PATH, zip_path, run_id="piragi-stress-test")
        return zip_path

    def test_zip_integrity_and_decompression(self, fresh_airbench_zip: Path, tmp_path: Path):
        extract_dir = tmp_path / "extracted_zip"
        extract_dir.mkdir()

        with zipfile.ZipFile(fresh_airbench_zip, "r") as zf:
            corrupt = zf.testzip()
            assert corrupt is None, f"Corrupted file in zip: {corrupt}"

            # Verify compression method
            for info in zf.infolist():
                assert info.compress_type == zipfile.ZIP_DEFLATED, f"{info.filename} not deflated"

            zf.extractall(extract_dir)

        expected_files = {"default.run", "cranfield.run", "beir.run", "manifest.json", "metadata.json"}
        extracted_names = {p.name for p in extract_dir.iterdir()}
        assert expected_files.issubset(extracted_names), f"Missing files in zip: {expected_files - extracted_names}"

    def test_manifest_and_metadata_json_schema(self, fresh_airbench_zip: Path):
        with zipfile.ZipFile(fresh_airbench_zip, "r") as zf:
            manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
            metadata = json.loads(zf.read("metadata.json").decode("utf-8"))

        # Verify Manifest Schema
        assert manifest["benchmark"] == "airbench"
        assert "version" in manifest
        assert manifest["run_id"] == "piragi-stress-test"
        assert isinstance(manifest["total_queries"], int) and manifest["total_queries"] > 0
        assert isinstance(manifest["files"], list) and len(manifest["files"]) >= 3
        assert manifest["format"] == "trec"
        assert "sota_enhancements" in manifest
        assert manifest["sota_enhancements"]["hybrid_bm25_dense"] is True
        assert manifest["sota_enhancements"]["reciprocal_rank_fusion_k"] == 60
        assert "verified_metrics" in manifest
        assert manifest["verified_metrics"]["ndcg@10_gain_pct"] == 12.0

        # Verify Metadata Schema
        assert metadata["submission_name"] == "Piragi SOTA RAG"
        assert metadata["team_name"] == "Piragi"
        assert metadata["run_id"] == "piragi-stress-test"
        assert "parameters" in metadata
        assert metadata["parameters"]["top_k"] == 100

    def test_trec_byte_level_line_formatting(self, fresh_airbench_zip: Path):
        with zipfile.ZipFile(fresh_airbench_zip, "r") as zf:
            for run_name in ["default.run", "cranfield.run", "beir.run"]:
                raw_bytes = zf.read(run_name)
                # Ensure valid UTF-8
                text = raw_bytes.decode("utf-8")
                lines = text.splitlines()
                assert len(lines) > 0, f"{run_name} is empty"

                queries: Dict[str, List[tuple[int, float]]] = {}
                for line_idx, line in enumerate(lines, start=1):
                    assert not line.startswith(" "), f"{run_name}:{line_idx} has leading whitespace"
                    assert not line.endswith(" "), f"{run_name}:{line_idx} has trailing whitespace"

                    parts = line.split(" ")
                    assert len(parts) == 6, f"{run_name}:{line_idx} expected 6 columns, got {len(parts)}: {line}"
                    qid, iter_tag, doc_id, rank_str, score_str, run_id = parts

                    assert iter_tag == "Q0", f"{run_name}:{line_idx} iter tag is not Q0"
                    assert run_id == "piragi-stress-test"
                    assert len(qid) > 0 and len(doc_id) > 0

                    rank = int(rank_str)
                    score = float(score_str)
                    queries.setdefault(qid, []).append((rank, score))

                # Verify rank progression and non-increasing scores per query
                for qid, items in queries.items():
                    ranks = [item[0] for item in items]
                    scores = [item[1] for item in items]

                    # Rank must start at 1
                    assert ranks[0] == 1, f"{run_name} query {qid} rank starts at {ranks[0]} instead of 1"
                    # Strictly sequential: 1, 2, 3, ...
                    expected_ranks = list(range(1, len(ranks) + 1))
                    assert ranks == expected_ranks, f"{run_name} query {qid} ranks not sequential: {ranks}"

                    # Scores strictly non-increasing
                    for i in range(1, len(scores)):
                        assert scores[i] <= scores[i - 1], (
                            f"{run_name} query {qid} score not non-increasing: {scores[i-1]} < {scores[i]}"
                        )


# ===========================================================================
# 4. Concurrency Stress Testing
# ===========================================================================

class TestConcurrencyStress:
    """Stress-test concurrent exports across threads and processes."""

    def test_multithreaded_concurrent_exports_separate_dirs(self, tmp_path: Path):
        """Execute 16 concurrent worker threads generating all artifacts into isolated dirs."""
        num_workers = 16
        base_dir = tmp_path / "threads_isolated"

        def _worker_task(worker_id: int) -> Dict[str, Any]:
            worker_dir = base_dir / f"worker_{worker_id}"
            paths = export_all(
                input_file=BENCHMARK_RESULTS_PATH,
                output_dir=worker_dir,
                run_id=f"thread-worker-{worker_id}",
            )
            val_results = {}
            for b_name, p in paths.items():
                is_valid, errs = validate_submission_file(p, benchmark_type=b_name)
                val_results[b_name] = (is_valid, errs)
            return {"worker_id": worker_id, "paths": paths, "validations": val_results}

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(_worker_task, i) for i in range(num_workers)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert len(results) == num_workers
        for res in results:
            for b_name, (valid, errs) in res["validations"].items():
                assert valid is True, f"Worker {res['worker_id']} failed on {b_name}: {errs}"

    def test_multithreaded_concurrent_exports_shared_dir(self, tmp_path: Path):
        """Execute multiple threads exporting to the same directory simultaneously."""
        shared_dir = tmp_path / "shared_export_dir"

        def _export_enterpriserag(idx: int):
            target = shared_dir / f"sub_enterpriserag_{idx}.jsonl"
            return export_enterpriserag(BENCHMARK_RESULTS_PATH, target)

        def _export_airbench(idx: int):
            target = shared_dir / f"sub_airbench_{idx}.zip"
            return export_airbench(BENCHMARK_RESULTS_PATH, target)

        def _export_gaia(idx: int):
            target = shared_dir / f"sub_gaia_{idx}.jsonl"
            return export_gaia(BENCHMARK_RESULTS_PATH, target)

        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
            f_erag = [executor.submit(_export_enterpriserag, i) for i in range(4)]
            f_air = [executor.submit(_export_airbench, i) for i in range(4)]
            f_gaia = [executor.submit(_export_gaia, i) for i in range(4)]

            all_futures = f_erag + f_air + f_gaia
            for f in concurrent.futures.as_completed(all_futures):
                path_str = f.result()
                assert Path(path_str).exists()
                valid, errs = validate_submission_file(path_str)
                assert valid is True, f"File {path_str} validation failed: {errs}"

    def test_multiprocess_cli_concurrency(self, tmp_path: Path):
        """Execute 6 concurrent external CLI subprocesses and verify zero race corruption."""
        num_procs = 6
        base_dir = tmp_path / "proc_runs"
        procs = []

        for i in range(num_procs):
            proc_dir = base_dir / f"proc_{i}"
            cmd = [
                sys.executable,
                str(Path("evals/leaderboard_exporters.py").resolve()),
                "--benchmark", "all",
                "--output-dir", str(proc_dir),
                "--validate",
            ]
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            procs.append((i, p, proc_dir))

        for idx, p, proc_dir in procs:
            stdout, stderr = p.communicate(timeout=60)
            assert p.returncode == 0, f"Subprocess {idx} failed (code {p.returncode}):\n{stderr}\n{stdout}"
            assert (proc_dir / "submission_enterpriserag.jsonl").exists()
            assert (proc_dir / "airbench_submission.zip").exists()
            assert (proc_dir / "gaia_predictions.jsonl").exists()


# ===========================================================================
# 5. Mock Credential Injection and Submission Safety
# ===========================================================================

class TestCredentialSafetyAndMockSubmission:
    """Stress-test credential isolation, mock injection, and leak prevention."""

    def test_submit_without_credentials_returns_clean_skip(self, tmp_path: Path):
        clean_env = {k: v for k, v in os.environ.items() if not any(x in k.lower() for x in ["hf", "hugging"])}
        dummy_artifact = tmp_path / "dummy_submission.zip"
        dummy_artifact.write_text("dummy content", encoding="utf-8")

        with patch.dict(os.environ, clean_env, clear=True):
            ok, msg = submit_to_leaderboard("airbench", dummy_artifact, token=None)
            assert ok is False
            assert "No API credentials found" in msg
            assert "LEADERBOARD_GUIDE.md" in msg

    def test_mock_token_not_leaked_in_cli_stdout_or_stderr(self, tmp_path: Path):
        mock_secret = "hf_SUPER_SECRET_TOKEN_DO_NOT_LEAK_9999"
        out_dir = tmp_path / "cred_test_out"

        with patch("evals.leaderboard_exporters.submit_to_leaderboard") as mock_submit:
            mock_submit.return_value = (True, "Simulated successful upload")

            old_stdout, old_stderr = sys.stdout, sys.stderr
            stdout_buf, stderr_buf = io.StringIO(), io.StringIO()
            sys.stdout, sys.stderr = stdout_buf, stderr_buf

            try:
                ret = main([
                    "--benchmark", "enterpriserag",
                    "--output-dir", str(out_dir),
                    "--submit",
                    "--token", mock_secret,
                ])
            finally:
                sys.stdout, sys.stderr = old_stdout, old_stderr

            captured_out = stdout_buf.getvalue()
            captured_err = stderr_buf.getvalue()

            assert ret == 0
            assert mock_secret not in captured_out, "Secret mock token leaked in stdout!"
            assert mock_secret not in captured_err, "Secret mock token leaked in stderr!"

    def test_mock_token_not_leaked_in_generated_artifacts(self, tmp_path: Path):
        mock_secret = "hf_SECRET_KEY_NEVER_WRITE_TO_DISK_0000"
        out_dir = tmp_path / "artifact_cred_check"

        with patch("evals.leaderboard_exporters.submit_to_leaderboard") as mock_submit:
            mock_submit.return_value = (True, "Simulated upload")
            main([
                "--benchmark", "all",
                "--output-dir", str(out_dir),
                "--submit",
                "--token", mock_secret,
            ])

        # Check all artifacts
        for artifact in out_dir.iterdir():
            if artifact.suffix == ".jsonl":
                content = artifact.read_text(encoding="utf-8")
                assert mock_secret not in content, f"Secret token found in {artifact.name}!"
            elif artifact.suffix == ".zip":
                with zipfile.ZipFile(artifact, "r") as zf:
                    for name in zf.namelist():
                        raw = zf.read(name).decode("utf-8")
                        assert mock_secret not in raw, f"Secret token found in zip entry {name}!"

    def test_mock_hf_api_successful_submission(self, tmp_path: Path):
        mock_secret = "hf_valid_mock_token_123"
        artifact = tmp_path / "test_artifact.zip"
        artifact.write_text("test artifact", encoding="utf-8")

        mock_hf_api_cls = MagicMock()
        mock_api_instance = MagicMock()
        mock_hf_api_cls.return_value = mock_api_instance

        with patch.dict("sys.modules", {"huggingface_hub": MagicMock(HfApi=mock_hf_api_cls)}):
            ok, msg = submit_to_leaderboard("airbench", artifact, token=mock_secret)

            assert ok is True
            assert "Successfully uploaded test_artifact.zip" in msg
            mock_hf_api_cls.assert_called_once_with(token=mock_secret)
            mock_api_instance.upload_file.assert_called_once_with(
                path_or_fileobj=str(artifact.resolve()),
                path_in_repo="airbench/test_artifact.zip",
                repo_id="piragi-team/rag-leaderboard-submissions",
                repo_type="dataset",
            )

    def test_mock_hf_api_exception_handled_gracefully(self, tmp_path: Path):
        mock_secret = "hf_error_triggering_token"
        artifact = tmp_path / "test_artifact.jsonl"
        artifact.write_text("test content", encoding="utf-8")

        mock_hf_api_cls = MagicMock()
        mock_api_instance = MagicMock()
        mock_api_instance.upload_file.side_effect = RuntimeError("Connection timed out to HuggingFace hub")
        mock_hf_api_cls.return_value = mock_api_instance

        with patch.dict("sys.modules", {"huggingface_hub": MagicMock(HfApi=mock_hf_api_cls)}):
            ok, msg = submit_to_leaderboard("enterpriserag", artifact, token=mock_secret)
            assert ok is False
            assert "Automated upload failed" in msg
            assert "Connection timed out" in msg
            assert "LEADERBOARD_GUIDE.md" in msg

"""End-to-end observability, telemetry, and SOTA retrieval tests for piragi.

Verifies every requirement of the observability-and-instrumentation checklist:
1. Structured JSON logs with stable event names, correlation ID (requestId), and entryPoint
2. Zero secrets, tokens, or raw prompts in logs, metrics, or span attributes
3. Bounded-cardinality RED/USE metrics with p50/p95/p99 latency histograms
4. Unbroken parent-child trace spans across sync, AsyncRagi, ThreadPoolExecutor, and AsyncUpdater
5. Outbound W3C traceparent + X-Request-ID header propagation on LLMClient
6. Symptom-based alerts ('page' and 'ticket' only) test-fired with verified runbook files
7. Induced staging failure diagnosed from telemetry alone via requestId
8. De-glued PDF/EPUB subword tokenization & length-sorted CrossEncoderReranker stage fusion
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from piragi import AsyncRagi, Citation, CrossEncoderReranker, HybridSearcher, Ragi
from piragi.async_updater import AsyncUpdater, UpdateTask
from piragi.telemetry import (
    BOUNDED_OPERATIONS,
    BOUNDED_PROVIDERS,
    BOUNDED_STATUS_CLASSES,
    VALID_ENTRY_POINTS,
    TelemetryCollector,
    extract_inbound_headers,
    outbound_headers,
    parse_traceparent,
    request_context,
)


def _make_mock_embedder(dim: int = 768):
    embedder = MagicMock()
    embedder.model_name = "all-mpnet-base-v2"

    def _embed_chunks(chunks, on_progress=None):
        for idx, ch in enumerate(chunks):
            vec = np.zeros(dim, dtype=np.float32)
            vec[idx % dim] = 1.0
            ch.embedding = vec.tolist()
        return chunks

    def _embed_query(q):
        vec = np.zeros(dim, dtype=np.float32)
        vec[0] = 1.0
        return vec.tolist()

    embedder.embed_chunks.side_effect = _embed_chunks
    embedder.embed_query.side_effect = _embed_query
    return embedder


class TestStructuredLoggingAndRedaction:
    """Verify structured JSON logging, entryPoint propagation, and secret redaction."""

    def test_structured_json_output_and_secret_redaction(self, caplog):
        collector = TelemetryCollector(enabled=True, structured_logs=True)
        secret_api_key = "sk-liveSecretKey1234567890abcdef"
        secret_hf = "hf_SuperSecretToken987654321"

        with caplog.at_level(logging.INFO, logger="piragi.telemetry"):
            with request_context(entry_point="sync_api", request_id="req-redact-01", preserve_existing=False):
                rec = collector.log(
                    "info",
                    "llm_call_finished",
                    operation="llm_complete",
                    provider="openai",
                    durationMs=42.5,
                    api_key=secret_api_key,
                    authorization="Bearer {}".format(secret_api_key),
                    prompt="Full confidential customer PII prompt",
                    messages=[{"role": "user", "content": "SSN 000-00-0000"}],
                    errorCode="err_with_{}_and_{}".format(secret_api_key, secret_hf),
                )

        # Disallowed keys are stripped completely
        assert "api_key" not in rec
        assert "authorization" not in rec
        assert "prompt" not in rec
        assert "messages" not in rec

        # Secret patterns inside allowed string fields are scrubbed
        assert secret_api_key not in json.dumps(rec)
        assert secret_hf not in json.dumps(rec)
        assert "[REDACTED]" in rec["errorCode"]

        # Emitted log line is valid single-line JSON (never '[object Object]')
        telemetry_lines = [r.message for r in caplog.records if r.name == "piragi.telemetry"]
        assert len(telemetry_lines) >= 1
        parsed = json.loads(telemetry_lines[-1])
        assert parsed["event"] == "llm_call_finished"
        assert parsed["requestId"] == "req-redact-01"
        assert parsed["entryPoint"] == "sync_api"
        assert "[object Object]" not in telemetry_lines[-1]

    def test_entry_point_set_at_origin_and_preserved_across_wrappers(self, tmp_path):
        doc = tmp_path / "guide.md"
        doc.write_text("Piragi deploys cleanly on Kubernetes and Cloud Run with zero ceremony.")
        embedder = _make_mock_embedder()

        with patch("openai.OpenAI") as mock_openai:
            client_inst = MagicMock()
            comp = MagicMock()
            comp.choices = [MagicMock(message=MagicMock(content="Deploy on Cloud Run [1]."))]
            client_inst.chat.completions.create.return_value = comp
            mock_openai.return_value = client_inst

            kb = Ragi(
                persist_dir=str(tmp_path / "db"),
                embedder=embedder,
                config={"auto_update": {"enabled": False}, "llm": {"enable_query_expansion": False}},
                hooks={"post_load": lambda docs: [d.model_copy(update={"metadata": {**d.metadata, "env": "prod"}}) for d in docs]},
            )
            kb.add(str(doc))
            kb.retrieve("Cloud Run", top_k=1)
            kb.filter(env="prod").retrieve("Kubernetes", top_k=1)

            async def _run_async():
                akb = AsyncRagi(
                    persist_dir=str(tmp_path / "db2"),
                    embedder=embedder,
                    config={"auto_update": {"enabled": False}},
                )
                await akb.add(str(doc))
                await akb.retrieve("Cloud Run", top_k=1)
                return await akb.stats()

            async_stats = asyncio.run(_run_async())

        sync_entry_points = {
            s["labels"]["entry_point"]
            for s in kb.stats()["series"]
            if s["labels"]["operation"] == "retrieve"
        }
        assert "sync_api" in sync_entry_points
        assert "filtered_api" in sync_entry_points

        async_entry_points = {
            s["labels"]["entry_point"]
            for s in async_stats["series"]
            if s["labels"]["operation"] in ("ingest", "retrieve")
        }
        assert "async_api" in async_entry_points


class TestBoundedMetricsAndPercentiles:
    """Verify RED/USE metrics use bounded labels and report p50/p95/p99 histograms."""

    def test_unbounded_labels_clamped_and_percentiles_accurate(self):
        collector = TelemetryCollector()
        # Attempt cardinality bomb with random user IDs, URLs, and raw status codes
        for i in range(1, 101):
            collector.record_red(
                operation="user_{}_custom_op".format(i),
                duration_ms=float(i),
                status_class="http_{}".format(500 + (i % 5)),
                provider="https://tenant-{}.example.com/v1".format(i),
                entry_point="webhook_{}".format(i),
            )

        snap = collector.snapshot()
        # All 100 high-cardinality calls collapsed into a SINGLE bounded series!
        assert len(snap["series"]) == 1
        labels = snap["series"][0]["labels"]
        assert labels["operation"] in BOUNDED_OPERATIONS
        assert labels["entry_point"] in VALID_ENTRY_POINTS
        assert labels["status_class"] in BOUNDED_STATUS_CLASSES
        assert labels["provider"] in BOUNDED_PROVIDERS

        op_stats = snap["red"]["other"]
        assert op_stats["requests"] == 100
        assert op_stats["p50_ms"] == 50.0
        assert op_stats["p95_ms"] == 95.0
        assert op_stats["p99_ms"] == 99.0


class TestDistributedTracingAndPropagation:
    """Verify unbroken trace trees across ThreadPoolExecutor, AsyncUpdater, and LLMClient."""

    def test_end_to_end_trace_without_broken_spans_and_outbound_headers(self, tmp_path):
        doc = tmp_path / "arch.md"
        doc.write_text("Distributed tracing propagates W3C traceparent across worker threads.")
        embedder = _make_mock_embedder()

        captured_headers = {}

        with patch("openai.OpenAI") as mock_openai:
            client_inst = MagicMock()

            def _fake_create(**kwargs):
                if "extra_headers" in kwargs:
                    captured_headers.update(kwargs["extra_headers"])
                msg_text = str(kwargs.get("messages", ""))
                comp = MagicMock()
                if "alternative" in msg_text.lower() or "variation" in msg_text.lower():
                    comp.choices = [MagicMock(message=MagicMock(content="W3C traceparent\nworker threads"))]
                else:
                    comp.choices = [MagicMock(message=MagicMock(content="Tracing uses W3C traceparent [1]."))]
                return comp

            client_inst.chat.completions.create.side_effect = _fake_create
            mock_openai.return_value = client_inst

            kb = Ragi(
                str(doc),
                persist_dir=str(tmp_path / "db"),
                embedder=embedder,
                config={
                    "auto_update": {"enabled": False},
                    "llm": {"api_key": "sk-topSecretApiKey999999", "enable_query_expansion": True},
                    "retrieval": {"use_hybrid_search": True},
                },
            )
            kb.telemetry.reset()

            inbound = extract_inbound_headers({
                "X-Request-ID": "req-e2e-trace-77",
                "traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
            })
            with request_context(
                entry_point="playground_http",
                request_id=inbound["request_id"],
                trace_id=inbound["trace_id"],
                parent_span_id=inbound["parent_span_id"],
                preserve_existing=False,
            ):
                ans = kb.ask("How does tracing propagate?", top_k=1)

        assert "traceparent" in ans.text.lower()
        # Outbound LLM call carried correlation ID and W3C traceparent
        assert captured_headers.get("X-Request-ID") == "req-e2e-trace-77"
        assert captured_headers.get("X-Piragi-Entry-Point") == "playground_http"
        tp_tid, tp_sid, tp_sampled = parse_traceparent(captured_headers.get("traceparent"))
        assert tp_tid == "4bf92f3577b34da6a3ce929d0e0e4736"
        assert tp_sid is not None
        assert tp_sampled is True

        # Verify every span in the trace shares traceId, requestId, entryPoint, and valid parent linkage
        spans = kb.telemetry.get_spans(request_id="req-e2e-trace-77")
        assert len(spans) >= 5
        span_ids = {s["spanId"] for s in spans}
        root_span = [s for s in spans if s["name"] == "ragi.ask"][0]
        assert root_span["parentSpanId"] == "00f067aa0ba902b7"

        for sp in spans:
            assert sp["traceId"] == "4bf92f3577b34da6a3ce929d0e0e4736"
            assert sp["requestId"] == "req-e2e-trace-77"
            assert sp["entryPoint"] == "playground_http"
            assert "sk-topSecretApiKey999999" not in json.dumps(sp)
            if sp["spanId"] != root_span["spanId"]:
                assert sp["parentSpanId"] in span_ids

    def test_background_updater_propagates_queue_trace_metadata(self, tmp_path):
        target = tmp_path / "live.txt"
        target.write_text("v1")
        collector = TelemetryCollector()
        refreshed = []

        updater = AsyncUpdater(
            refresh_callback=lambda src: refreshed.append(src),
            check_interval=1.0,
            max_workers=1,
            telemetry=collector,
        )
        updater.register_source(str(target), "v1", check_interval=0.01)
        target.write_text("v2 updated content")

        task = UpdateTask(
            str(target),
            trace_meta={
                "request_id": "bg-req-42",
                "entry_point": "background_updater",
                "traceparent": "00-11111111111111111111111111111111-2222222222222222-01",
            },
        )
        tid, psid, _ = parse_traceparent(task.traceparent)
        with request_context(
            entry_point="background_updater",
            request_id=task.request_id,
            trace_id=tid,
            parent_span_id=psid,
            preserve_existing=False,
        ):
            updater._check_and_refresh(task.source)

        assert refreshed == [str(target)]
        spans = collector.get_spans(request_id="bg-req-42")
        assert len(spans) == 1
        assert spans[0]["entryPoint"] == "background_updater"
        assert spans[0]["parentSpanId"] == "2222222222222222"


class TestAlertsRunbooksAndFailureDiagnosis:
    """Test-fire every symptom alert, verify runbooks exist, and diagnose an induced failure."""

    def test_all_symptom_alerts_fire_and_runbooks_resolve(self):
        repo_root = Path(__file__).resolve().parent.parent
        collector = TelemetryCollector()

        # Induce symptoms across all 4 alert rules
        for _ in range(9):
            collector.record_red("ask", duration_ms=50.0, status_class="ok", provider="openai")
        collector.record_red("ask", duration_ms=2500.0, status_class="timeout", provider="openai")

        for _ in range(8):
            collector.record_retrieval_quality(citation_count=3)
        for _ in range(2):
            collector.record_retrieval_quality(citation_count=0)

        collector.record_updater_queue(queue_depth=4, workers=2, oldest_task_age_s=650.0, error=False)

        alerts = collector.evaluate_alerts()
        by_name = {a["name"]: a for a in alerts}
        assert set(by_name.keys()) == {
            "ragi_high_error_rate",
            "ragi_high_p99_latency",
            "ragi_empty_retrieval_spike",
            "ragi_refresh_lag",
        }

        for alert in alerts:
            # Only 'page' and 'ticket' severities allowed
            assert alert["severity"] in ("page", "ticket")
            runbook_path = repo_root / alert["runbook"]
            assert runbook_path.is_file(), "Missing runbook: {}".format(runbook_path)
            content = runbook_path.read_text(encoding="utf-8")
            assert "**Means:**" in content
            assert "**First check:**" in content
            assert "**Escalate to:**" in content

    def test_induced_failure_located_via_telemetry_alone(self, tmp_path):
        doc = tmp_path / "service.md"
        doc.write_text("Service SLA is 99.95%.")
        embedder = _make_mock_embedder()

        with patch("openai.OpenAI") as mock_openai:
            client_inst = MagicMock()
            client_inst.chat.completions.create.side_effect = TimeoutError("upstream gateway timed out")
            mock_openai.return_value = client_inst

            kb = Ragi(
                str(doc),
                persist_dir=str(tmp_path / "db"),
                embedder=embedder,
                config={
                    "auto_update": {"enabled": False},
                    "llm": {"enable_query_expansion": False, "base_url": "https://api.openai.com/v1"},
                    "telemetry": {"structured_logs": True},
                },
            )

            with patch("piragi.retry.time.sleep"):
                with request_context(entry_point="sync_api", request_id="inc-9001", preserve_existing=False):
                    with pytest.raises((TimeoutError, RuntimeError)):
                        kb.ask("What is the SLA?", top_k=1)

        # Diagnose purely from telemetry by correlation ID 'inc-9001'
        err_logs = kb.telemetry.get_logs(request_id="inc-9001", level="error")
        assert len(err_logs) >= 1
        assert err_logs[0]["operation"] == "llm_complete"
        assert err_logs[0]["provider"] == "openai"
        assert err_logs[0]["statusClass"] == "timeout"
        assert err_logs[0]["errorType"] == "TimeoutError"

        spans = kb.telemetry.get_spans(request_id="inc-9001")
        failed_spans = [s for s in spans if s["status"] == "error"]
        assert any(s["name"] == "llm.complete" for s in failed_spans)
        healthy_spans = [s for s in spans if s["status"] == "ok"]
        assert any(s["name"] == "store.vector_search" for s in healthy_spans)


class TestDegluedTokenizationAndFastReranker:
    """Verify PDF/EPUB glued-word recovery and multi-batch length-sorted CrossEncoder reranking."""

    def test_hybrid_searcher_recovers_glued_pdf_tokens(self):
        corpus = [
            Citation(
                source="paper.pdf",
                chunk="The actualpowerusageofaGPUisdependentonitsutilization during training.",
                score=0.2,
            ),
            Citation(
                source="notes.pdf",
                chunk="Actual power and utilization metrics are measured in watts per node.",
                score=0.2,
            ),
            Citation(
                source="safety.pdf",
                chunk="Researchers work to understandandimprovemodelsafety across benchmarks.",
                score=0.2,
            ),
            Citation(
                source="vocab.pdf",
                chunk="We understand how to improve model safety and dependent usage.",
                score=0.1,
            ),
        ]
        hs = HybridSearcher(vector_weight=0.1, bm25_weight=0.9, use_rrf=True)
        hits_gpu = hs.search_corpus("GPU power usage dependent utilization", corpus, corpus, top_k=2)
        assert hits_gpu[0].source == "paper.pdf"

        hits_safety = hs.search_corpus("understand improve model safety", corpus, corpus, top_k=2)
        assert hits_safety[0].source in ("safety.pdf", "vocab.pdf")
        assert any(c.source == "safety.pdf" for c in hits_safety)

    @patch("sentence_transformers.CrossEncoder")
    def test_cross_encoder_length_sorted_batching_and_stage_fusion(self, mock_ce_cls):
        mock_model = MagicMock()

        # Score proportional to presence of keyword 'winner' regardless of input order
        def _predict(pairs, batch_size=4):
            return np.array([4.0 if "winner" in p[1] else -2.0 for p in pairs], dtype=np.float32)

        mock_model.predict.side_effect = _predict
        mock_ce_cls.return_value = mock_model

        citations = [
            Citation(source="d{}".format(i), chunk="x" * (100 - i * 7), score=0.5)
            for i in range(10)
        ]
        citations[6] = Citation(source="winner_doc", chunk="short winner text", score=0.95)

        reranker = CrossEncoderReranker(batch_size=4, fusion_weight=0.2)
        ranked = reranker.rerank("find winner", citations, top_k=3)
        assert ranked[0].source == "winner_doc"
        assert len(ranked) == 3

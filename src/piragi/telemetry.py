"""Production observability and telemetry for piragi.

Implements structured JSON logging, bounded-cardinality RED/USE metrics with
latency histograms (p50/p95/p99), W3C trace context propagation across sync,
async, thread-pool, and background-queue boundaries, and symptom-based alerting
with runbooks — all with zero extra ceremony on piragi's public API.

On-Call Questions Mapped to Signals:
1. Are queries (ask/retrieve) or ingestions failing, and at which stage?
   - Metric: red.<operation>.error_rate by (operation, entry_point, status_class, provider)
   - Trace:  Hierarchical spans (ragi.ask -> search.*, rerank.*, llm.complete)
   - Log:    operation_failed JSON event with requestId + entryPoint + errorType
2. Where is latency going when ask or retrieve p99 exceeds 2000ms?
   - Metric: red.<operation>.{p50_ms, p95_ms, p99_ms} histogram
   - Trace:  Per-hop span duration_ms breakdown
3. Are external LLM / Vertex AI calls retrying, throttling, or timing out?
   - Metric: red.llm_complete by provider and status_class
   - Log:    llm_retry (warn) and llm_request_failed (error) with attempt & delayMs
4. Are caches saturating, background updates lagging, or queries returning empty citations?
   - Metric: use.bm25_cache, use.updater_queue, quality.empty_result_rate
   - Alert:  Symptom-based page/ticket alerts linked to docs/runbooks/*.md
"""

from collections import deque
from contextlib import contextmanager
import contextvars
from datetime import datetime, timezone
import json
import logging
import math
import random
import re
import threading
import time
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple
import uuid

telemetry_logger = logging.getLogger("piragi.telemetry")

# Strictly bounded label sets (prevents metric cardinality explosions)
VALID_ENTRY_POINTS = (
    "sync_api",
    "async_api",
    "filtered_api",
    "background_updater",
    "playground_http",
    "cli_eval",
    "other",
)

BOUNDED_OPERATIONS = (
    "ingest",
    "retrieve",
    "ask",
    "refresh",
    "embed",
    "vector_search",
    "hybrid_fusion",
    "rerank",
    "query_transform",
    "llm_complete",
    "other",
)

BOUNDED_STATUS_CLASSES = (
    "ok",
    "2xx",
    "4xx",
    "5xx",
    "timeout",
    "rate_limit",
    "error",
    "other",
)

BOUNDED_PROVIDERS = (
    "local",
    "ollama",
    "openai",
    "vertexai",
    "custom",
    "lancedb",
    "qdrant",
    "postgres",
    "pinecone",
    "bm25",
    "cross_encoder",
    "other",
)

HISTOGRAM_BUCKETS_MS = (
    1.0,
    2.5,
    5.0,
    10.0,
    25.0,
    50.0,
    100.0,
    250.0,
    500.0,
    1000.0,
    2000.0,
    5000.0,
    10000.0,
)

# Strict allowlist of metadata keys permitted in structured logs and span attributes.
# Raw prompts, message bodies, API keys, tokens, and PII are never allowed.
ALLOWED_TELEMETRY_FIELDS = frozenset({
    "operation",
    "provider",
    "statusClass",
    "durationMs",
    "attempt",
    "maxAttempts",
    "delayMs",
    "errorType",
    "errorCode",
    "topK",
    "candidateCount",
    "citationCount",
    "chunkCount",
    "docCount",
    "variationCount",
    "cacheHit",
    "cacheSize",
    "maxCacheSize",
    "queueDepth",
    "oldestTaskAgeS",
    "hasFilters",
    "strategy",
    "modelFamily",
    "abstained",
    "httpMethod",
    "route",
})

_SECRET_KEY_PATTERN = re.compile(
    r"(api[_-]?key|secret|token|password|passwd|authorization|auth|credential|bearer|prompt|messages|body|content|email)",
    re.IGNORECASE,
)

_SECRET_VALUE_PATTERN = re.compile(
    r"(sk-[A-Za-z0-9_\-]{8,}|hf_[A-Za-z0-9]{8,}|ya29\.[A-Za-z0-9_\-\.]{8,}|Bearer\s+[A-Za-z0-9_\-\.]+)",
    re.IGNORECASE,
)

_W3C_TRACEPARENT_RE = re.compile(
    r"^[ \t]*00-([0-9a-f]{32})-([0-9a-f]{16})-([0-9a-f]{2})[ \t]*$"
)

# Context variables propagated across threads, asyncio, queues, and HTTP headers
_request_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "piragi_request_id", default=None
)
_entry_point_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "piragi_entry_point", default=None
)
_trace_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "piragi_trace_id", default=None
)
_span_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "piragi_span_id", default=None
)
_sampled_var: contextvars.ContextVar[Optional[bool]] = contextvars.ContextVar(
    "piragi_sampled", default=None
)


def _bounded(value: Optional[str], allowed: Tuple[str, ...], default: str = "other") -> str:
    if not value or not isinstance(value, str):
        return default
    cleaned = value.strip().lower()
    return cleaned if cleaned in allowed else default


def classify_provider(base_url: Optional[str] = None, provider: Optional[str] = None) -> str:
    """Map provider or base_url to a strictly bounded provider label."""
    if provider:
        p = provider.strip().lower()
        if p in ("vertex", "vertexai", "gcp"):
            return "vertexai"
        if p in BOUNDED_PROVIDERS:
            return p
    if not base_url:
        return "ollama"
    lowered = base_url.lower()
    if "11434" in lowered or "ollama" in lowered:
        return "ollama"
    if "googleapis.com" in lowered or "aiplatform" in lowered:
        return "vertexai"
    if "api.openai.com" in lowered:
        return "openai"
    return "custom"


def classify_store_provider(store: Any) -> str:
    """Map a vector store instance to a strictly bounded provider label."""
    name = type(store).__name__.lower()
    if "lance" in name:
        return "lancedb"
    if "qdrant" in name:
        return "qdrant"
    if "postgres" in name:
        return "postgres"
    if "pinecone" in name:
        return "pinecone"
    return "custom"


def classify_exception(exc: BaseException) -> str:
    """Map an exception to a strictly bounded status_class label."""
    name = type(exc).__name__.lower()
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        if status_code == 429:
            return "rate_limit"
        if 400 <= status_code < 500:
            return "4xx"
        if status_code >= 500:
            return "5xx"
    if "ratelimit" in name or "rate_limit" in name or "quota" in name:
        return "rate_limit"
    if "timeout" in name or isinstance(exc, TimeoutError):
        return "timeout"
    if "validation" in name or "badrequest" in name or isinstance(exc, ValueError):
        return "4xx"
    if "internalserver" in name or "connection" in name or isinstance(exc, ConnectionError):
        return "5xx"
    return "error"


def sanitize_value(value: Any) -> Any:
    """Scrub secret token patterns and coerce values into JSON-safe primitives."""
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return round(value, 4)
    text = str(value)
    scrubbed = _SECRET_VALUE_PATTERN.sub("[REDACTED]", text)
    if len(scrubbed) > 160:
        scrubbed = "{}...".format(scrubbed[:157])
    return scrubbed


def sanitize_fields(fields: Dict[str, Any]) -> Dict[str, Any]:
    """Allowlist and scrub telemetry fields so secrets/PII/bodies can never leak."""
    clean: Dict[str, Any] = {}
    for key, val in fields.items():
        if val is None:
            continue
        if _SECRET_KEY_PATTERN.search(str(key)):
            continue
        if key not in ALLOWED_TELEMETRY_FIELDS:
            continue
        clean[key] = sanitize_value(val)
    return clean


def current_request_id() -> Optional[str]:
    return _request_id_var.get()


def current_entry_point() -> Optional[str]:
    return _entry_point_var.get()


def current_trace_id() -> Optional[str]:
    return _trace_id_var.get()


def current_span_id() -> Optional[str]:
    return _span_id_var.get()


def parse_traceparent(header_value: Optional[str]) -> Tuple[Optional[str], Optional[str], Optional[bool]]:
    """Parse W3C traceparent header ('00-<trace_id>-<span_id>-<flags>')."""
    if not header_value or not isinstance(header_value, str):
        return None, None, None
    match = _W3C_TRACEPARENT_RE.match(header_value)
    if not match:
        return None, None, None
    trace_id, parent_span_id, flags_hex = match.groups()
    if trace_id == "0" * 32 or parent_span_id == "0" * 16:
        return None, None, None
    sampled = (int(flags_hex, 16) & 0x01) == 0x01
    return trace_id, parent_span_id, sampled


def format_traceparent(
    trace_id: Optional[str] = None,
    span_id: Optional[str] = None,
    sampled: bool = True,
) -> Optional[str]:
    """Format W3C traceparent header from current or supplied trace context."""
    tid = trace_id or _trace_id_var.get()
    sid = span_id or _span_id_var.get()
    if not tid or not sid:
        return None
    flags = "01" if sampled else "00"
    return "00-{}-{}-{}".format(tid, sid, flags)


def extract_inbound_headers(headers: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract request_id, entry_point, and W3C traceparent from inbound HTTP/queue metadata."""
    if not headers:
        return {}
    lowered = {str(k).lower(): v for k, v in headers.items() if v is not None}
    req_id = lowered.get("x-request-id") or lowered.get("x-correlation-id")
    if req_id is not None:
        req_id = sanitize_value(req_id)
    ep = lowered.get("x-piragi-entry-point")
    trace_id, parent_span_id, sampled = parse_traceparent(lowered.get("traceparent"))
    return {
        "request_id": req_id,
        "entry_point": _bounded(ep, VALID_ENTRY_POINTS) if ep else None,
        "trace_id": trace_id,
        "parent_span_id": parent_span_id,
        "sampled": sampled,
    }


def outbound_headers() -> Dict[str, str]:
    """Build outbound correlation & W3C trace headers for external HTTP/LLM calls."""
    headers: Dict[str, str] = {}
    req_id = _request_id_var.get()
    ep = _entry_point_var.get()
    if req_id:
        headers["X-Request-ID"] = req_id
    if ep:
        headers["X-Piragi-Entry-Point"] = ep
    tp = format_traceparent(sampled=bool(_sampled_var.get() is not False))
    if tp:
        headers["traceparent"] = tp
    return headers


def capture_context_metadata(default_entry_point: str = "sync_api") -> Dict[str, str]:
    """Capture correlation & trace metadata for queue tasks (e.g. AsyncUpdater)."""
    req_id = _request_id_var.get() or uuid.uuid4().hex
    ep = _entry_point_var.get() or _bounded(default_entry_point, VALID_ENTRY_POINTS, "sync_api")
    tid = _trace_id_var.get() or uuid.uuid4().hex
    sid = _span_id_var.get() or uuid.uuid4().hex[:16]
    return {
        "request_id": req_id,
        "entry_point": ep,
        "traceparent": "00-{}-{}-01".format(tid, sid),
    }


def wrap_with_context(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a callable with a snapshot of contextvars so thread pools keep trace/request IDs."""
    ctx = contextvars.copy_context()

    def _wrapped(*args: Any, **kwargs: Any) -> Any:
        return ctx.run(fn, *args, **kwargs)

    return _wrapped


@contextmanager
def request_context(
    entry_point: str = "sync_api",
    request_id: Optional[str] = None,
    trace_id: Optional[str] = None,
    parent_span_id: Optional[str] = None,
    sample_rate: float = 1.0,
    preserve_existing: bool = True,
):
    """Establish correlation ID, entryPoint, and root trace context at a system boundary.

    When preserve_existing=True (default), an outer boundary (e.g. async_api,
    filtered_api, background_updater, playground_http) that already stamped
    requestId and entryPoint is preserved downstream rather than overwritten.
    """
    existing_req = _request_id_var.get()
    existing_ep = _entry_point_var.get()
    existing_tid = _trace_id_var.get()
    existing_sid = _span_id_var.get()
    existing_sampled = _sampled_var.get()

    tokens = []
    if not (preserve_existing and existing_req):
        new_req = str(request_id) if request_id else uuid.uuid4().hex
        tokens.append((_request_id_var, _request_id_var.set(new_req)))

    if not (preserve_existing and existing_ep):
        bounded_ep = _bounded(entry_point, VALID_ENTRY_POINTS, "sync_api")
        tokens.append((_entry_point_var, _entry_point_var.set(bounded_ep)))

    if not (preserve_existing and existing_tid):
        new_tid = trace_id if (trace_id and len(trace_id) == 32) else uuid.uuid4().hex
        tokens.append((_trace_id_var, _trace_id_var.set(new_tid)))

    if parent_span_id and not (preserve_existing and existing_sid):
        tokens.append((_span_id_var, _span_id_var.set(parent_span_id)))

    if existing_sampled is None:
        is_sampled = True if sample_rate >= 1.0 else (random.random() < max(0.0, sample_rate))
        tokens.append((_sampled_var, _sampled_var.set(is_sampled)))

    try:
        yield {
            "requestId": _request_id_var.get(),
            "entryPoint": _entry_point_var.get(),
            "traceId": _trace_id_var.get(),
        }
    finally:
        for var, token in reversed(tokens):
            var.reset(token)


class LatencyHistogram:
    """Thread-safe fixed-bucket histogram + bounded window for exact p50/p95/p99."""

    def __init__(self, buckets: Tuple[float, ...] = HISTOGRAM_BUCKETS_MS, window_size: int = 2048):
        self.buckets = buckets
        self.bucket_counts: Dict[str, int] = {"le_{}".format(b): 0 for b in buckets}
        self.bucket_counts["le_inf"] = 0
        self.samples: Deque[float] = deque(maxlen=window_size)
        self.count: int = 0
        self.sum_ms: float = 0.0

    def observe(self, duration_ms: float) -> None:
        val = max(0.0, float(duration_ms))
        self.count += 1
        self.sum_ms += val
        self.samples.append(val)
        for b in self.buckets:
            if val <= b:
                self.bucket_counts["le_{}".format(b)] += 1
        self.bucket_counts["le_inf"] += 1

    def percentile(self, q: float) -> float:
        if not self.samples:
            return 0.0
        ordered = sorted(self.samples)
        idx = min(len(ordered) - 1, max(0, math.ceil((q / 100.0) * len(ordered)) - 1))
        return round(ordered[idx], 3)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "count": self.count,
            "p50_ms": self.percentile(50.0),
            "p95_ms": self.percentile(95.0),
            "p99_ms": self.percentile(99.0),
            "buckets": dict(self.bucket_counts),
        }


class TelemetryCollector:
    """Zero-ceremony observability collector for RED/USE metrics, spans, logs, and alerts."""

    def __init__(
        self,
        enabled: bool = True,
        structured_logs: bool = False,
        sample_rate: float = 1.0,
        max_spans: int = 512,
        max_logs: int = 512,
        alert_thresholds: Optional[Dict[str, float]] = None,
    ):
        self.enabled = enabled
        self.structured_logs = structured_logs
        self.sample_rate = max(0.0, min(1.0, float(sample_rate)))
        self._lock = threading.RLock()
        self._started_at = time.monotonic()

        # RED series keyed by bounded (operation, entry_point, status_class, provider)
        self._series_histograms: Dict[Tuple[str, str, str, str], LatencyHistogram] = {}
        # Aggregate per-operation histograms for fast SLO / dashboard queries
        self._op_histograms: Dict[str, LatencyHistogram] = {}
        self._op_requests: Dict[str, int] = {}
        self._op_errors: Dict[str, int] = {}

        # USE resource metrics
        self._use_bm25 = {
            "utilization": 0.0,
            "cache_size": 0,
            "max_cache_size": 16,
            "hits": 0,
            "misses": 0,
            "saturation_evictions": 0,
            "errors": 0,
        }
        self._use_updater = {
            "utilization": 0.0,
            "queue_depth": 0,
            "workers": 0,
            "oldest_task_age_s": 0.0,
            "saturation_queued": 0,
            "errors": 0,
        }
        self._quality = {
            "total_queries": 0,
            "empty_results": 0,
        }

        self._spans: Deque[Dict[str, Any]] = deque(maxlen=max_spans)
        self._logs: Deque[Dict[str, Any]] = deque(maxlen=max_logs)

        defaults = {
            "error_rate_max": 0.01,
            "min_requests_for_error_alert": 5.0,
            "p99_latency_ms_max": 2000.0,
            "empty_retrieval_rate_max": 0.15,
            "min_queries_for_empty_alert": 10.0,
            "refresh_lag_seconds_max": 600.0,
        }
        if alert_thresholds:
            defaults.update(alert_thresholds)
        self.alert_thresholds = defaults

        # Optional OpenTelemetry tracer bridge (zero hard dependency)
        self._otel_tracer = None
        try:
            from opentelemetry import trace as otel_trace  # type: ignore
            self._otel_tracer = otel_trace.get_tracer("piragi")
        except Exception:
            self._otel_tracer = None

    def log(self, level: str, event: str, **fields: Any) -> Dict[str, Any]:
        """Emit a structured JSON log event with correlation ID, entryPoint, and allowlisted fields."""
        clean_level = level.lower() if level.lower() in ("debug", "info", "warn", "error") else "info"
        record: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": clean_level,
            "event": sanitize_value(event),
            "requestId": _request_id_var.get() or "unscoped",
            "entryPoint": _entry_point_var.get() or "sync_api",
        }
        tid = _trace_id_var.get()
        sid = _span_id_var.get()
        if tid:
            record["traceId"] = tid
        if sid:
            record["spanId"] = sid
        record.update(sanitize_fields(fields))

        if not self.enabled:
            return record

        serialized = json.dumps(record, separators=(",", ":"), sort_keys=False)
        with self._lock:
            self._logs.append(record)

        if self.structured_logs or clean_level in ("warn", "error"):
            py_level = {
                "debug": logging.DEBUG,
                "info": logging.INFO,
                "warn": logging.WARNING,
                "error": logging.ERROR,
            }[clean_level]
            telemetry_logger.log(py_level, "%s", serialized)
        return record

    def record_red(
        self,
        operation: str,
        duration_ms: float,
        status_class: str = "ok",
        provider: str = "local",
        entry_point: Optional[str] = None,
    ) -> None:
        """Record Rate, Errors, and Duration histogram with strictly bounded labels."""
        if not self.enabled:
            return
        op = _bounded(operation, BOUNDED_OPERATIONS)
        ep = _bounded(entry_point or _entry_point_var.get() or "sync_api", VALID_ENTRY_POINTS)
        sc = _bounded(status_class, BOUNDED_STATUS_CLASSES)
        prov = _bounded(provider, BOUNDED_PROVIDERS)
        is_err = sc not in ("ok", "2xx")

        series_key = (op, ep, sc, prov)
        with self._lock:
            if series_key not in self._series_histograms:
                self._series_histograms[series_key] = LatencyHistogram()
            self._series_histograms[series_key].observe(duration_ms)

            if op not in self._op_histograms:
                self._op_histograms[op] = LatencyHistogram()
            self._op_histograms[op].observe(duration_ms)
            self._op_requests[op] = self._op_requests.get(op, 0) + 1
            if is_err:
                self._op_errors[op] = self._op_errors.get(op, 0) + 1

    def record_bm25_cache(
        self,
        cache_size: int,
        max_cache_size: int,
        hit: Optional[bool] = None,
        evicted: bool = False,
        error: bool = False,
    ) -> None:
        """Update USE metrics for the BM25 corpus cache."""
        if not self.enabled:
            return
        with self._lock:
            cap = max(1, int(max_cache_size))
            self._use_bm25["cache_size"] = int(cache_size)
            self._use_bm25["max_cache_size"] = cap
            self._use_bm25["utilization"] = round(min(1.0, float(cache_size) / float(cap)), 4)
            if hit is True:
                self._use_bm25["hits"] += 1
            elif hit is False:
                self._use_bm25["misses"] += 1
            if evicted:
                self._use_bm25["saturation_evictions"] += 1
            if error:
                self._use_bm25["errors"] += 1

    def record_updater_queue(
        self,
        queue_depth: int,
        workers: int,
        oldest_task_age_s: float = 0.0,
        error: bool = False,
    ) -> None:
        """Update USE metrics for the AsyncUpdater worker queue."""
        if not self.enabled:
            return
        with self._lock:
            w = max(1, int(workers))
            qd = max(0, int(queue_depth))
            self._use_updater["queue_depth"] = qd
            self._use_updater["workers"] = w
            self._use_updater["utilization"] = round(min(1.0, float(qd) / float(w)), 4)
            self._use_updater["oldest_task_age_s"] = round(max(0.0, float(oldest_task_age_s)), 3)
            if qd > w:
                self._use_updater["saturation_queued"] += 1
            if error:
                self._use_updater["errors"] += 1

    def record_retrieval_quality(self, citation_count: int) -> None:
        """Track query empty-result rate for symptom alerting."""
        if not self.enabled:
            return
        with self._lock:
            self._quality["total_queries"] += 1
            if citation_count <= 0:
                self._quality["empty_results"] += 1

    @contextmanager
    def span(
        self,
        name: str,
        operation: Optional[str] = None,
        provider: str = "local",
        **attributes: Any,
    ):
        """Create a hierarchical trace span and optionally record RED metrics."""
        if not self.enabled:
            yield {}
            return

        # Ensure correlation/trace context exists even if an internal component is called directly
        req_token = None
        ep_token = None
        tid_token = None
        sampled_token = None
        if _request_id_var.get() is None:
            req_token = _request_id_var.set(uuid.uuid4().hex)
        if _entry_point_var.get() is None:
            ep_token = _entry_point_var.set("sync_api")
        if _trace_id_var.get() is None:
            tid_token = _trace_id_var.set(uuid.uuid4().hex)
        if _sampled_var.get() is None:
            is_sampled = True if self.sample_rate >= 1.0 else (random.random() < self.sample_rate)
            sampled_token = _sampled_var.set(is_sampled)

        parent_span_id = _span_id_var.get()
        span_id = uuid.uuid4().hex[:16]
        span_token = _span_id_var.set(span_id)

        start_ns = time.time_ns()
        start_mono = time.perf_counter()
        span_attrs: Dict[str, Any] = sanitize_fields(attributes)
        status = "ok"
        status_class = "ok"
        error_type: Optional[str] = None

        otel_cm = None
        otel_span = None
        if self._otel_tracer is not None:
            try:
                otel_cm = self._otel_tracer.start_as_current_span(name)
                otel_span = otel_cm.__enter__()
            except Exception:
                otel_cm = None
                otel_span = None

        try:
            yield span_attrs
        except BaseException as exc:
            status = "error"
            status_class = classify_exception(exc)
            error_type = type(exc).__name__
            span_attrs["errorType"] = error_type
            span_attrs["statusClass"] = status_class
            if otel_span is not None:
                try:
                    otel_span.record_exception(exc)
                except Exception:
                    pass
            raise
        finally:
            duration_ms = round((time.perf_counter() - start_mono) * 1000.0, 3)
            clean_attrs = sanitize_fields(span_attrs)
            if error_type:
                clean_attrs["errorType"] = sanitize_value(error_type)
                clean_attrs["statusClass"] = status_class

            if otel_span is not None:
                try:
                    req_id_val = _request_id_var.get()
                    ep_val = _entry_point_var.get()
                    if req_id_val:
                        otel_span.set_attribute("piragi.request_id", req_id_val)
                    if ep_val:
                        otel_span.set_attribute("piragi.entry_point", ep_val)
                    for k, v in clean_attrs.items():
                        if v is not None:
                            otel_span.set_attribute("piragi.{}".format(k), v)
                except Exception:
                    pass
            if otel_cm is not None:
                try:
                    otel_cm.__exit__(None, None, None)
                except Exception:
                    pass

            if operation:
                self.record_red(
                    operation=operation,
                    duration_ms=duration_ms,
                    status_class=status_class,
                    provider=provider,
                )

            # Head-based sampling, keeping 100% of error spans (tail-keep on error)
            keep_span = bool(_sampled_var.get()) or (status == "error")
            if keep_span:
                span_record = {
                    "traceId": _trace_id_var.get(),
                    "spanId": span_id,
                    "parentSpanId": parent_span_id,
                    "name": sanitize_value(name),
                    "requestId": _request_id_var.get(),
                    "entryPoint": _entry_point_var.get(),
                    "startTimeNs": start_ns,
                    "durationMs": duration_ms,
                    "status": status,
                    "attributes": clean_attrs,
                }
                with self._lock:
                    self._spans.append(span_record)

            if status == "error":
                self.log(
                    "error",
                    "operation_failed",
                    operation=operation or name,
                    provider=provider,
                    statusClass=status_class,
                    durationMs=duration_ms,
                    errorType=error_type,
                )

            _span_id_var.reset(span_token)
            if sampled_token is not None:
                _sampled_var.reset(sampled_token)
            if tid_token is not None:
                _trace_id_var.reset(tid_token)
            if ep_token is not None:
                _entry_point_var.reset(ep_token)
            if req_token is not None:
                _request_id_var.reset(req_token)

    def evaluate_alerts(self, overrides: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
        """Evaluate symptom-based alerts (severities: 'page' or 'ticket' only)."""
        thresholds = dict(self.alert_thresholds)
        if overrides:
            thresholds.update(overrides)

        alerts: List[Dict[str, Any]] = []
        with self._lock:
            # 1. User-facing error rate > threshold (severity: page)
            user_ops = ("ask", "retrieve", "ingest")
            total_req = sum(self._op_requests.get(op, 0) for op in user_ops)
            total_err = sum(self._op_errors.get(op, 0) for op in user_ops)
            min_req = int(thresholds["min_requests_for_error_alert"])
            if total_req >= max(1, min_req):
                err_rate = float(total_err) / float(total_req)
                if err_rate > thresholds["error_rate_max"]:
                    alerts.append({
                        "name": "ragi_high_error_rate",
                        "severity": "page",
                        "value": round(err_rate, 4),
                        "threshold": thresholds["error_rate_max"],
                        "summary": "User-facing error rate {:.2%} exceeds {:.2%}".format(
                            err_rate, thresholds["error_rate_max"]
                        ),
                        "runbook": "docs/runbooks/ragi-high-error-rate.md",
                    })

            # 2. User-facing p99 latency > threshold (severity: page)
            max_p99 = 0.0
            slowest_op = None
            for op in ("ask", "retrieve"):
                hist = self._op_histograms.get(op)
                if hist and hist.count > 0:
                    p99 = hist.percentile(99.0)
                    if p99 > max_p99:
                        max_p99 = p99
                        slowest_op = op
            if slowest_op is not None and max_p99 > thresholds["p99_latency_ms_max"]:
                alerts.append({
                    "name": "ragi_high_p99_latency",
                    "severity": "page",
                    "operation": slowest_op,
                    "value": round(max_p99, 3),
                    "threshold": thresholds["p99_latency_ms_max"],
                    "summary": "{} p99 latency {:.1f}ms exceeds {:.1f}ms".format(
                        slowest_op, max_p99, thresholds["p99_latency_ms_max"]
                    ),
                    "runbook": "docs/runbooks/ragi-high-p99-latency.md",
                })

            # 3. Empty retrieval spike > threshold (severity: ticket)
            q_total = self._quality["total_queries"]
            q_empty = self._quality["empty_results"]
            if q_total >= max(1, int(thresholds["min_queries_for_empty_alert"])):
                empty_rate = float(q_empty) / float(q_total)
                if empty_rate > thresholds["empty_retrieval_rate_max"]:
                    alerts.append({
                        "name": "ragi_empty_retrieval_spike",
                        "severity": "ticket",
                        "value": round(empty_rate, 4),
                        "threshold": thresholds["empty_retrieval_rate_max"],
                        "summary": "Empty retrieval rate {:.2%} exceeds {:.2%}".format(
                            empty_rate, thresholds["empty_retrieval_rate_max"]
                        ),
                        "runbook": "docs/runbooks/ragi-empty-retrieval-spike.md",
                    })

            # 4. AsyncUpdater refresh queue lag or background errors (severity: ticket)
            oldest_age = self._use_updater["oldest_task_age_s"]
            upd_errors = self._use_updater["errors"]
            if oldest_age > thresholds["refresh_lag_seconds_max"] or upd_errors > 0:
                alerts.append({
                    "name": "ragi_refresh_lag",
                    "severity": "ticket",
                    "value": round(oldest_age, 3),
                    "errors": upd_errors,
                    "threshold": thresholds["refresh_lag_seconds_max"],
                    "summary": "Background refresh lag {:.1f}s (errors={})".format(
                        oldest_age, upd_errors
                    ),
                    "runbook": "docs/runbooks/ragi-refresh-lag.md",
                })

        return alerts

    def get_spans(self, trace_id: Optional[str] = None, request_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Query recorded spans, optionally filtered by traceId or requestId."""
        with self._lock:
            spans = list(self._spans)
        if trace_id:
            spans = [s for s in spans if s.get("traceId") == trace_id]
        if request_id:
            spans = [s for s in spans if s.get("requestId") == request_id]
        return spans

    def get_logs(self, request_id: Optional[str] = None, level: Optional[str] = None) -> List[Dict[str, Any]]:
        """Query recent structured log events, optionally filtered by requestId or level."""
        with self._lock:
            logs = list(self._logs)
        if request_id:
            logs = [entry for entry in logs if entry.get("requestId") == request_id]
        if level:
            logs = [entry for entry in logs if entry.get("level") == level.lower()]
        return logs

    def snapshot(self) -> Dict[str, Any]:
        """Return a complete dashboard-ready RED/USE/Trace/Alert snapshot."""
        with self._lock:
            elapsed_s = max(0.001, time.monotonic() - self._started_at)
            red_ops: Dict[str, Any] = {}
            for op, hist in self._op_histograms.items():
                reqs = self._op_requests.get(op, 0)
                errs = self._op_errors.get(op, 0)
                snap = hist.snapshot()
                red_ops[op] = {
                    "requests": reqs,
                    "errors": errs,
                    "rate_per_sec": round(float(reqs) / elapsed_s, 4),
                    "error_rate": round(float(errs) / float(max(1, reqs)), 4),
                    "p50_ms": snap["p50_ms"],
                    "p95_ms": snap["p95_ms"],
                    "p99_ms": snap["p99_ms"],
                }

            series_list: List[Dict[str, Any]] = []
            for (op, ep, sc, prov), hist in self._series_histograms.items():
                h_snap = hist.snapshot()
                series_list.append({
                    "labels": {
                        "operation": op,
                        "entry_point": ep,
                        "status_class": sc,
                        "provider": prov,
                    },
                    "count": h_snap["count"],
                    "p50_ms": h_snap["p50_ms"],
                    "p95_ms": h_snap["p95_ms"],
                    "p99_ms": h_snap["p99_ms"],
                    "buckets": h_snap["buckets"],
                })

            q_tot = self._quality["total_queries"]
            q_emp = self._quality["empty_results"]
            quality_snap = {
                "total_queries": q_tot,
                "empty_results": q_emp,
                "empty_result_rate": round(float(q_emp) / float(max(1, q_tot)), 4) if q_tot else 0.0,
            }

            recent_spans = list(self._spans)[-20:]
            use_snap = {
                "bm25_cache": dict(self._use_bm25),
                "updater_queue": dict(self._use_updater),
            }

        return {
            "red": red_ops,
            "series": series_list,
            "use": use_snap,
            "quality": quality_snap,
            "traces": recent_spans,
            "alerts": self.evaluate_alerts(),
        }

    def reset(self) -> None:
        """Reset collected in-memory metrics, spans, and logs."""
        with self._lock:
            self._started_at = time.monotonic()
            self._series_histograms.clear()
            self._op_histograms.clear()
            self._op_requests.clear()
            self._op_errors.clear()
            self._spans.clear()
            self._logs.clear()
            self._use_bm25.update({
                "utilization": 0.0,
                "cache_size": 0,
                "hits": 0,
                "misses": 0,
                "saturation_evictions": 0,
                "errors": 0,
            })
            self._use_updater.update({
                "utilization": 0.0,
                "queue_depth": 0,
                "oldest_task_age_s": 0.0,
                "saturation_queued": 0,
                "errors": 0,
            })
            self._quality.update({"total_queries": 0, "empty_results": 0})


_DEFAULT_COLLECTOR = TelemetryCollector()


def get_telemetry() -> TelemetryCollector:
    """Return the active default TelemetryCollector."""
    return _DEFAULT_COLLECTOR

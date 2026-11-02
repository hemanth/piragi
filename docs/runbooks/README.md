# Piragi Observability & On-Call Runbooks

## On-Call Questions & Signal Mapping

Every telemetry signal emitted by `piragi` maps directly to one of four operational questions:

| # | On-Call Question | Metric ("That") | Trace ("Where") | Structured Log ("Why") |
|---|---|---|---|---|
| 1 | **Are queries (`ask`/`retrieve`) or ingestions failing, and at which stage?** | `red.<op>.error_rate` by bounded `(operation, entry_point, status_class, provider)` | Span tree (`ragi.ask` -> `search.variations`, `search.hybrid`, `rerank.cross_encoder`, `llm.complete`) with `status="error"` | `operation_failed` JSON event with `requestId`, `entryPoint`, `errorType`, `statusClass` |
| 2 | **Where is latency going when `ask` or `retrieve` p99 exceeds SLO (>2000ms)?** | `red.<op>.p50_ms`, `p95_ms`, `p99_ms` latency histograms | Child span `duration_ms` breakdown per hop | `slow_operation` / `operation_completed` with `durationMs` and candidate counts |
| 3 | **Are external LLM / Vertex AI providers throttling, timing out, or retrying?** | `red.llm_complete` by `provider` and `status_class` (`rate_limit`, `timeout`, `5xx`) | `llm.complete` span with `attempt` count and outbound `X-Request-ID` / `traceparent` | `llm_retry` (`warn`) and `llm_request_failed` (`error`) with `attempt`, `delayMs`, `provider` |
| 4 | **Are caches saturating, background updates lagging, or queries returning zero evidence?** | `use.bm25_cache`, `use.updater_queue` (`queue_depth`, `oldest_task_age_s`), `quality.empty_result_rate` | `ragi.refresh` span (`entry_point="background_updater"`) | `bm25_cache_evicted`, `updater_refresh_failed`, `empty_retrieval` |

## Active Symptom-Based Alerts

Only two severities exist (`page` = user-facing impact now; `ticket` = degradation to address this week):

- [`ragi-high-error-rate.md`](./ragi-high-error-rate.md) — **page**: User-facing error rate > 1%
- [`ragi-high-p99-latency.md`](./ragi-high-p99-latency.md) — **page**: `ask` or `retrieve` p99 latency > 2000ms
- [`ragi-empty-retrieval-spike.md`](./ragi-empty-retrieval-spike.md) — **ticket**: Empty citation rate > 15% across >= 10 queries
- [`ragi-refresh-lag.md`](./ragi-refresh-lag.md) — **ticket**: `AsyncUpdater` oldest queued refresh task > 600s or worker refresh errors > 0

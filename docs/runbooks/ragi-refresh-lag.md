# Runbook: ragi_refresh_lag (Severity: ticket)

**Means:** Background `AsyncUpdater` refresh tasks are queued longer than 600s (`oldest_task_age_s > 600`) or background refresh workers recorded errors, leaving indexed documents stale.
**First check:** Inspect `kb.stats()["use"]["updater_queue"]` (`queue_depth`, `oldest_task_age_s`, `errors`) and filter structured logs for `entryPoint="background_updater"` and `event="updater_refresh_failed"` to check for unreachable source URLs or deleted files.
**Escalate to:** Data Ingestion / Platform team (`#rag-oncall`) within current sprint; increase `auto_update.workers` if queue saturation persists under healthy sources.

# Runbook: ragi_empty_retrieval_spike (Severity: ticket)

**Means:** More than 15% of `retrieve` / `ask` calls are returning zero citations (or abstaining with `NO_EVIDENCE`), indicating corpus drift, overly strict metadata filters, or an unindexed store.
**First check:** Check `kb.count()` and `kb.stats()["quality"]`, then inspect structured `warn` logs with `event="empty_retrieval"` by `entryPoint` (`filtered_api` vs `sync_api` / `async_api`) to see if a specific filter or empty index is responsible.
**Escalate to:** Knowledge Base / Search Quality owner (`#search-quality`) within current sprint.

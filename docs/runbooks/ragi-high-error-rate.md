# Runbook: ragi_high_error_rate (Severity: page)

**Means:** More than 1% of user-facing `ask`, `retrieve`, or `ingest` operations are failing, impacting callers directly.
**First check:** Inspect `kb.stats()["red"]` for the failing `operation`, `provider`, and `status_class` (`rate_limit`, `timeout`, `5xx`), then filter structured JSON logs by `"level":"error"` and follow `requestId` + `entryPoint` across spans to pinpoint whether the failure is in `llm_complete`, `vector_search`, or `ingest`.
**Escalate to:** AI Platform / RAG Engineering on-call (`#rag-oncall`), or upstream LLM/Vector Store provider status if `provider` is `openai`, `vertexai`, `postgres`, `pinecone`, or `qdrant`.

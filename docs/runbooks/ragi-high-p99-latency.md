# Runbook: ragi_high_p99_latency (Severity: page)

**Means:** Tail latency (`p99_ms`) on `ask` or `retrieve` exceeded the 2000ms SLO threshold, causing slow user responses or upstream timeouts.
**First check:** Compare `p99_ms` across child operations in `kb.stats()["red"]` (`query_transform`, `embed`, `vector_search`, `hybrid_fusion`, `rerank`, `llm_complete`) and inspect the slowest trace in `kb.stats()["traces"]` to identify the bottleneck stage.
**Escalate to:** AI Platform / RAG Engineering on-call (`#rag-oncall`). If `rerank` dominates CPU latency, verify GPU/MPS device placement or reduce `candidate_k`; if `llm_complete` dominates, check provider latency and retry warnings (`event="llm_retry"`).

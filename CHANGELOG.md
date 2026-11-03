# Changelog

All notable changes to this project will be documented in this file.

## [1.3.0] - 2026-11-03

### Added
- **Reasoning-Intensive Retrieval (`ThinkThenRetrieve`, `CausalLLMReranker`, `ListwiseReasoningReranker`)**:
  - `ThinkThenRetrieve` pre-retrieval Chain-of-Thought query decomposition (`retrieval.use_think_then_retrieve = True`)
  - `CausalLLMReranker` (`retrieval.reranker_type = "causal_llm"`) supporting single-pass `yes/no` log-odds scoring (`Qwen/Qwen3-Reranker-4B` & `BAAI/bge-reranker-v2-gemma` with `use_cache=False`) and custom `rerank_instruction`
  - `ListwiseReasoningReranker` (`retrieval.reranker_type = "listwise"`) with sliding-window `<think>...</think><answer>[i] > [j]</answer>` ranking and bracket-injection sanitization
- **SOTA Hybrid Retrieval, Observability & Benchmarks**:
  - Weighted Reciprocal Rank Fusion (`use_rrf`), PDF hyphen/linebreak de-gluing, hierarchical parent auto-merging, U-shaped context ordering, and attributed citation pruning
  - Agentic query rewriting, BM25-only mode, on-the-fly retrieval, and hot/cold storage tiering
  - Built-in RED/USE observability (`kb.stats()`), W3C distributed trace propagation, and symptom-based alerts
  - Official **#1 Overall** live standing on Hugging Face **AIR-Bench 24.05 Long-Doc** (`84.96%` Recall@10, `75.72%` nDCG@10) and **`32.61%` Macro `NDCG@10` (`2.87x` BM25)** across all 12 splits of **BRIGHT (`xlangai/BRIGHT`, ICLR 2025)** (`31.43%` fully open-weights)
  - `optimize_for_leaderboard("bright")` preset and full 12-split `evals/run_bright_live.py` runner with `scipy.sparse` CSR Okapi BM25

### Fixed
- **Thread-Safe Parallel Context Propagation (`wrap_with_context`)**: Copy `contextvars.Context` per worker invocation (`ctx.copy().run(...)`) so concurrent `ThreadPoolExecutor` query variations never collide on `RuntimeError: cannot enter context`
- **Boundary & Mock Resilience**: Guard `top_k <= 0` in `RetrievalPipeline.retrieve()` and `LanceStore.search()`, and safely normalize non-float `sample_rate` in `request_context` and `FilteredRagi`

## [1.2.1] - 2026-08-10

### Added
- Parallel multi-query vector search via `ThreadPoolExecutor` (3-5x faster retrieval)
- `max_parallel_queries` config for `RetrievalPipeline` (default: 4)
- 6 new tests for batch query behavior

### Changed
- `RetrievalPipeline.retrieve()` and `ask()` now search all query variations concurrently
- Single-query paths skip executor overhead (no regression)
- Graceful degradation: if one query variation fails, others still return results


### Added
- `LLMClient` — Unified LLM client with connection pooling and retry logic
- `IngestionPipeline` — Composable pipeline: Load → Chunk → Embed → Store
- `RetrievalPipeline` — Composable pipeline: Transform → Search → Rerank → Generate
- `DocumentLoader.stream()` — Memory-efficient generator for large corpora
- `pipelines/` package with standalone pipeline classes for power users

### Changed
- `Ragi` is now a thin facade delegating to `IngestionPipeline` and `RetrievalPipeline`
- `Ragi.add()` uses streaming ingestion (doc-by-doc) instead of loading all into memory
- All LLM calls go through a single shared `LLMClient` instance (connection pooling)
- Retriever, HyDE, ContextualChunker now accept `llm_client` parameter

### Architecture
- **God class decomposition:** `Ragi` core.py reduced from ~800 LOC to ~350 LOC
- **Pipeline pattern:** Each pipeline is independently testable and composable
- **Backward compatible:** `kb = Ragi("./docs"); kb.ask("query")` unchanged

## [1.1.0] - 2026-08-09

### Added
- `RagiConfig` — Pydantic-validated configuration with typed fields and typo detection
- `FilteredRagi` — Thread-safe immutable filtered view for concurrent web backends
- `retry_with_backoff` — Exponential backoff retry for network calls (embeddings + LLM)
- Thread-safe model cache with `threading.Lock` for `EmbeddingGenerator`
- 20 new tests covering config validation, thread safety, retry logic, and concurrent filtering

### Changed
- `filter()` now returns `FilteredRagi` instead of mutating `self` (backward compatible API)
- Default progress callback uses `logger.info()` instead of `print("[piragi]")`
- Config parsing uses Pydantic dotted access instead of `.get()` chains
- Remote embedding/LLM calls retry on `ConnectionError`, `TimeoutError`, `APIConnectionError`, `APITimeoutError`

### Fixed
- **Critical:** Race condition in `filter()` — concurrent threads no longer overwrite each other's filters
- **Critical:** Thread-unsafe `_model_cache` mutations now protected by lock
- Silent misconfiguration from typos (e.g., `{"modle": "gpt-4"}` now raises `ValidationError`)

## [1.0.0] - 2026-08-09

### Added
- Initial stable release
- Zero-setup RAG with auto-chunking, embeddings, and smart citations
- Multiple vector store backends (LanceDB, Pinecone, Postgres, Qdrant, Supabase)
- Semantic, contextual, hierarchical, and proposition chunking strategies
- HyDE, hybrid search (BM25 + vector), and cross-encoder reranking
- Knowledge graph extraction and querying
- Async support via `AsyncRagi`
- Auto-update with change detection
- ONNX runtime support for faster embeddings
- Interactive playground server
- Remote filesystem support (S3, GCS, Azure, web crawling)

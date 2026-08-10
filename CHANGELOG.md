# Changelog

All notable changes to this project will be documented in this file.

## [1.2.0] - 2026-08-10

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

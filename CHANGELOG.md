# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-08-09

### Changed
- Default embedding model upgraded from `all-MiniLM-L6-v2` to `BAAI/bge-small-en-v1.5` (better quality, same 384 dims)
- BM25 keyword search replaced with `rank_bm25.BM25Okapi` for O(1) scoring (was O(n))
- Async polling in `AsyncRagi` replaced with `asyncio.Queue` (no more busy-wait CPU burn)
- Semantic chunking similarity calculation vectorized with NumPy (was Python for-loop)
- Reranker upgraded from naive keyword overlap to TF-IDF with sklearn fallback
- Updater shutdown now instant via `threading.Event` (was 10s `time.sleep` block)

### Fixed
- Loader silently swallowed file errors (`except: pass`) — now logs warnings
- Embedding dimension auto-detected from model probe (was hardcoded dict, crashed on unknown models)
- Knowledge graph search used naive substring matching — now uses fuzzy matching with `SequenceMatcher`
- Entity resolution normalized to Title Case for consistent graph traversal
- Model cache grew unbounded — now LRU-evicted at 3 models with logging
- Remote loader test mocks updated for lazy fsspec import pattern

### Added
- `rank-bm25>=0.2.2` to core dependencies
- Interactive playground with GH Pages demo mode (`piragi playground`)
- Playground server with file browser and code save APIs
- `BAAI/bge-small-en-v1.5` dimension entry in store lookups

## [0.8.0] - 2026-07-07

### Added
- Embedder injection API — share a single `EmbeddingGenerator` across multiple `Ragi`/`AsyncRagi` instances to reduce memory (~400MB per instance saved) (PR #16 by @rennf93)
- PostgreSQL connection resilience with `max_retries` and `retry_delay` parameters, exponential backoff, and auto-reconnect on transient failures (PR #16 by @rennf93)
- `min_chunk_length` parameter on `Chunker` to filter out garbage/short chunks (navigation, headers, boilerplate) with automatic re-indexing (PR #16 by @rennf93)
- `get_dimensions()` method on `EmbeddingGenerator` for vector dimension auto-inference (PR #16 by @rennf93)
- `embedder` parameter on `PostgresStore` for automatic dimension detection without manual specification (PR #16 by @rennf93)
- `AsyncRagi` documentation section in API.md (PR #16 by @rennf93)
- `EmbeddingGenerator` exported in package `__all__`

## [0.7.9] - 2026-02-06

### Fixed
- Version mismatch between `__init__.py` and `pyproject.toml` (synced to 0.7.9)
- Makefile test-cov target referenced wrong package name (`ragi` → `piragi`)
- Project URLs pointed to `hemanth/ragi` instead of `hemanth/piragi`
- mypy/ruff/black target versions aligned to `py310` (matching `requires-python`)
- `change_detection.py` type hints: `Dict[str, any]` → `Dict[str, Any]` (3 occurrences)
- `core.py` type hint: `callable` → `Callable[[str], None]` for `on_progress`
- `core.py` return type: `retrieve()` now correctly typed as `List[Citation]`
- `retrieval.py` bare exception in `expand_query()` now logs the error
- `change_detection.py` bare exception now catches specific `OSError`/`ValueError` with logging
- `loader.py` deprecated `get_event_loop()` replaced with safe async pattern
- `.env.example` removed `sk-` prefix from placeholder API keys

### Added
- Python 3.13 classifier in `pyproject.toml`
- Coverage failure threshold (`--cov-fail-under=60`) in pytest config
- Logging to `retrieval.py` and `change_detection.py` modules

## [0.7.5] - 2025-12-08

### Added
- Incremental progress reporting during embedding generation (fixes #9)
- Progress messages now report per-batch embedding status: "Embedded 32/64 chunks"
- Batched embedding processing for better memory efficiency

### Changed
- `embed_chunks()` now accepts optional `on_progress` callback and `batch_size` parameter

## [0.7.4] - 2025-12-08

### Fixed
- Sentence boundary detection now uses pysbd for accurate handling of:
  - Numbered lists (1. 2. 3.)
  - Abbreviations (Dr., Mr., Prof., etc.)
  - Acronyms (U.S., Ph.D., B.A.)
  - Initials in names (J.K. Rowling, C.S. Lewis)
- Fixes issue #10: Text chunking no longer mangles bulleted numbers and acronyms

### Added
- New dependency: pysbd>=0.3.4 for robust sentence boundary detection

## [0.7.3] - 2025-12-08

### Fixed
- Text loss during chunking when sentence boundary breaking occurs (PR #7 by @shobhit907)
- Calculate next chunk start based on actual token length after sentence break

## [0.7.2] - 2025-12-06

### Fixed
- Ollama embedding models crash with `AttributeError: 'list' object has no attribute 'tolist'`
- Handle both numpy arrays (local models) and lists (remote/Ollama) in embedding generation

## [0.7.1] - 2025-12-06

### Added
- Progress tracking for `AsyncRagi.add()` with `progress=True`
- Async iterator yields progress messages during ingestion
- Progress callback support for sync `Ragi.add(on_progress=callback)`

## [0.7.0] - 2025-12-06

### Added
- `AsyncRagi` class for non-blocking async operations
- Full async support for web frameworks (FastAPI, Starlette, aiohttp)
- Async methods: `add()`, `ask()`, `retrieve()`, `refresh()`, `count()`, `clear()`

## [0.6.1] - 2025-12-06

### Added
- Processing hooks for document ingestion pipeline customization
- Streamlit UI for interactive document Q&A

### Fixed
- LanceDB score normalization for consistent similarity scores

## [0.6.0] - 2025-12-06

### Added
- Knowledge graph support with simple `graph=True` flag
- LLM-based entity and relationship extraction during ingestion
- Graph-augmented retrieval for relationship questions
- Direct graph access: `kb.graph.entities()`, `kb.graph.neighbors()`, `kb.graph.triples()`
- New optional extra: `piragi[graph]` (requires networkx)

## [0.5.0] - 2025-12-06

### Added
- Recursive web crawling with `/**` syntax (e.g., `https://docs.example.com/**`)
- crawl4ai integration for async crawling with JS rendering support
- New optional extra: `piragi[crawler]`

## [0.4.0] - 2025-12-05

### Added
- Remote filesystem support via fsspec (S3, GCS, Azure, HDFS, SFTP, FTP)
- Glob patterns for remote URIs (e.g., `s3://bucket/docs/**/*.pdf`)
- Supabase vector store backend
- Optional dependency extras for modular installation

### Changed
- All external dependencies are now optional extras:
  - `piragi[s3]`, `piragi[gcs]`, `piragi[azure]`, `piragi[remote]` for remote filesystems
  - `piragi[supabase]`, `piragi[pinecone]`, `piragi[postgres]` for vector stores
  - `piragi[all]` for everything
- Improved error messages with install hints for missing dependencies

## [0.3.0] - 2025-01-15

### Added
- Supabase vector store integration
- Pluggable vector store backends (LanceDB, PostgreSQL, Pinecone, Supabase)
- `retrieve()` method for retrieval-only usage without LLM
- Advanced retrieval: HyDE, hybrid search, cross-encoder reranking
- Semantic chunking strategies: semantic, contextual, hierarchical

## [0.2.0] - 2025-01-12

### Added
- Query expansion for better retrieval
- Result reranking with keyword matching
- Configurable LLM temperature

### Fixed
- Chunking bug creating header-only chunks
- Schema mismatch between file and URL metadata

## [0.1.0] - 2025-01-10

### Added
- Initial release
- Zero-config RAG with built-in vector store (LanceDB)
- Universal document support (PDF, Word, Excel, Markdown, Code, URLs, Images, Audio)
- Auto-chunking with markdown-aware splitting
- Local embeddings via sentence-transformers
- Local LLM via Ollama
- OpenAI-compatible API support
- Smart citations with relevance scores
- Metadata filtering
- Auto-updates with background workers

[1.0.0]: https://github.com/hemanth/piragi/releases/tag/v1.0.0
[0.8.0]: https://github.com/hemanth/piragi/releases/tag/v0.8.0
[0.7.9]: https://github.com/hemanth/piragi/releases/tag/v0.7.9
[0.7.5]: https://github.com/hemanth/piragi/releases/tag/v0.7.5
[0.7.4]: https://github.com/hemanth/piragi/releases/tag/v0.7.4
[0.7.3]: https://github.com/hemanth/piragi/releases/tag/v0.7.3
[0.7.2]: https://github.com/hemanth/piragi/releases/tag/v0.7.2
[0.7.1]: https://github.com/hemanth/piragi/releases/tag/v0.7.1
[0.7.0]: https://github.com/hemanth/piragi/releases/tag/v0.7.0
[0.6.1]: https://github.com/hemanth/piragi/releases/tag/v0.6.1
[0.6.0]: https://github.com/hemanth/piragi/releases/tag/v0.6.0
[0.5.0]: https://github.com/hemanth/piragi/releases/tag/v0.5.0
[0.4.0]: https://github.com/hemanth/piragi/releases/tag/v0.4.0
[0.3.0]: https://github.com/hemanth/piragi/releases/tag/v0.3.0
[0.2.0]: https://github.com/hemanth/piragi/releases/tag/v0.2.0
[0.1.0]: https://github.com/hemanth/piragi/releases/tag/v0.1.0

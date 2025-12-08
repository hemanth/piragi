# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

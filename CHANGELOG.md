# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.2] - 2025-01-10

### Changed
- **BREAKING**: Changed default embedding model from `nvidia/llama-embed-nemotron-8b` (~8GB) to `all-mpnet-base-v2` (~420MB)
- Prevents memory crashes on standard machines while maintaining good quality
- Users can still opt into larger models via config

### Fixed
- Memory exhaustion when processing large websites
- Crash on Macs with <16GB RAM when loading default embedding model

## [0.1.1] - 2025-01-10

### Changed
- Renamed package module from `ragi` to `piragi`
- Import is now `from piragi import Ragi` (class name stays `Ragi`)
- Updated default persist directory from `.ragi` to `.piragi`

## [0.1.0] - 2025-01-10

### Added
- Initial release of piragi
- Zero-config RAG with built-in vector store (LanceDB)
- Universal document support (PDF, Word, Excel, Markdown, Code, URLs, Images, Audio)
- Auto-chunking with markdown-aware splitting
- Local embeddings via sentence-transformers (nvidia/llama-embed-nemotron-8b)
- Remote embeddings via OpenAI-compatible APIs
- Local LLM via Ollama (llama3.2)
- OpenAI-compatible LLM support
- Smart citations with relevance scores
- Metadata filtering
- Auto-updates with background workers
- Change detection for files (mtime + hash) and URLs (HTTP HEAD)
- Concurrent query support
- Single unified config dict
- Examples: quickstart, ollama, code_qa, multi_format, embedding_options, update_documents
- Comprehensive API documentation

### Features
- **Simple Setup** - Works with free local models out of the box
- **All Formats** - PDF, Word, Excel, Markdown, Code, URLs, Images, Audio
- **Auto-Updates** - Background refresh, queries never blocked
- **Smart Citations** - Every answer includes ranked source citations
- **OpenAI Compatible** - Drop-in support for any OpenAI-compatible API

[0.1.2]: https://github.com/hemanth/ragi/releases/tag/v0.1.2
[0.1.1]: https://github.com/hemanth/ragi/releases/tag/v0.1.1
[0.1.0]: https://github.com/hemanth/ragi/releases/tag/v0.1.0

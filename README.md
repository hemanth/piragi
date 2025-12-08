# piragi

**The best RAG interface yet.**

```python
from piragi import Ragi

kb = Ragi(["./docs", "s3://bucket/data/**/*.pdf", "https://api.example.com/docs"])
answer = kb.ask("How do I deploy this?")
```

Built-in vector store, embeddings, citations, and auto-updates. Free & local by default.

## Installation

```bash
pip install piragi

# Optional: Install Ollama for local LLM
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2

# Optional extras
pip install piragi[s3]       # S3 support
pip install piragi[gcs]      # Google Cloud Storage
pip install piragi[azure]    # Azure Blob Storage
pip install piragi[crawler]  # Recursive web crawling
pip install piragi[graph]    # Knowledge graph
pip install piragi[postgres] # PostgreSQL/pgvector
pip install piragi[pinecone] # Pinecone
pip install piragi[supabase] # Supabase
pip install piragi[all]      # Everything
```

## Features

- **Zero Config** - Works with free local models out of the box
- **All Formats** - PDF, Word, Excel, Markdown, Code, URLs, Images, Audio
- **Remote Storage** - Read from S3, GCS, Azure, HDFS, SFTP with glob patterns
- **Web Crawling** - Recursively crawl websites with `/**` syntax
- **Auto-Updates** - Background refresh, queries never blocked
- **Smart Citations** - Every answer includes sources
- **Pluggable Stores** - LanceDB, PostgreSQL, Pinecone, Supabase, or custom
- **Advanced Retrieval** - HyDE, hybrid search, cross-encoder reranking
- **Semantic Chunking** - Context-aware and hierarchical chunking
- **Knowledge Graph** - Entity/relationship extraction for better answers
- **Async Support** - Non-blocking API for web frameworks

## Quick Start

```python
from piragi import Ragi

# Local files
kb = Ragi("./docs")

# Multiple sources with globs
kb = Ragi(["./docs/*.pdf", "https://api.docs.com", "./code/**/*.py"])

# Remote filesystems
kb = Ragi("s3://bucket/docs/**/*.pdf")
kb = Ragi("gs://bucket/reports/*.md")

# Ask questions
answer = kb.ask("What is the API rate limit?")
print(answer.text)

# View citations
for cite in answer.citations:
    print(f"{cite.source}: {cite.score:.0%}")
```

## Remote Filesystems

Read files from cloud storage using glob patterns:

```python
# S3
kb = Ragi("s3://my-bucket/docs/**/*.pdf")

# Google Cloud Storage
kb = Ragi("gs://my-bucket/reports/*.md")

# Azure Blob Storage
kb = Ragi("az://my-container/files/*.txt")

# Mix local and remote
kb = Ragi([
    "./local-docs",
    "s3://bucket/remote-docs/**/*.pdf",
    "https://example.com/api-docs"
])
```

Requires optional extras: `pip install piragi[s3]`, `piragi[gcs]`, or `piragi[azure]`

## Web Crawling

Recursively crawl websites using `/**` suffix:

```python
# Crawl entire site
kb = Ragi("https://docs.example.com/**")

# Crawl specific section
kb = Ragi("https://docs.example.com/api/**")

# Mix with other sources
kb = Ragi([
    "./local-docs",
    "https://docs.example.com/**",
    "s3://bucket/data/*.pdf"
])
```

Crawls same-domain links up to depth 3, max 100 pages by default.

Requires: `pip install piragi[crawler]`

## Vector Store Backends

```python
from piragi import Ragi
from piragi.stores import PineconeStore, SupabaseStore

# LanceDB (default) - local or S3-backed
kb = Ragi("./docs")
kb = Ragi("./docs", store="s3://bucket/indices")

# PostgreSQL with pgvector
kb = Ragi("./docs", store="postgres://user:pass@localhost/db")

# Pinecone
kb = Ragi("./docs", store=PineconeStore(api_key="...", index_name="my-index"))

# Supabase
kb = Ragi("./docs", store=SupabaseStore(url="https://xxx.supabase.co", key="..."))
```

## Advanced Retrieval

```python
kb = Ragi("./docs", config={
    "retrieval": {
        "use_hyde": True,           # Hypothetical document embeddings
        "use_hybrid_search": True,  # BM25 + vector search
        "use_cross_encoder": True,  # Neural reranking
    }
})
```

## Chunking Strategies

```python
# Semantic - splits at topic boundaries
kb = Ragi("./docs", config={"chunk": {"strategy": "semantic"}})

# Hierarchical - parent-child for context + precision
kb = Ragi("./docs", config={"chunk": {"strategy": "hierarchical"}})

# Contextual - LLM-generated context per chunk
kb = Ragi("./docs", config={"chunk": {"strategy": "contextual"}})
```

## Knowledge Graph

Extract entities and relationships for better multi-hop reasoning:

```python
# Enable with single flag
kb = Ragi("./docs", graph=True)

# Automatic - extracts entities/relationships during ingestion
# Uses them to augment retrieval for relationship questions
answer = kb.ask("Who reports to Alice?")

# Direct graph access
kb.graph.entities()           # ["alice", "bob", "project x"]
kb.graph.neighbors("alice")   # ["bob", "engineering team"]
kb.graph.triples()            # [("alice", "manages", "bob"), ...]
```

Requires: `pip install piragi[graph]`

## Configuration

```python
config = {
    "llm": {
        "model": "llama3.2",
        "base_url": "http://localhost:11434/v1"
    },
    "embedding": {
        "model": "all-mpnet-base-v2",
        "batch_size": 32
    },
    "chunk": {
        "strategy": "fixed",
        "size": 512,
        "overlap": 50
    },
    "retrieval": {
        "use_hyde": False,
        "use_hybrid_search": False,
        "use_cross_encoder": False
    },
    "auto_update": {
        "enabled": True,
        "interval": 300
    }
}
```

## Async Support

Use `AsyncRagi` for non-blocking operations in async web frameworks:

```python
from piragi import AsyncRagi

kb = AsyncRagi("./docs")

# Simple await
await kb.add("./more-docs")
answer = await kb.ask("What is X?")

# With progress tracking
async for progress in kb.add("./large-docs", progress=True):
    print(progress)
    # "Discovering files..."
    # "Found 10 documents"
    # "Chunking 1/10: doc1.md"
    # ...
    # "Generating embeddings for 150 chunks..."
    # "Embedded 32/150 chunks"
    # "Embedded 64/150 chunks"
    # ...
    # "Embeddings complete"
    # "Done"

# With FastAPI
@app.post("/ingest")
async def ingest(files: list[str]):
    await kb.add(files)
    return {"status": "done"}
```

All methods are async: `add()`, `ask()`, `retrieve()`, `refresh()`, `count()`, `clear()`.

## Retrieval Only

Use piragi as a retrieval layer without LLM:

```python
chunks = kb.retrieve("How does auth work?", top_k=5)
for chunk in chunks:
    print(chunk.chunk, chunk.source, chunk.score)

# Use with your own LLM
context = "\n".join(c.chunk for c in chunks)
response = your_llm(f"Context:\n{context}\n\nQuestion: {query}")
```

## API

```python
# Sync API
kb = Ragi(sources, persist_dir=".piragi", config=None, store=None, graph=False)
kb.add("./more-docs")
kb.ask(query, top_k=5)
kb.retrieve(query, top_k=5)
kb.filter(**metadata).ask(query)
kb.refresh("./docs")
kb.count()
kb.clear()

# Async API (same methods, just await them)
kb = AsyncRagi(sources, persist_dir=".piragi", config=None, store=None, graph=False)
await kb.add("./more-docs")
await kb.ask(query, top_k=5)
```

Full docs: [API.md](API.md)

## License

MIT

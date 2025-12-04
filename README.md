# piragi

**The best RAG interface yet.**

```python
from piragi import Ragi

kb = Ragi(["./docs", "./code/**/*.py", "https://api.example.com/docs"])
answer = kb.ask("How do I deploy this?")
```

That's it. Built-in vector store, embeddings, citations, and auto-updates. Free & local by default.

---

## Installation

```bash
pip install piragi

# Optional: Install Ollama for local LLM
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2
```

---

## Features

- **Simple Setup** - Works with free local models out of the box
- **All Formats** - PDF, Word, Excel, Markdown, Code, URLs, Images, Audio
- **Auto-Updates** - Background refresh, queries never blocked
- **Smart Citations** - Every answer includes sources
- **OpenAI Compatible** - Drop-in support for any OpenAI-compatible API
- **Advanced Retrieval** - HyDE, hybrid search, cross-encoder reranking
- **Semantic Chunking** - Context-aware, proposition-based, and hierarchical chunking
- **Pluggable Storage** - Local, S3, PostgreSQL, Pinecone, or custom backends

---

## Examples

```python
# Basic
kb = Ragi("./docs")
answer = kb("What is this?")

# Multiple sources
kb = Ragi(["./docs/*.pdf", "https://api.docs.com", "./code/**/*.py"])

# OpenAI
kb = Ragi("./docs", config={
    "llm": {"model": "gpt-4o-mini", "api_key": "sk-..."},
    "embedding": {"model": "text-embedding-3-small", "api_key": "sk-..."}
})

# Filter
answer = kb.filter(file_type="pdf").ask("What's in the PDFs?")
```

---

## Advanced Retrieval

Enable state-of-the-art retrieval techniques for better answer quality:

```python
kb = Ragi("./docs", config={
    "retrieval": {
        "use_hyde": True,           # HyDE: generate hypothetical docs for better matching
        "use_hybrid_search": True,  # Combine BM25 + vector search
        "use_cross_encoder": True,  # Cross-encoder reranking for precision
    }
})
```

**Available techniques:**
- **HyDE** - Hypothetical Document Embeddings for improved semantic matching
- **Hybrid Search** - BM25 keyword matching + vector similarity with RRF fusion
- **Cross-Encoder Reranking** - Neural reranking for high precision results

---

## Chunking Strategies

Choose the chunking strategy that fits your documents:

```python
# Semantic chunking - splits at natural topic boundaries
kb = Ragi("./docs", config={
    "chunk": {"strategy": "semantic", "similarity_threshold": 0.5}
})

# Contextual chunking - LLM-generated context for each chunk
kb = Ragi("./docs", config={
    "chunk": {"strategy": "contextual"}
})

# Hierarchical chunking - parent-child relationships for context + precision
kb = Ragi("./docs", config={
    "chunk": {"strategy": "hierarchical", "parent_size": 2000, "child_size": 400}
})
```

**Available strategies:**
- **fixed** (default) - Simple token-based chunking with overlap
- **semantic** - Embedding-based splitting at topic boundaries
- **contextual** - Each chunk includes LLM-generated document context
- **hierarchical** - Large parent chunks for context, small children for retrieval

---

## Configuration

```python
# Defaults (all optional)
config = {
    "llm": {
        "model": "llama3.2",
        "base_url": "http://localhost:11434/v1"
    },
    "embedding": {
        "model": "all-mpnet-base-v2"  # ~420MB, good quality
        # For max quality: "nvidia/llama-embed-nemotron-8b" (~8GB)
        # For minimal: "all-MiniLM-L6-v2" (~90MB)
    },
    "chunk": {
        "strategy": "fixed",  # or "semantic", "contextual", "hierarchical"
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
        "interval": 300  # seconds
    }
}
```

---

## Auto-Updates

Changes detected and refreshed automatically in background. Zero query latency.

```python
kb = Ragi(["./docs", "https://api.docs.com"])
# That's it - auto-updates enabled by default

# Disable if needed
kb = Ragi("./docs", config={"auto_update": {"enabled": False}})
```

---

## Custom Storage Backends

Use your own infrastructure for production:

```python
from piragi import Ragi
from piragi.stores import PineconeStore, PostgresStore, LanceStore

# S3-backed storage (via LanceDB)
kb = Ragi("./docs", store="s3://my-bucket/indices")

# PostgreSQL with pgvector
kb = Ragi("./docs", store="postgres://user:pass@localhost/db")

# Pinecone
kb = Ragi("./docs", store=PineconeStore(
    api_key="your-api-key",
    index_name="my-index"
))

# Or use URI format
kb = Ragi("./docs", store="pinecone://my-index?api_key=...")
```

**Supported backends:**
- **LanceDB** (default) - Local or S3-backed, zero config
- **PostgreSQL** - pgvector extension for production databases
- **Pinecone** - Managed vector database

**Custom stores:**

```python
from piragi.stores import VectorStoreProtocol

class MyStore:
    def add_chunks(self, chunks): ...
    def search(self, query_embedding, top_k=5, filters=None): ...
    def delete_by_source(self, source): ...
    def count(self): ...
    def clear(self): ...
    def get_all_chunk_texts(self): ...

kb = Ragi("./docs", store=MyStore())
```

---

## Retrieval Only (No LLM)

Use piragi as a pure retrieval layer - bring your own LLM:

```python
from piragi import Ragi

kb = Ragi("./docs")

# Just get relevant chunks - no LLM involved
chunks = kb.retrieve("How does authentication work?", top_k=5)

for chunk in chunks:
    print(chunk.chunk)   # text content
    print(chunk.source)  # source file/url
    print(chunk.score)   # relevance score

# Use with your own LLM / framework
context = "\n".join(c.chunk for c in chunks)
response = your_llm(f"Based on:\n{context}\n\nAnswer: {query}")
```

Works with LangChain, LlamaIndex, direct API calls, or any framework.

---

## API

```python
kb = Ragi(sources, persist_dir=".piragi", config=None, store=None)
kb.add("./more-docs")
kb.ask(query, top_k=5)              # Full RAG (retrieval + LLM)
kb.retrieve(query, top_k=5)         # Retrieval only (no LLM)
kb(query)                           # Shorthand for ask()
kb.filter(**metadata).ask(query)
kb.filter(**metadata).retrieve(query)
kb.refresh("./docs")                # Force refresh sources
kb.count()
kb.clear()
```

**Advanced components (for custom pipelines):**

```python
from piragi import (
    # Vector stores
    VectorStoreProtocol, LanceStore, PostgresStore, PineconeStore,
    # Reranking
    CrossEncoderReranker, TFIDFReranker, HybridReranker,
    # Hybrid search
    BM25, HybridSearcher,
    # Query transformation
    HyDE, QueryExpander, MultiQueryRetriever, StepBackPrompting,
    # Chunking strategies
    SemanticChunker, ContextualChunker, PropositionChunker, HierarchicalChunker,
)
```

Full docs: [API.md](API.md)

---

MIT License | **piragi** = **P**owerful **I**nterface for **R**etrieval **A**ugmented **G**eneration **I**ntelligence

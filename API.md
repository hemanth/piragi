# API Reference

## Ragi

Main interface for RAG systems.

```python
from piragi import Ragi
```

### Constructor

```python
Ragi(
    sources: Union[str, List[str], None] = None,
    persist_dir: str = ".piragi",
    config: Optional[Dict[str, Any]] = None,
    store: Union[str, Dict, VectorStoreProtocol, None] = None,
)
```

**Parameters:**
- `sources` - File paths, URLs, glob patterns, remote URIs (s3://, gs://, az://), or crawl URLs (https://.../**)
- `persist_dir` - Directory to persist vector database
- `config` - Configuration dict (see Configuration section)
- `store` - Vector store backend (URI string, dict, or VectorStoreProtocol instance)

**Examples:**
```python
# Basic
kb = Ragi("./docs")

# Remote filesystem
kb = Ragi("s3://bucket/docs/**/*.pdf")

# Multiple sources
kb = Ragi(["./docs", "s3://bucket/data", "https://api.example.com"])

# With config
kb = Ragi("./docs", config={
    "llm": {"model": "gpt-4o-mini", "api_key": "sk-..."},
})

# Custom store
kb = Ragi("./docs", store="postgres://user:pass@localhost/db")
```

### Methods

#### `add(sources) -> Ragi`
Add documents to the knowledge base.

```python
kb.add("./more-docs")
kb.add(["./docs/*.pdf", "s3://bucket/files/*.md"])
```

#### `ask(query, top_k=5, system_prompt=None) -> Answer`
Ask a question and get an answer with citations.

```python
answer = kb.ask("How do I install this?")
print(answer.text)
print(answer.citations)
```

#### `retrieve(query, top_k=5) -> List[Citation]`
Retrieve relevant chunks without LLM generation.

```python
chunks = kb.retrieve("authentication")
for c in chunks:
    print(c.chunk, c.source, c.score)
```

#### `filter(**kwargs) -> Ragi`
Filter by metadata for the next query.

```python
answer = kb.filter(file_type="pdf").ask("What's in the PDFs?")
```

#### `refresh(sources) -> Ragi`
Force refresh specific sources.

```python
kb.refresh("./docs/api.md")
```

#### `count() -> int`
Number of chunks in the knowledge base.

#### `clear() -> None`
Clear all data.

## Data Types

### Answer
```python
answer.text        # Generated answer
answer.citations   # List[Citation]
answer.query       # Original query
```

### Citation
```python
citation.source    # File path or URL
citation.chunk     # Text content
citation.score     # Relevance score (0-1)
citation.metadata  # Dict of metadata
```

## Configuration

```python
config = {
    "llm": {
        "model": "llama3.2",
        "base_url": "http://localhost:11434/v1",
        "api_key": "not-needed",
        "temperature": 0.1,
    },
    "embedding": {
        "model": "all-mpnet-base-v2",
        "device": None,  # auto-detect
        "base_url": None,  # for remote embeddings
        "api_key": None,
    },
    "chunk": {
        "strategy": "fixed",  # fixed, semantic, contextual, hierarchical
        "size": 512,
        "overlap": 50,
    },
    "retrieval": {
        "use_hyde": False,
        "use_hybrid_search": False,
        "use_cross_encoder": False,
    },
    "auto_update": {
        "enabled": True,
        "interval": 300,
    },
}
```

## Vector Stores

### LanceDB (default)
```python
kb = Ragi("./docs")  # Local
kb = Ragi("./docs", store="s3://bucket/indices")  # S3-backed
```

### PostgreSQL
```python
kb = Ragi("./docs", store="postgres://user:pass@localhost/db")
```
Requires: `pip install piragi[postgres]`

### Pinecone
```python
from piragi.stores import PineconeStore

kb = Ragi("./docs", store=PineconeStore(
    api_key="...",
    index_name="my-index",
))
```
Requires: `pip install piragi[pinecone]`

### Supabase
```python
from piragi.stores import SupabaseStore

kb = Ragi("./docs", store=SupabaseStore(
    url="https://xxx.supabase.co",
    key="your-service-role-key",
))
```
Requires: `pip install piragi[supabase]`

### Custom Store
Implement `VectorStoreProtocol`:

```python
from piragi.stores import VectorStoreProtocol

class MyStore:
    def add_chunks(self, chunks: List[Chunk]) -> None: ...
    def search(self, query_embedding, top_k=5, filters=None) -> List[Citation]: ...
    def delete_by_source(self, source: str) -> int: ...
    def count(self) -> int: ...
    def clear(self) -> None: ...
    def get_all_chunk_texts(self) -> List[str]: ...

kb = Ragi("./docs", store=MyStore())
```

## Remote Filesystems

Supported schemes: `s3://`, `gs://`, `gcs://`, `az://`, `abfs://`, `hdfs://`, `sftp://`, `ftp://`

```python
# S3
kb = Ragi("s3://bucket/docs/**/*.pdf")

# Google Cloud Storage
kb = Ragi("gs://bucket/reports/*.md")

# Azure Blob
kb = Ragi("az://container/files/*.txt")

# Mix sources
kb = Ragi([
    "./local",
    "s3://bucket/remote/**/*.pdf",
    "https://example.com/docs"
])
```

Requires optional extras:
- `pip install piragi[s3]` for S3
- `pip install piragi[gcs]` for GCS
- `pip install piragi[azure]` for Azure
- `pip install piragi[remote]` for all

## Web Crawling

Recursively crawl websites using `/**` suffix:

```python
# Crawl entire site (same domain, max depth 3, max 100 pages)
kb = Ragi("https://docs.example.com/**")

# Crawl specific section
kb = Ragi("https://docs.example.com/api/**")
```

Features:
- Follows same-domain internal links only
- Respects max depth (default: 3) and max pages (default: 100)
- Uses crawl4ai with headless browser for JS rendering
- Returns markdown content for each page

Requires: `pip install piragi[crawler]`

## Supported File Formats

- **Documents:** PDF, Word (.docx), PowerPoint (.pptx), Excel (.xlsx)
- **Text:** Markdown, Plain text, Source code, HTML
- **Data:** JSON, XML, CSV
- **Media:** Images (OCR), Audio (transcription)
- **Web:** URLs
- **Archives:** ZIP
- **E-books:** EPUB

## Environment Variables

```bash
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=not-needed
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_SERVICE_KEY=your-key
```

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

## API

```python
kb = Ragi(sources, persist_dir=".piragi", config=None)
kb.add("./more-docs")
kb.ask(query, top_k=5)
kb(query)  # Shorthand
kb.filter(**metadata).ask(query)
kb.count()
kb.clear()
```

Full docs: [API.md](API.md)

---

MIT License | **piragi** = **R**etrieval **A**ugmented **G**eneration **I**nterface

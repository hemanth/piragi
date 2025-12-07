# Piragi UI

A Streamlit-based chat interface for Piragi with full configuration of chunking and retrieval strategies.

## Features

- 📁 **Document Upload** - Drag & drop files (txt, md, pdf, html, docx) or add URLs
- 💾 **Persistent Storage** - Uploads saved to `~/.piragi/uploads/`, survives restarts
- 🔄 **Re-indexing** - Change settings and re-process docs with one click
- 💬 **Chat Interface** - Ask questions, get grounded answers with citations
- ⚙️ **Full Configuration** - Chunking strategies, retrieval options, model selection
- 🔍 **Debug Mode** - See timing and active settings
- 🏠 **100% Local** - Runs entirely on your machine with Ollama

## Quick Start

```bash
# From the piragi directory
./ui/run.sh              # Uses "default" project
./ui/run.sh my-client    # Uses "my-client" project
```

Each project gets isolated storage in `~/.piragi/projects/<name>/`.
## Requirements

- Python 3.10+
- Ollama running locally with a model (e.g., `ollama pull llama3.2`)
- Piragi installed (`pip install -e .` from repo root)

## Configuration Options

### Chunking Strategies

| Strategy | Description |
|----------|-------------|
| **fixed** | Split by token count with overlap |
| **semantic** | Split at natural semantic boundaries |
| **hierarchical** | Parent/child chunks for context |
| **contextual** | LLM-determined boundaries |

### Retrieval Enhancements

| Option | Description |
|--------|-------------|
| **HyDE** | Generate hypothetical answer to improve search |
| **Hybrid Search** | Combine semantic + BM25 keyword matching |
| **Reranker** | Cross-encoder reranking for better accuracy |

## Screenshots

Upload docs → Ask questions → Get grounded answers with citations.

All processing happens locally in milliseconds.

---

## Demo Version (for Hugging Face Spaces)

There's also a `demo_app.py` designed for hosted demos:

```bash
# Set API key (HF or OpenAI)
export HF_TOKEN=your_token
# or
export OPENAI_API_KEY=your_key

# Run demo
streamlit run demo_app.py
```

**Differences from local version:**
- Uses HF Inference API or OpenAI (no Ollama required)
- Pre-loads piragi's own docs as sample content
- Ephemeral uploads (session-only, not persisted)
- Simpler UI focused on showcasing features

# Examples

Quick examples demonstrating piragi features.

## Basic

### quickstart.py
Load documents and ask questions.
```bash
python examples/quickstart.py
```

### ollama_example.py
Using piragi with Ollama (free local LLM).
```bash
python examples/ollama_example.py
```

### code_qa.py
Question-answering over a Python codebase.
```bash
python examples/code_qa.py
```

### multi_format.py
Working with multiple document formats.
```bash
python examples/multi_format.py
```

## Advanced

### embedding_options.py
Different embedding configurations.
```bash
python examples/embedding_options.py
```

### update_documents.py
Manual document refresh workflow.
```bash
python examples/update_documents.py
```

### auto_update_detection.py
Automatic change detection for files and URLs.
```bash
python examples/auto_update_detection.py
```

## Setup

```bash
pip install piragi

# For local LLM
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2

# For remote filesystems
pip install piragi[s3]  # or piragi[gcs], piragi[azure]
```

## Quick Template

```python
from piragi import Ragi

# Load from local, remote, or URLs
kb = Ragi([
    "./docs",
    "s3://bucket/data/**/*.pdf",
    "https://api.example.com/docs"
])

# Ask questions
answer = kb.ask("Your question")
print(answer.text)

# View citations
for cite in answer.citations:
    print(f"{cite.source} ({cite.score:.0%})")
```

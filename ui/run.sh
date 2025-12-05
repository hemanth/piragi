#!/bin/bash
# Quick start script for Piragi UI

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

echo "🚀 Starting Piragi UI..."

# Check for Ollama
if ! command -v ollama &> /dev/null; then
    echo "⚠️  Ollama not found. Install from https://ollama.ai"
    echo "   Then run: ollama pull llama3.2"
    exit 1
fi

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "⚠️  Ollama not running. Start it with: ollama serve"
    exit 1
fi

# Install piragi if needed
if ! python -c "import piragi" 2>/dev/null; then
    echo "📦 Installing piragi..."
    pip install -e "$REPO_DIR" -q
fi

# Install streamlit if needed
if ! python -c "import streamlit" 2>/dev/null; then
    echo "📦 Installing streamlit..."
    pip install streamlit -q
fi

# Run the app
echo "✅ Opening http://localhost:8501"
cd "$SCRIPT_DIR"
streamlit run app.py --server.headless true

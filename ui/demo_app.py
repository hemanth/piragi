"""Piragi Demo UI - Hosted version for Hugging Face Spaces.

This is a simplified version designed for public demos:
- Uses HF Inference API or OpenAI instead of local Ollama
- Pre-loads sample docs to showcase piragi
- Session-isolated uploads (each user gets their own)
- No persistent storage
"""

import os
import sys
import shutil
import tempfile
import logging
import time
import uuid
from pathlib import Path

import streamlit as st

# Configure logging (set to WARNING for production, INFO for debugging)
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from piragi import Ragi

# Page config
st.set_page_config(
    page_title="Piragi Demo",
    page_icon="📚",
    layout="wide",
)

# Initialize session-specific state
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.session_dir = Path(tempfile.mkdtemp(prefix=f"piragi_demo_{st.session_state.session_id[:8]}_"))
if "messages" not in st.session_state:
    st.session_state.messages = []
if "demo_kb" not in st.session_state:
    st.session_state.demo_kb = None
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

# Session-isolated directory
SESSION_DIR = st.session_state.session_dir

# Custom CSS
st.markdown("""
<style>
    .stApp { max-width: 100%; }
    .stMainBlockContainer { max-width: 65%; margin: 0 auto; }
</style>
""", unsafe_allow_html=True)


def get_llm_config():
    """Get LLM config - prefers HF_TOKEN, falls back to OPENAI_API_KEY."""
    hf_token = os.environ.get("HF_TOKEN", "")
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    
    if openai_key:
        return {
            "model": "gpt-4o-mini",
            "api_key": openai_key,
        }
    elif hf_token:
        return {
            "model": "meta-llama/Llama-3.2-3B-Instruct",
            "base_url": "https://api-inference.huggingface.co/v1",
            "api_key": hf_token,
        }
    else:
        # Fallback to local Ollama
        return {
            "model": os.environ.get("OLLAMA_MODEL", "llama3.2"),
            "base_url": "http://localhost:11434/v1",
            "api_key": "ollama",
        }


def get_kb_config():
    """Get KB config from session state."""
    strategy = st.session_state.get("chunk_strategy", "fixed")
    
    chunk_cfg = {"strategy": strategy}
    if strategy == "fixed":
        chunk_cfg["size"] = st.session_state.get("chunk_size", 512)
        chunk_cfg["overlap"] = st.session_state.get("chunk_overlap", 50)
    elif strategy == "semantic":
        chunk_cfg["similarity_threshold"] = st.session_state.get("similarity_threshold", 0.5)
        chunk_cfg["min_size"] = st.session_state.get("min_chunk_size", 100)
        chunk_cfg["max_size"] = st.session_state.get("max_chunk_size", 2000)
    elif strategy == "hierarchical":
        chunk_cfg["parent_size"] = st.session_state.get("parent_size", 2000)
        chunk_cfg["child_size"] = st.session_state.get("child_size", 400)
    
    return {
        "llm": get_llm_config(),
        "embedding": {"model": "all-MiniLM-L6-v2"},
        "chunk": chunk_cfg,
        "retrieval": {
            "use_hyde": st.session_state.get("use_hyde", False),
            "use_hybrid_search": st.session_state.get("use_hybrid", False),
            "use_cross_encoder": st.session_state.get("use_reranker", False),
        },
        "auto_update": {"enabled": False},
    }


def get_or_create_kb():
    """Get or create KB for this session, recreating if config changed."""
    config = get_kb_config()
    config_key = str(config.get("chunk", {}))
    
    # Recreate KB if chunking config changed
    if st.session_state.get("last_config_key") != config_key:
        if st.session_state.demo_kb is not None:
            st.session_state.demo_kb.clear()
            st.session_state.demo_kb = None
        st.session_state.last_config_key = config_key
    
    if st.session_state.demo_kb is None:
        st.session_state.demo_kb = Ragi(persist_dir=str(SESSION_DIR), config=config)
    
    return st.session_state.demo_kb


def load_sample_docs(kb):
    """Load piragi's main docs as sample content."""
    repo_root = Path(__file__).resolve().parent.parent
    sample_files = [
        ("README.md", repo_root / "README.md"),
        ("API.md", repo_root / "API.md"),
    ]
    
    loaded = []
    for name, filepath in sample_files:
        if filepath.exists() and filepath.stat().st_size > 1000:
            try:
                kb.add(str(filepath))
                loaded.append(name)
            except Exception as e:
                logger.error(f"Failed to load {name}: {e}")
    return loaded


def main():
    st.title("📚 Piragi Demo")
    st.caption("Zero-setup RAG with smart citations • [GitHub](https://github.com/hemanth/piragi)")

    kb = get_or_create_kb()

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Learn more expander
        with st.expander("📖 What do these settings mean?"):
            st.markdown("""
**Chunking Strategies**

How documents are split into searchable pieces:

- **Fixed**: Split by token count (e.g., every 512 tokens). Simple and fast.
- **Semantic**: Split at natural boundaries (paragraphs, topic shifts). Better for prose.
- **Hierarchical**: Creates parent/child chunks. Children are searched, parents provide context.

**Retrieval Enhancements**

Ways to improve search accuracy:

- **HyDE** (Hypothetical Document Embeddings): Generates a hypothetical answer first, then searches for similar content. Helps with abstract queries.
- **Hybrid Search**: Combines semantic search (meaning) with BM25 (keywords). Best of both worlds.
- **Reranker**: Uses a cross-encoder model to re-score results. Slower but more accurate.

**Top K**

How many chunks to retrieve. More = broader context but potentially more noise.
            """)
        
        st.divider()
        
        # Chunking strategy
        strategy = st.selectbox(
            "Chunking Strategy",
            options=["fixed", "semantic", "hierarchical"],
            key="chunk_strategy",
            help="How to split documents into chunks. Note: 'contextual' (LLM-based) is disabled in demo to avoid rate limits.",
        )
        
        if strategy == "fixed":
            col1, col2 = st.columns(2)
            with col1:
                st.number_input("Chunk size", min_value=100, max_value=2000, value=512, step=100, key="chunk_size")
            with col2:
                st.number_input("Overlap", min_value=0, max_value=200, value=50, step=10, key="chunk_overlap")
        elif strategy == "semantic":
            st.slider("Similarity threshold", 0.0, 1.0, 0.5, 0.05, key="similarity_threshold")
        elif strategy == "hierarchical":
            col1, col2 = st.columns(2)
            with col1:
                st.number_input("Parent size", value=2000, key="parent_size")
            with col2:
                st.number_input("Child size", value=400, key="child_size")
        
        st.subheader("Retrieval")
        st.checkbox("HyDE", key="use_hyde", help="Hypothetical Document Embeddings")
        st.checkbox("Hybrid Search", key="use_hybrid", help="Semantic + BM25")
        st.checkbox("Reranker", key="use_reranker", help="Cross-encoder reranking")
        st.number_input("Top K", min_value=1, max_value=20, value=5, key="top_k")
        
        st.divider()
        st.header("📁 Documents")
        
        # Load sample docs button
        if st.button("📄 Load Sample Docs", type="primary", use_container_width=True):
            with st.spinner("Loading piragi docs..."):
                loaded = load_sample_docs(kb)
                if loaded:
                    st.session_state.loaded_sample_docs = loaded
                    st.success(f"Loaded: {', '.join(loaded)}")
                else:
                    st.warning("No sample docs found")
            st.rerun()
        
        # Show what sample docs contain
        if st.session_state.get("loaded_sample_docs"):
            st.caption("📚 Loaded: Piragi README & API docs")
        
        # File uploader (ephemeral)
        uploaded_files = st.file_uploader(
            "Or upload your own",
            type=["txt", "md", "pdf", "html", "docx"],
            accept_multiple_files=True,
        )
        
        if uploaded_files:
            for file in uploaded_files:
                if file.name not in st.session_state.uploaded_files:
                    # Save to session-isolated temp dir and add
                    tmp_path = SESSION_DIR / file.name
                    tmp_path.write_bytes(file.read())
                    try:
                        kb.add(str(tmp_path))
                        st.session_state.uploaded_files.append(file.name)
                        st.success(f"✅ {file.name}")
                    except Exception as e:
                        st.error(f"❌ {file.name}: {e}")
        
        # Stats
        st.divider()
        chunk_count = kb.count()
        st.metric("Chunks indexed", chunk_count)
        
        # Reset button
        if chunk_count > 0:
            if st.button("🗑️ Reset Demo", type="secondary", use_container_width=True):
                kb.clear()
                # Clean up session temp files
                shutil.rmtree(SESSION_DIR, ignore_errors=True)
                SESSION_DIR.mkdir(parents=True, exist_ok=True)
                # Reset state
                st.session_state.messages = []
                st.session_state.uploaded_files = []
                st.session_state.pop("loaded_sample_docs", None)
                st.session_state.demo_kb = None
                st.rerun()

    # Main chat area
    if kb.count() == 0:
        st.info("👈 Load sample docs or upload your own to get started!")
        
        # Show what piragi can do
        st.markdown("""
        ### What is Piragi?
        
        Piragi is a zero-setup RAG (Retrieval-Augmented Generation) library that makes it easy to:
        
        - 📄 **Load documents** - PDFs, Markdown, HTML, DOCX, URLs
        - 🔍 **Smart chunking** - Fixed, semantic, or hierarchical strategies
        - 🎯 **Accurate retrieval** - HyDE, hybrid search, reranking
        - 💬 **Grounded answers** - Every response cites its sources
        
        Try loading the sample docs to see it in action!
        """)
        return
    
    # Show example questions if sample docs loaded and no messages yet
    if st.session_state.get("loaded_sample_docs") and len(st.session_state.messages) == 0 and "pending_question" not in st.session_state:
        st.markdown("### 💡 Try asking:")
        example_questions = [
            "How do I get started with piragi?",
            "What chunking strategies are available?",
            "How do I use Pinecone or PostgreSQL?",
            "What is HyDE and how does it work?",
        ]
        cols = st.columns(2)
        for i, q in enumerate(example_questions):
            with cols[i % 2]:
                if st.button(q, key=f"example_{i}", use_container_width=True):
                    st.session_state.pending_question = q
                    st.rerun()

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "citations" in message:
                # Sort by score descending (scores are already absolute from when saved)
                sorted_cites = sorted(message["citations"], key=lambda c: c["score"], reverse=True)
                with st.expander("📎 Sources"):
                    for cite in sorted_cites:
                        st.markdown(f"**{cite['source']}** ({cite['score']}% match)")
                        st.caption(cite["preview"])

    # Chat input (always render this)
    chat_query = st.chat_input("Ask anything about the documents...")
    
    # Handle pending question from example buttons
    query = st.session_state.pop("pending_question", None) or chat_query
    
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    top_k = st.session_state.get("top_k", 5)
                    
                    start = time.time()
                    answer = kb.ask(query, top_k=top_k)
                    elapsed = time.time() - start
                    
                    st.markdown(answer.text)
                    st.caption(f"⏱️ {elapsed:.2f}s")
                    
                    if answer.citations:
                        citations_data = []
                        # Sort by score - if negative (distance), sort ascending; if positive (similarity), sort descending
                        is_distance = any(c.score < 0 for c in answer.citations)
                        sorted_citations = sorted(answer.citations, key=lambda c: c.score, reverse=not is_distance)
                        with st.expander("📎 Sources"):
                            for cite in sorted_citations:
                                # Display as positive percentage (relevance)
                                score_pct = abs(int(cite.score * 100))
                                st.markdown(f"**{cite.source}** ({score_pct}% match)")
                                st.caption(cite.chunk[:200] + "..." if len(cite.chunk) > 200 else cite.chunk)
                                citations_data.append({
                                    "source": cite.source,
                                    "score": score_pct,
                                    "preview": cite.chunk[:200],
                                })
                        
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer.text,
                            "citations": citations_data,
                        })
                    else:
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer.text,
                        })

                except Exception as e:
                    error_msg = f"Error: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                    })


if __name__ == "__main__":
    main()

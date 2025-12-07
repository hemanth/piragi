"""Piragi UI - Simple RAG interface with local LLM chat."""

import os
import shutil
import sys
from pathlib import Path

import streamlit as st

# Project-based storage directories
# Set PIRAGI_PROJECT env var or pass project name to run.sh
PROJECT_NAME = os.environ.get("PIRAGI_PROJECT", "default")
PIRAGI_HOME = Path.home() / ".piragi" / "projects" / PROJECT_NAME
UPLOADS_DIR = PIRAGI_HOME / "uploads"
DATA_DIR = PIRAGI_HOME / "data"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from piragi import Ragi

# Page config
st.set_page_config(
    page_title="Piragi",
    page_icon="📚",
    layout="wide",
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Custom CSS
st.markdown("""
<style>
    .stApp { max-width: 100%; }
    .stMainBlockContainer { max-width: 65%; margin: 0 auto; }
    .citation-box {
        background: #f0f2f6;
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
        border-left: 3px solid #4CAF50;
    }
    .score-badge {
        background: #4CAF50;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8em;
    }
</style>
""", unsafe_allow_html=True)


def get_kb_config():
    """Get current KB config from session state."""
    strategy = st.session_state.get("chunk_strategy", "fixed")
    
    # Build chunk config based on strategy
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
    # contextual uses LLM defaults
    
    return {
        "llm": {
            "model": st.session_state.get("llm_model", "gpt-oss:20b"),
            "base_url": "http://localhost:11434/v1",
            "api_key": "ollama",
        },
        "embedding": {
            "model": st.session_state.get("embed_model", "all-MiniLM-L6-v2"),
        },
        "chunk": chunk_cfg,
        "retrieval": {
            "use_hyde": st.session_state.get("use_hyde", False),
            "use_hybrid_search": st.session_state.get("use_hybrid", False),
            "use_cross_encoder": st.session_state.get("use_reranker", False),
        },
        "auto_update": {"enabled": False},
    }


@st.cache_resource
def get_kb(_config_hash: str):
    """Initialize or get cached knowledge base."""
    config = get_kb_config()
    return Ragi(persist_dir=str(DATA_DIR), config=config)


def config_hash():
    """Generate hash of current config for cache invalidation."""
    import hashlib
    import json
    config = get_kb_config()
    return hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()


def main():
    st.title(f"📚 Piragi ({PROJECT_NAME})")
    st.caption("Zero-setup RAG with smart citations")

    # Track processed files to avoid duplicates
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()

    # Get KB with current config (before sidebar to avoid ghost rendering)
    kb = get_kb(config_hash())

    # Sidebar (single block)
    with st.sidebar:
        # Settings in expander at top
        with st.expander("⚙️ Settings", expanded=False):
            st.subheader("LLM")
            st.text_input(
                "Model",
                value="gpt-oss:20b",
                key="llm_model",
                help="Ollama model name (e.g., llama3.2, mistral, gpt-oss:20b)",
            )
            
            st.subheader("Chunking")
            strategy = st.selectbox(
                "Strategy",
                options=["fixed", "semantic", "contextual", "hierarchical"],
                key="chunk_strategy",
                help="How to split documents into chunks",
            )
            
            # Show relevant params based on strategy
            if strategy == "fixed":
                col1, col2 = st.columns(2)
                with col1:
                    st.number_input(
                        "Chunk size",
                        min_value=100,
                        max_value=2000,
                        value=512,
                        step=100,
                        key="chunk_size",
                    )
                with col2:
                    st.number_input(
                        "Overlap",
                        min_value=0,
                        max_value=200,
                        value=50,
                        step=10,
                        key="chunk_overlap",
                    )
            elif strategy == "semantic":
                st.slider(
                    "Similarity threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.05,
                    key="similarity_threshold",
                    help="Lower = more chunks, Higher = fewer larger chunks",
                )
                col1, col2 = st.columns(2)
                with col1:
                    st.number_input("Min size", value=100, key="min_chunk_size")
                with col2:
                    st.number_input("Max size", value=2000, key="max_chunk_size")
            elif strategy == "hierarchical":
                col1, col2 = st.columns(2)
                with col1:
                    st.number_input("Parent size", value=2000, key="parent_size")
                with col2:
                    st.number_input("Child size", value=400, key="child_size")
            elif strategy == "contextual":
                st.caption("🤖 LLM determines chunk boundaries automatically")
            
            st.subheader("Retrieval")
            st.checkbox(
                "HyDE",
                key="use_hyde",
                help="Hypothetical Document Embeddings - generates a hypothetical answer to improve search",
            )
            st.checkbox(
                "Hybrid Search",
                key="use_hybrid",
                help="Combine semantic search with BM25 keyword matching",
            )
            st.checkbox(
                "Reranker",
                key="use_reranker",
                help="Use cross-encoder to rerank results for better accuracy",
            )
            
            st.number_input(
                "Top K results",
                min_value=1,
                max_value=20,
                value=5,
                key="top_k",
            )
            
            st.divider()
            st.checkbox("🔍 Show debug info", key="show_debug")
            st.caption("💡 Change settings, then re-upload docs to apply")
        
        st.divider()
        st.header("📁 Documents")
        
        # File uploader (key changes on reset to clear widget state)
        uploader_key = st.session_state.get("uploader_key", 0)
        uploaded_files = st.file_uploader(
            "Add documents",
            type=["txt", "md", "pdf", "html", "docx"],
            accept_multiple_files=True,
            key=f"file_uploader_{uploader_key}",
        )
        
        if uploaded_files:
            # Only process new files
            new_files = [f for f in uploaded_files if f.name not in st.session_state.processed_files]
            
            if new_files:
                with st.spinner("Processing..."):
                    for file in new_files:
                        # Save to persistent uploads directory
                        save_path = UPLOADS_DIR / file.name
                        save_path.write_bytes(file.read())
                        
                        try:
                            kb.add(str(save_path))
                            st.session_state.processed_files.add(file.name)
                            st.success(f"✅ {file.name}")
                        except Exception as e:
                            st.error(f"❌ {file.name}: {e}")
                            save_path.unlink(missing_ok=True)  # Clean up on error
        
        # URL input
        st.divider()
        url = st.text_input("Add URL", placeholder="https://...")
        if url and st.button("Add URL"):
            with st.spinner("Fetching..."):
                try:
                    kb.add(url)
                    st.success(f"✅ Added {url}")
                except Exception as e:
                    st.error(f"❌ {e}")
        
        # Stats
        st.divider()
        chunk_count = kb.count()
        st.metric("Chunks indexed", chunk_count)
        
        # Show saved documents
        saved_docs = list(UPLOADS_DIR.glob("*"))
        if saved_docs:
            with st.expander(f"📂 Saved documents ({len(saved_docs)})"):
                for doc in saved_docs:
                    st.caption(f"• {doc.name}")
        
        # Action buttons
        col1, col2 = st.columns(2)
        with col1:
            if chunk_count > 0:
                if st.button("💬 Clear chat", type="secondary", use_container_width=True):
                    st.session_state.messages = []
                    st.rerun()
        with col2:
            if saved_docs:
                if st.button("🔄 Re-index", type="secondary", use_container_width=True):
                    kb.clear()
                    st.session_state.processed_files = set()
                    with st.spinner("Re-indexing..."):
                        for doc in saved_docs:
                            try:
                                kb.add(str(doc))
                                st.session_state.processed_files.add(doc.name)
                            except Exception as e:
                                st.error(f"❌ {doc.name}: {e}")
                    st.rerun()
        
        # Danger zone
        if saved_docs or chunk_count > 0:
            with st.expander("🗑️ Danger Zone"):
                st.warning("This will delete all uploaded files and indexed data.")
                if st.button("🗑️ Delete everything", type="primary"):
                    # Clear KB
                    kb.clear()
                    get_kb.clear()  # Clear cached KB
                    # Delete all uploads
                    for f in UPLOADS_DIR.glob("*"):
                        f.unlink()
                    # Clear session state
                    st.session_state.messages = []
                    st.session_state.processed_files = set()
                    # Clear file uploader by incrementing key
                    st.session_state["uploader_key"] = st.session_state.get("uploader_key", 0) + 1
                    st.rerun()

    # Main chat area
    if kb.count() == 0:
        st.info("👈 Add some documents to get started!")
        return

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

    # Chat input
    if query := st.chat_input("Ask anything about your documents..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    import time
                    top_k = st.session_state.get("top_k", 5)
                    
                    # Time the retrieval
                    start = time.time()
                    answer = kb.ask(query, top_k=top_k)
                    elapsed = time.time() - start
                    
                    st.markdown(answer.text)
                    
                    # Debug info - just timing and config
                    if st.session_state.get("show_debug", False):
                        config = get_kb_config()
                        st.caption(f"⏱️ {elapsed:.2f}s | 📄 {len(answer.citations)} citations | 🔧 {config['chunk']['strategy']} + {'hybrid' if config['retrieval']['use_hybrid_search'] else 'semantic'}")
                    
                    # Show citations
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
                        
                        # Save to history
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

"""Tests for processing hooks.

Run with:
    pytest tests/test_hooks.py -v
"""

import os
import sys
import pytest

# Add src to path for local development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from piragi import Ragi
from piragi.types import Document, Chunk


@pytest.fixture
def test_file(tmp_path):
    """Create a temporary test document."""
    content = """
# Test Document

This is a test document for hooks.

## Section One
First section content.

## Section Two
Second section content.
"""
    path = tmp_path / "test.md"
    path.write_text(content)
    return str(path)


class TestProcessingHooks:
    """Test suite for processing hooks."""

    def test_post_load_hook(self, test_file, tmp_path):
        """post_load hook transforms documents before chunking."""
        hook_called = {"count": 0}
        
        def add_metadata(docs):
            hook_called["count"] += 1
            for doc in docs:
                doc.metadata["processed"] = True
                doc.metadata["custom_field"] = "test_value"
            return docs
        
        kb = Ragi(
            test_file,
            persist_dir=str(tmp_path / ".piragi"),
            config={
                "embedding": {"model": "all-MiniLM-L6-v2"},
                "auto_update": {"enabled": False},
            },
            hooks={
                "post_load": add_metadata,
            },
        )
        
        assert hook_called["count"] == 1

    def test_post_chunk_hook(self, test_file, tmp_path):
        """post_chunk hook transforms chunks before embedding."""
        chunks_seen = {"chunks": []}
        
        def enrich_chunks(chunks):
            for chunk in chunks:
                chunk.metadata["enriched"] = True
                chunk.metadata["word_count"] = len(chunk.text.split())
            chunks_seen["chunks"] = chunks
            return chunks
        
        kb = Ragi(
            test_file,
            persist_dir=str(tmp_path / ".piragi"),
            config={
                "embedding": {"model": "all-MiniLM-L6-v2"},
                "auto_update": {"enabled": False},
            },
            hooks={
                "post_chunk": enrich_chunks,
            },
        )
        
        assert len(chunks_seen["chunks"]) > 0
        for chunk in chunks_seen["chunks"]:
            assert chunk.metadata.get("enriched") is True
            assert "word_count" in chunk.metadata

    def test_post_embed_hook(self, test_file, tmp_path):
        """post_embed hook transforms chunks after embedding, before storage."""
        chunks_with_embeddings = {"chunks": []}
        
        def extract_entities(chunks):
            for chunk in chunks:
                # Verify embeddings exist at this stage
                assert chunk.embedding is not None
                # Add entity extraction results to metadata
                chunk.metadata["entities"] = ["TestEntity"]
                chunk.metadata["has_embedding"] = True
            chunks_with_embeddings["chunks"] = chunks
            return chunks
        
        kb = Ragi(
            test_file,
            persist_dir=str(tmp_path / ".piragi"),
            config={
                "embedding": {"model": "all-MiniLM-L6-v2"},
                "auto_update": {"enabled": False},
            },
            hooks={
                "post_embed": extract_entities,
            },
        )
        
        assert len(chunks_with_embeddings["chunks"]) > 0
        for chunk in chunks_with_embeddings["chunks"]:
            assert chunk.metadata.get("entities") == ["TestEntity"]
            assert chunk.metadata.get("has_embedding") is True

    def test_multiple_hooks(self, test_file, tmp_path):
        """Multiple hooks can be used together."""
        call_order = []
        
        def post_load_hook(docs):
            call_order.append("post_load")
            return docs
        
        def post_chunk_hook(chunks):
            call_order.append("post_chunk")
            return chunks
        
        def post_embed_hook(chunks):
            call_order.append("post_embed")
            return chunks
        
        kb = Ragi(
            test_file,
            persist_dir=str(tmp_path / ".piragi"),
            config={
                "embedding": {"model": "all-MiniLM-L6-v2"},
                "auto_update": {"enabled": False},
            },
            hooks={
                "post_load": post_load_hook,
                "post_chunk": post_chunk_hook,
                "post_embed": post_embed_hook,
            },
        )
        
        assert call_order == ["post_load", "post_chunk", "post_embed"]

    def test_hook_can_filter_chunks(self, test_file, tmp_path):
        """Hooks can filter out chunks."""
        def filter_short_chunks(chunks):
            # Only keep chunks with more than 20 characters
            return [c for c in chunks if len(c.text) > 20]
        
        kb = Ragi(
            test_file,
            persist_dir=str(tmp_path / ".piragi"),
            config={
                "embedding": {"model": "all-MiniLM-L6-v2"},
                "auto_update": {"enabled": False},
            },
            hooks={
                "post_chunk": filter_short_chunks,
            },
        )
        
        # Should have fewer chunks after filtering
        assert kb.count() > 0

    def test_no_hooks_by_default(self, test_file, tmp_path):
        """Without hooks config, processing works normally."""
        kb = Ragi(
            test_file,
            persist_dir=str(tmp_path / ".piragi"),
            config={
                "embedding": {"model": "all-MiniLM-L6-v2"},
                "auto_update": {"enabled": False},
            },
        )
        
        assert kb.count() > 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

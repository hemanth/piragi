"""Tests for SupabaseStore.

Requires environment variables:
    SUPABASE_URL: Your Supabase project URL
    SUPABASE_SERVICE_KEY: Your Supabase service role key

Run with:
    pytest tests/test_supabase_store.py -v
"""

import os
import sys
import pytest

# Add src to path for local development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from piragi import Ragi
from piragi.stores.supabase import SupabaseStore


# Skip all tests if credentials not available
pytestmark = pytest.mark.skipif(
    not os.environ.get("SUPABASE_URL") or not os.environ.get("SUPABASE_SERVICE_KEY"),
    reason="SUPABASE_URL and SUPABASE_SERVICE_KEY required",
)


@pytest.fixture
def store():
    """Create a fresh store for each test."""
    s = SupabaseStore()
    s.clear()
    yield s
    s.clear()


@pytest.fixture
def test_file(tmp_path):
    """Create a temporary test document."""
    content = """
# Test Document

This is a test document for the Supabase store.

## Section One
The first section contains information about testing.

## Section Two
The second section discusses vector databases and embeddings.
"""
    path = tmp_path / "test.md"
    path.write_text(content)
    return str(path)


class TestSupabaseStore:
    """Test suite for SupabaseStore."""

    def test_init_with_env_vars(self):
        """Store initializes with environment variables."""
        store = SupabaseStore()
        assert store.url is not None
        assert store.key is not None

    def test_init_missing_credentials(self, monkeypatch):
        """Store raises error when credentials missing."""
        monkeypatch.delenv("SUPABASE_URL", raising=False)
        monkeypatch.delenv("SUPABASE_SERVICE_KEY", raising=False)
        
        with pytest.raises(ValueError, match="Supabase URL and key required"):
            SupabaseStore()

    def test_count_empty(self, store):
        """Empty store returns count of 0."""
        assert store.count() == 0

    def test_add_and_count(self, store, test_file):
        """Adding chunks increases count."""
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        kb = Ragi(
            test_file,
            store=store,
            config={
                "embedding": {"model": "all-MiniLM-L6-v2"},
                "auto_update": {"enabled": False},
            },
        )
        
        assert kb.count() > 0

    def test_search_returns_results(self, store, test_file):
        """Search returns relevant results."""
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        kb = Ragi(
            test_file,
            store=store,
            config={
                "embedding": {"model": "all-MiniLM-L6-v2"},
                "auto_update": {"enabled": False},
            },
        )
        
        results = kb.retrieve("vector databases", top_k=2)
        
        assert len(results) > 0
        assert results[0].score > 0

    def test_delete_by_source(self, store, test_file):
        """Deleting by source removes chunks."""
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        kb = Ragi(
            test_file,
            store=store,
            config={
                "embedding": {"model": "all-MiniLM-L6-v2"},
                "auto_update": {"enabled": False},
            },
        )
        
        initial_count = kb.count()
        assert initial_count > 0
        
        deleted = store.delete_by_source(test_file)
        
        assert deleted == initial_count
        assert store.count() == 0

    def test_clear(self, store, test_file):
        """Clear removes all chunks."""
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        kb = Ragi(
            test_file,
            store=store,
            config={
                "embedding": {"model": "all-MiniLM-L6-v2"},
                "auto_update": {"enabled": False},
            },
        )
        
        assert kb.count() > 0
        
        store.clear()
        
        assert store.count() == 0

    def test_get_all_chunk_texts(self, store, test_file):
        """get_all_chunk_texts returns texts for hybrid search."""
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        kb = Ragi(
            test_file,
            store=store,
            config={
                "embedding": {"model": "all-MiniLM-L6-v2"},
                "auto_update": {"enabled": False},
            },
        )
        
        texts = store.get_all_chunk_texts()
        
        assert len(texts) == kb.count()
        assert all(isinstance(t, str) for t in texts)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

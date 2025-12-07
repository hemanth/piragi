"""Supabase vector store using pgvector."""

from typing import Any, Dict, List, Optional
import json
import os

from ..types import Chunk, Citation


class SupabaseStore:
    """
    Supabase vector store using pgvector extension.

    Requires:
        pip install supabase

    Setup SQL (run in Supabase SQL editor):
        -- Enable pgvector
        CREATE EXTENSION IF NOT EXISTS vector;

        -- Create chunks table
        CREATE TABLE IF NOT EXISTS piragi_chunks (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            text TEXT NOT NULL,
            source TEXT NOT NULL,
            chunk_index INTEGER,
            metadata JSONB DEFAULT '{}',
            embedding vector(384),  -- Adjust dimension for your model
            created_at TIMESTAMPTZ DEFAULT now()
        );

        -- Create vector search index
        CREATE INDEX IF NOT EXISTS piragi_chunks_embedding_idx
        ON piragi_chunks USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100);

        -- Create source index for filtering
        CREATE INDEX IF NOT EXISTS piragi_chunks_source_idx
        ON piragi_chunks (source);

        -- Create vector search function
        CREATE OR REPLACE FUNCTION match_piragi_chunks(
            query_embedding vector(384),  -- Match your dimension
            match_count int DEFAULT 5,
            filter_source text DEFAULT NULL
        )
        RETURNS TABLE (
            id uuid,
            text text,
            source text,
            chunk_index int,
            metadata jsonb,
            similarity float
        )
        LANGUAGE plpgsql
        AS $$
        BEGIN
            RETURN QUERY
            SELECT
                pc.id,
                pc.text,
                pc.source,
                pc.chunk_index,
                pc.metadata,
                1 - (pc.embedding <=> query_embedding) AS similarity
            FROM piragi_chunks pc
            WHERE pc.embedding IS NOT NULL
                AND (filter_source IS NULL OR pc.source = filter_source)
            ORDER BY pc.embedding <=> query_embedding
            LIMIT match_count;
        END;
        $$;

    Examples:
        >>> store = SupabaseStore(
        ...     url="https://xxx.supabase.co",
        ...     key="your-service-role-key",
        ... )
        >>>
        >>> # Or use environment variables
        >>> store = SupabaseStore()  # Uses SUPABASE_URL and SUPABASE_SERVICE_KEY
        >>>
        >>> # Custom table and dimensions
        >>> store = SupabaseStore(
        ...     table_name="my_chunks",
        ...     function_name="match_my_chunks",
        ...     vector_dimension=1536,  # OpenAI ada-002
        ... )
    """

    def __init__(
        self,
        url: Optional[str] = None,
        key: Optional[str] = None,
        table_name: str = "piragi_chunks",
        function_name: str = "match_piragi_chunks",
        vector_dimension: int = 384,
    ) -> None:
        """
        Initialize Supabase store.

        Args:
            url: Supabase project URL (or SUPABASE_URL env var)
            key: Supabase service role key (or SUPABASE_SERVICE_KEY env var)
            table_name: Table name for storing chunks
            function_name: RPC function name for vector search
            vector_dimension: Dimension of embedding vectors (384 for MiniLM, 1536 for OpenAI)
        """
        try:
            from supabase import create_client
        except ImportError:
            raise ImportError(
                "SupabaseStore requires supabase-py. "
                "Install with: pip install piragi[supabase]"
            )

        self.url = url or os.environ.get("SUPABASE_URL")
        self.key = key or os.environ.get("SUPABASE_SERVICE_KEY")

        if not self.url or not self.key:
            raise ValueError(
                "Supabase URL and key required. "
                "Set SUPABASE_URL and SUPABASE_SERVICE_KEY environment variables "
                "or pass url and key parameters."
            )

        self.client = create_client(self.url, self.key)
        self.table_name = table_name
        self.function_name = function_name
        self.vector_dimension = vector_dimension
        self._chunk_texts: List[str] = []

        # Load existing chunk texts for hybrid search
        self._load_chunk_texts()

    def _load_chunk_texts(self) -> None:
        """Load all chunk texts for hybrid search indexing."""
        try:
            result = self.client.table(self.table_name).select("text").execute()
            self._chunk_texts = [row["text"] for row in result.data]
        except Exception:
            # Table may not exist yet
            self._chunk_texts = []

    def add_chunks(self, chunks: List[Chunk]) -> None:
        """
        Add chunks with embeddings to the store.

        Args:
            chunks: List of Chunk objects with embeddings

        Raises:
            ValueError: If any chunk is missing an embedding
        """
        if not chunks:
            return

        for chunk in chunks:
            if chunk.embedding is None:
                raise ValueError("All chunks must have embeddings")

        rows = [
            {
                "text": chunk.text,
                "source": chunk.source,
                "chunk_index": chunk.chunk_index,
                "metadata": chunk.metadata,
                "embedding": chunk.embedding,
            }
            for chunk in chunks
        ]

        self.client.table(self.table_name).insert(rows).execute()
        self._chunk_texts.extend([c.text for c in chunks])

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Citation]:
        """
        Search for similar chunks using cosine similarity.

        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            filters: Optional filters (supports 'source' key)

        Returns:
            List of Citation objects with similarity scores
        """
        # Build RPC parameters
        params: Dict[str, Any] = {
            "query_embedding": query_embedding,
            "match_count": top_k,
        }

        # Add source filter if provided
        if filters and "source" in filters:
            params["filter_source"] = filters["source"]

        try:
            result = self.client.rpc(self.function_name, params).execute()

            return [
                Citation(
                    source=row["source"],
                    chunk=row["text"],
                    score=float(row.get("similarity", 0)),
                    metadata=row.get("metadata") or {},
                )
                for row in result.data
            ]
        except Exception as e:
            # Fall back to client-side similarity if RPC fails
            return self._search_fallback(query_embedding, top_k, filters)

    def _search_fallback(
        self,
        query_embedding: List[float],
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Citation]:
        """
        Fallback search computing similarity client-side.

        Used when RPC function is not available.
        """
        import numpy as np

        # Build query
        query = self.client.table(self.table_name).select("*")

        if filters and "source" in filters:
            query = query.eq("source", filters["source"])

        result = query.limit(1000).execute()

        query_vec = np.array(query_embedding, dtype=np.float32)
        scored = []

        for row in result.data:
            emb = row.get("embedding")
            if emb:
                # Handle string-encoded vectors from Supabase
                if isinstance(emb, str):
                    emb = json.loads(emb)
                chunk_vec = np.array(emb, dtype=np.float32)
                # Cosine similarity
                norm_product = np.linalg.norm(query_vec) * np.linalg.norm(chunk_vec)
                if norm_product > 0:
                    sim = float(np.dot(query_vec, chunk_vec) / norm_product)
                    scored.append((row, sim))

        scored.sort(key=lambda x: x[1], reverse=True)

        return [
            Citation(
                source=row["source"],
                chunk=row["text"],
                score=score,
                metadata=row.get("metadata") or {},
            )
            for row, score in scored[:top_k]
        ]

    def delete_by_source(self, source: str) -> int:
        """
        Delete all chunks from a specific source.

        Args:
            source: Source identifier to delete

        Returns:
            Number of chunks deleted
        """
        result = (
            self.client.table(self.table_name)
            .delete()
            .eq("source", source)
            .execute()
        )
        deleted = len(result.data)
        self._load_chunk_texts()
        return deleted

    def count(self) -> int:
        """Return the number of chunks in the store."""
        result = (
            self.client.table(self.table_name)
            .select("id", count="exact")
            .execute()
        )
        return result.count or 0

    def clear(self) -> None:
        """Clear all data from the store."""
        # Delete all rows (Supabase requires a condition)
        self.client.table(self.table_name).delete().neq(
            "id", "00000000-0000-0000-0000-000000000000"
        ).execute()
        self._chunk_texts = []

    def get_all_chunk_texts(self) -> List[str]:
        """Get all chunk texts for hybrid search indexing."""
        return self._chunk_texts

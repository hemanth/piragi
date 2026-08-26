"""On-the-fly embedding: embed the corpus at query time instead of pre-indexing.

Best for high-churn content (>10% daily updates) where keeping a vector store
in sync costs more than just re-embedding the small corpus per query.
Swapping embedding models is trivial here - there's no index to rebuild.
"""

import logging
from typing import Dict, List, Optional

from .types import Chunk, Citation

logger = logging.getLogger(__name__)


class OnTheFlyIndex:
    """
    Keeps raw chunks in memory (no persisted vectors) and embeds them at
    query time. Embeddings are cached per chunk text so a query only pays
    the embedding cost for chunks it hasn't seen before - new/changed
    documents are embedded lazily, matching high-churn corpora.
    """

    def __init__(self) -> None:
        self._chunks: List[Citation] = []
        self._embedding_cache: Dict[str, List[float]] = {}

    def add_chunks(self, chunks: List[Chunk]) -> None:
        """Add raw (unembedded) chunks to the index."""
        for chunk in chunks:
            self._chunks.append(
                Citation(
                    source=chunk.source,
                    chunk=chunk.text,
                    score=0.0,
                    metadata=chunk.metadata,
                )
            )

    def count(self) -> int:
        return len(self._chunks)

    def clear(self) -> None:
        self._chunks = []
        self._embedding_cache = {}

    def delete_by_source(self, source: str) -> int:
        """Delete all chunks from a specific source. Returns number deleted."""
        before = len(self._chunks)
        self._chunks = [c for c in self._chunks if c.source != source]
        return before - len(self._chunks)

    def search(self, embedder, query: str, top_k: int = 10) -> List[Citation]:
        """Embed the corpus (using the cache) and the query, then rank by cosine similarity."""
        import numpy as np

        if not self._chunks:
            return []

        to_embed = [c.chunk for c in self._chunks if c.chunk not in self._embedding_cache]
        if to_embed:
            fresh = embedder._generate_embeddings(to_embed, batch_size=embedder.batch_size)
            for text, embedding in zip(to_embed, fresh):
                self._embedding_cache[text] = list(embedding)

        query_embedding = np.array(embedder.embed_query(query))
        query_norm = np.linalg.norm(query_embedding) or 1.0

        scored = []
        for citation in self._chunks:
            vec = np.array(self._embedding_cache[citation.chunk])
            vec_norm = np.linalg.norm(vec) or 1.0
            similarity = float(np.dot(query_embedding, vec) / (query_norm * vec_norm))
            scored.append((similarity, citation))

        scored.sort(key=lambda x: x[0], reverse=True)

        return [
            Citation(source=c.source, chunk=c.chunk, score=score, metadata=c.metadata)
            for score, c in scored[:top_k]
        ]

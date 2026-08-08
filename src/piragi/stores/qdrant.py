"""
Qdrant vector store backend for piragi.

Supports in-memory, local file, and remote Qdrant servers.

Example:
    >>> from piragi import Ragi
    >>> from piragi.stores import QdrantStore
    >>>
    >>> # In-memory (default)
    >>> kb = Ragi("./docs", store=QdrantStore())
    >>>
    >>> # Remote server
    >>> kb = Ragi("./docs", store=QdrantStore(url="http://localhost:6333"))
    >>>
    >>> # Quantization for reduced memory usage
    >>> kb = Ragi("./docs", store=QdrantStore(quantization="scalar"))
"""

import logging
import uuid
from typing import Any, Dict, List, Optional

from ..types import Chunk, Citation

try:
    from qdrant_client import QdrantClient, models
except ImportError:
    QdrantClient = None
    models = None

logger = logging.getLogger(__name__)


class QdrantStore:
    """
    Qdrant implementation of VectorStoreProtocol.
    """

    def __init__(
        self,
        url: str = ":memory:",
        collection_name: str = "chunks",
        vector_dimension: int = 768,
        api_key: Optional[str] = None,
        quantization: Optional[str] = None,
        quantization_bits: int = 4,
        embedder: Optional[Any] = None,
    ) -> None:
        """
        Initialize the Qdrant store.
        
        Args:
            url: Qdrant URL. Defaults to ":memory:"
            collection_name: Name of the collection. Defaults to "chunks".
            vector_dimension: Embedding dimension size.
            api_key: Qdrant API key if required.
            quantization: Quantization type (None, "scalar", "turboquant").
            quantization_bits: Bits for quantization (default 4).
            embedder: Optional embedder model to infer dimension size.
        """
        if QdrantClient is None:
            raise ImportError(
                "qdrant-client not installed. Install with `pip install piragi[qdrant]`"
            )

        self.url = url
        self.collection_name = collection_name
        self.api_key = api_key
        self.quantization = quantization
        self.quantization_bits = quantization_bits
        self._chunk_texts: List[str] = []
        
        # Auto-infer dimension if embedder provided
        if embedder is not None and hasattr(embedder, "get_dimensions"):
            logger.info("Auto-inferring vector dimensions from embedder...")
            self.vector_dimension = embedder.get_dimensions()
            logger.info(f"Detected vector dimension: {self.vector_dimension}")
        else:
            self.vector_dimension = vector_dimension

        if self.url == ":memory:":
            self.client = QdrantClient(location=":memory:")
        else:
            self.client = QdrantClient(url=self.url, api_key=self.api_key)

        self._init_collection()

    def _get_quantization_config(self) -> Optional[Any]:
        """Get quantization configuration based on parameters."""
        if not self.quantization:
            return None

        if self.quantization.lower() == "scalar":
            return models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(
                    type=models.ScalarType.INT8,
                    always_ram=True,
                )
            )
        elif self.quantization.lower() == "turboquant":
            try:
                # Attempt to use TurboQuantization if available
                return models.TurboQuantization(
                    turbo=models.TurboQuantizationConfig()
                )
            except AttributeError:
                logger.warning(
                    "TurboQuant not supported in this version of qdrant-client. "
                    "Falling back to scalar quantization."
                )
                return models.ScalarQuantization(
                    scalar=models.ScalarQuantizationConfig(
                        type=models.ScalarType.INT8,
                        always_ram=True,
                    )
                )

        logger.warning(f"Unknown quantization type: {self.quantization}. Disabling quantization.")
        return None

    def _init_collection(self) -> None:
        """Create the collection if it doesn't exist."""
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_dimension,
                    distance=models.Distance.COSINE
                ),
                quantization_config=self._get_quantization_config(),
            )
        self._load_chunk_texts()

    def _load_chunk_texts(self) -> None:
        """Load all chunk texts for hybrid search."""
        self._chunk_texts = []
        
        offset = None
        while True:
            result, offset = self.client.scroll(
                collection_name=self.collection_name,
                with_payload=True,
                with_vectors=False,
                limit=1000,
                offset=offset,
            )
            for point in result:
                if "text" in point.payload:
                    self._chunk_texts.append(point.payload["text"])
            if offset is None:
                break

    def add_chunks(self, chunks: List[Chunk]) -> None:
        """Add chunks with embeddings to the store."""
        if not chunks:
            return

        points = []
        for chunk in chunks:
            if chunk.embedding is None:
                raise ValueError("All chunks must have embeddings")
                
            point_id = str(uuid.uuid4())
            payload = {
                "text": chunk.text,
                "source": chunk.source,
                "chunk_index": chunk.chunk_index,
            }
            if chunk.metadata:
                for k, v in chunk.metadata.items():
                    payload[k] = v
                    
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=chunk.embedding,
                    payload=payload,
                )
            )
            self._chunk_texts.append(chunk.text)
            
        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        min_chunk_length: int = 100,
    ) -> List[Citation]:
        """Search for similar chunks."""
        query_filter = None
        if filters:
            must_conditions = []
            for key, value in filters.items():
                must_conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value)
                    )
                )
            query_filter = models.Filter(must=must_conditions)
            
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            query_filter=query_filter,
            limit=top_k * 3,  # Fetch extra for min length filtering
            with_payload=True,
        )
        
        citations = []
        for point in results.points:
            text = point.payload.get("text", "")
            if len(text) < min_chunk_length:
                continue
                
            metadata = {
                k: v for k, v in point.payload.items()
                if k not in ("text", "source", "chunk_index")
            }
            
            citations.append(
                Citation(
                    source=point.payload.get("source", ""),
                    chunk=text,
                    score=point.score,
                    metadata=metadata,
                )
            )
            
            if len(citations) >= top_k:
                break
                
        return citations

    def delete_by_source(self, source: str) -> int:
        """Delete all chunks from a specific source."""
        count_result = self.client.count(
            collection_name=self.collection_name,
            count_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="source",
                        match=models.MatchValue(value=source)
                    )
                ]
            )
        )
        count = count_result.count
        
        if count > 0:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="source",
                                match=models.MatchValue(value=source)
                            )
                        ]
                    )
                )
            )
            self._load_chunk_texts()
            
        return count

    def count(self) -> int:
        """Return the number of chunks in the store."""
        return self.client.count(collection_name=self.collection_name).count

    def clear(self) -> None:
        """Clear all data from the store."""
        self.client.delete_collection(collection_name=self.collection_name)
        self._chunk_texts = []
        self._init_collection()

    def get_all_chunk_texts(self) -> List[str]:
        """Get all chunk texts for hybrid search."""
        return self._chunk_texts


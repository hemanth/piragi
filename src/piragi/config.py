from pydantic import BaseModel, Field
from typing import Optional

class LLMConfig(BaseModel):
    model_config = {"extra": "forbid"}
    model: str = "llama3.2"
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    temperature: float = 0.1
    enable_reranking: bool = True
    enable_query_expansion: bool = True

class EmbeddingConfig(BaseModel):
    model_config = {"extra": "forbid"}
    model: str = "all-mpnet-base-v2"
    device: Optional[str] = None
    backend: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    batch_size: int = 32

class ChunkConfig(BaseModel):
    model_config = {"extra": "forbid"}
    strategy: str = "fixed"
    size: int = 512
    overlap: int = 50
    min_chunk_length: int = 0
    similarity_threshold: float = 0.5
    min_size: int = 100
    max_size: int = 2000
    parent_size: int = 2000
    child_size: int = 400

class RetrievalConfig(BaseModel):
    model_config = {"extra": "forbid"}
    use_hyde: bool = False
    use_hybrid_search: bool = False
    use_cross_encoder: bool = False
    vector_weight: float = 0.5
    bm25_weight: float = 0.5
    use_rrf: bool = True
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    cross_encoder_device: Optional[str] = None
    trust_remote_code: bool = False
    rerank_top_n: int = 0   # candidate pool size for cross-encoder / vector over-fetch (0 = auto: top_k*4)
    hybrid_top_n: int = 0   # BM25 candidate pool for hybrid fusion (0 = auto: max(top_k*4, 50))
    warm_models: bool = True  # eager-load cross-encoder at init to avoid cold-start p99 spike

class AutoUpdateConfig(BaseModel):
    model_config = {"extra": "forbid"}
    enabled: bool = True
    interval: float = 300.0
    workers: int = 2

class RagiConfig(BaseModel):
    """Validated configuration for Ragi."""
    model_config = {"extra": "forbid"}
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    chunk: ChunkConfig = Field(default_factory=ChunkConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    auto_update: AutoUpdateConfig = Field(default_factory=AutoUpdateConfig)

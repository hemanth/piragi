from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator


class DecisionConfig(BaseModel):
    """Opt-in evidence judging. Thresholds are uncalibrated option scores."""

    model_config = {"extra": "forbid"}
    engine: Optional[Literal["openjev"]] = None
    enabled: bool = False
    model_path: Optional[str] = None
    revision: Optional[str] = Field(default=None, min_length=1)
    backend: Literal["auto", "torch", "mlx"] = "auto"
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    dtype: Literal["auto", "bfloat16", "float16", "float32"] = "auto"
    cache_dir: Optional[str] = None
    local_files_only: bool = False
    max_tokens: int = Field(default=4096, ge=1)
    min_score: float = Field(default=0.7, ge=0, le=1, allow_inf_nan=False)
    min_margin: float = Field(default=0.2, ge=0, le=1, allow_inf_nan=False)
    candidate_k: int = Field(default=20, ge=1)
    mlx_bits: Optional[Literal[4, 8]] = 4

    @model_validator(mode="before")
    @classmethod
    def enable_selected_engine(cls, data):
        if isinstance(data, dict) and data.get("engine") and "enabled" not in data:
            return {**data, "enabled": True}
        return data


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
    mode: str = "dense"  # "dense" | "bm25_only" | "on_the_fly" | "hot_cold"
    use_hyde: bool = False
    use_hybrid_search: bool = False
    use_cross_encoder: bool = False
    use_query_rewrite: bool = False
    use_step_back: bool = False
    vector_weight: float = 0.5
    bm25_weight: float = 0.5
    use_rrf: bool = True
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    cross_encoder_device: Optional[str] = None
    trust_remote_code: bool = False
    rerank_top_n: Optional[int] = Field(default=None, ge=1)
    rerank_fusion_weight: float = Field(default=0.0, ge=0.0, le=1.0, allow_inf_nan=False)
    context_ordering: Literal["u_shaped", "descending"] = "u_shaped"
    prune_unreferenced_citations: bool = True
    auto_merge_parents: bool = True

class AutoUpdateConfig(BaseModel):
    model_config = {"extra": "forbid"}
    enabled: bool = True
    interval: float = 300.0
    workers: int = 2

class TelemetryConfig(BaseModel):
    model_config = {"extra": "forbid"}
    enabled: bool = True
    structured_logs: bool = False
    sample_rate: float = Field(default=1.0, ge=0.0, le=1.0, allow_inf_nan=False)

class RagiConfig(BaseModel):
    """Validated configuration for Ragi."""
    model_config = {"extra": "forbid"}
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    chunk: ChunkConfig = Field(default_factory=ChunkConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    auto_update: AutoUpdateConfig = Field(default_factory=AutoUpdateConfig)
    decision: DecisionConfig = Field(default_factory=DecisionConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)

    @model_validator(mode="before")
    @classmethod
    def evidence_defaults(cls, data):
        if not isinstance(data, dict):
            return data
        decision = DecisionConfig.model_validate(data.get("decision", {}))
        if not decision.enabled:
            return data
        data = dict(data)
        # Explicit settings win over this opt-in preset.
        for section, defaults in {
            "llm": {"enable_query_expansion": False, "enable_reranking": False},
            "retrieval": {"use_hybrid_search": True},
        }.items():
            supplied = data.get(section, {})
            if isinstance(supplied, dict):
                data[section] = {**defaults, **supplied}
        return data

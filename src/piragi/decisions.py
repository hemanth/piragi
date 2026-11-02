"""Optional on-device semantic decisions using SemIf (formerly OpenJev)."""

from __future__ import annotations

import importlib
import importlib.util
import logging
import math
import platform
import re
from collections.abc import Mapping
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)

# SemIf's reference checkpoint, shared by its Torch and MLX implementations.
DEFAULT_MODEL = "Qwen/Qwen3.5-4B"
DEFAULT_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"


class DecisionResult(BaseModel):
    """Conditional option scores, not calibrated confidence in correctness."""

    choice: str
    scores: dict[str, float]
    model: dict[str, Any] = Field(default_factory=dict)
    elapsed_seconds: float = Field(default=0, ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_scores(self) -> DecisionResult:
        if len(self.scores) < 2 or self.choice not in self.scores:
            raise ValueError("A decision requires at least two scores and a listed choice")
        if any(not math.isfinite(p) or not 0 <= p <= 1 for p in self.scores.values()):
            raise ValueError("Option scores must be finite probabilities")
        if not math.isclose(sum(self.scores.values()), 1, abs_tol=1e-5):
            raise ValueError("Option scores must sum to one")
        if self.scores[self.choice] != max(self.scores.values()):
            raise ValueError("Choice must have the highest option score")
        return self

    @property
    def margin(self) -> float:
        ordered = sorted(self.scores.values(), reverse=True)
        return ordered[0] - ordered[1]


class DecisionEngine(Protocol):
    """A local or application-provided scorer for a finite set of decisions."""

    def decide(
        self, *, state: str | dict | list, question: str, options: Mapping[str, str]
    ) -> DecisionResult: ...

    def decide_many(
        self, *, state: str | dict | list, questions: Mapping[str, DecisionQuestion]
    ) -> dict[str, DecisionResult]: ...


class DecisionQuestion(BaseModel):
    """One independent question in a shared-state batch."""

    question: str
    options: dict[str, str]


class SemIfDecisionEngine:
    """Lazy, serialized adapter to SemIf's native direct scoring API.

    The first decision downloads/caches a pinned checkpoint unless model_path is
    provided. Runtime selection and loading are lazy. Install SemIf separately;
    MLX supports upstream's native Qwen3.5 checkpoints.
    """

    def __init__(
        self,
        model_path: str | None = None,
        revision: str | None = None,
        *,
        backend: Literal["auto", "torch", "mlx"] = "auto",
        device: Literal["auto", "cpu", "cuda", "mps"] = "auto",
        dtype: Literal["auto", "bfloat16", "float16", "float32"] = "auto",
        max_tokens: int = 4096,
        cache_dir: str | None = None,
        local_files_only: bool = False,
        mlx_bits: Literal[4, 8] | None = 4,
    ) -> None:
        path = Path(model_path).expanduser().resolve() if model_path is not None else None
        if path is not None and not path.is_dir():
            raise ValueError("model_path must be an existing local checkpoint directory")
        revision = revision if revision is not None else ("local" if path else DEFAULT_REVISION)
        if not revision.strip():
            raise ValueError("revision must identify the local checkpoint")
        if path is None and not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("Downloaded models require a pinned 40-character revision")
        if backend not in {"auto", "torch", "mlx"}:
            raise ValueError("backend must be auto, torch or mlx")
        if device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("Unsupported Torch device")
        if dtype not in {"auto", "bfloat16", "float16", "float32"}:
            raise ValueError("Unsupported Torch dtype")
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        self.model_path = str(path) if path is not None else None
        self.revision = revision
        self.backend = backend
        self.device = device
        self.dtype = dtype
        self.max_tokens = max_tokens
        self.cache_dir = str(Path(cache_dir).expanduser()) if cache_dir is not None else None
        self.local_files_only = local_files_only
        if mlx_bits not in (None, 4, 8):
            raise ValueError("mlx_bits must be 4, 8 or None")
        self.mlx_bits = mlx_bits
        self._lock = Lock()
        self._loaded = None
        self._score = None
        self._score_shared = None

    def _runtime(self) -> tuple[str, str, str]:
        backend = self.backend
        if backend == "auto":
            backend = "torch"
            if (
                self.device == "auto"
                and platform.system() == "Darwin"
                and platform.machine() == "arm64"
                and importlib.util.find_spec("mlx") is not None
                and importlib.util.find_spec("mlx_lm") is not None
            ):
                mx = importlib.import_module("mlx.core")
                if mx.metal.is_available():
                    backend = "mlx"
        if backend == "mlx":
            return backend, self.device, self.dtype

        torch = importlib.import_module("torch")
        device = self.device
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        dtype = self.dtype
        if dtype == "auto":
            if device == "cpu":
                dtype = "float32"
            elif device == "cuda" and torch.cuda.is_bf16_supported():
                dtype = "bfloat16"
            else:
                dtype = "float16"
        return backend, device, dtype

    def _checkpoint(self) -> str:
        if self.model_path is not None:
            # Never interpret a removed local checkpoint as a Hub identifier.
            if not Path(self.model_path).is_dir():
                raise FileNotFoundError("Local decision checkpoint is no longer available")
            return self.model_path

        hub = importlib.import_module("huggingface_hub")
        logger.info("Loading OpenJev model %s from cache or Hugging Face", DEFAULT_MODEL)
        return hub.snapshot_download(
            repo_id=DEFAULT_MODEL,
            revision=self.revision,
            cache_dir=self.cache_dir,
            local_files_only=self.local_files_only,
            allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt", "*.model"],
        )

    def _load(self) -> None:
        try:
            backend, device, dtype = self._runtime()
            if backend == "mlx":
                module = importlib.import_module("semif_phase1.mlx_backend")
                loaded = module.load_model(self._checkpoint(), self.revision, bits=self.mlx_bits)
                shared = module
            else:
                core = importlib.import_module("semif_phase1.core")
                module = importlib.import_module("semif_phase1.direct")
                shared = importlib.import_module("semif_phase1.shared")
                loaded = core.load_causal_model(self._checkpoint(), self.revision, device, dtype)
        except ImportError as error:
            raise ImportError(
                "Install SemIf and its selected backend; see docs/local-decisions.md"
            ) from error
        self._score = module.score
        self._score_shared = shared.score_shared
        self._loaded = loaded

    def decide_many(
        self, *, state: str | dict | list, questions: Mapping[str, DecisionQuestion]
    ) -> dict[str, DecisionResult]:
        """Score independent questions with one shared prefill and batched suffixes.

        Uses SemIf's native shared scorer, not a loop of direct model calls.
        Upstream Torch MPS uses cached serial suffixes; MLX/CUDA batch them.
        """
        if not questions:
            return {}
        rows = [
            {
                "id": key,
                "state": state,
                "question": spec.question,
                "options": [{"id": k, "description": v} for k, v in spec.options.items()],
            }
            for key, spec in questions.items()
        ]
        with self._lock:
            started = perf_counter()
            core = importlib.import_module("semif_phase1.core")
            for row in rows:
                core.validate_row(row)
            if self._loaded is None:
                self._load()
            model, tokenizer, metadata = self._loaded
            raw, timing = self._score_shared(model, tokenizer, rows, metadata, self.max_tokens)
            if len(raw) != len(rows) or [r["id"] for r in raw] != list(questions):
                raise ValueError("SemIf returned a different question set")
            elapsed = perf_counter() - started
            results = {}
            for row, item in zip(rows, raw):
                ids, probabilities = item["option_ids"], item["probabilities"]
                if ids != list(questions[row["id"]].options) or len(probabilities) != len(ids):
                    raise ValueError("SemIf returned a different option set")
                scores = dict(zip(ids, probabilities))
                results[row["id"]] = DecisionResult(
                    choice=max(scores, key=scores.get),
                    scores=scores,
                    model={**item["model"], "batch_timing": timing},
                    # Batch elapsed is shared, not additive across questions.
                    elapsed_seconds=elapsed,
                )
            return results

    def decide(
        self, *, state: str | dict | list, question: str, options: Mapping[str, str]
    ) -> DecisionResult:
        row = {
            "id": "piragi-decision",
            "state": state,
            "question": question,
            "options": [{"id": key, "description": value} for key, value in options.items()],
        }
        # One model instance may be shared by concurrent Ragi / AsyncRagi calls.
        with self._lock:
            started = perf_counter()
            try:
                core = importlib.import_module("semif_phase1.core")
            except ImportError as error:
                raise ImportError(
                    "Install SemIf to use local decisions; see docs/local-decisions.md"
                ) from error
            core.validate_row(row)
            if self._loaded is None:
                self._load()
            model, tokenizer, metadata = self._loaded
            raw = self._score(model, tokenizer, row, metadata, self.max_tokens)
            ids, probabilities = raw["option_ids"], raw["probabilities"]
            if ids != list(options) or len(probabilities) != len(ids):
                raise ValueError("SemIf returned a different option set")
            scores = dict(zip(ids, probabilities))
            return DecisionResult(
                choice=max(scores, key=scores.get),
                scores=scores,
                model=raw["model"],
                elapsed_seconds=perf_counter() - started,
            )

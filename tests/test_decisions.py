"""Local decision scoring contracts without downloading model weights."""

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from piragi import AsyncRagi, DecisionResult, Ragi, SemIfDecisionEngine
from piragi.config import RagiConfig
from piragi.decisions import DEFAULT_REVISION


def result(original=0.96):
    return DecisionResult(
        choice="original" if original >= 0.5 else "expand",
        scores={"original": original, "expand": 1 - original},
    )


@pytest.fixture
def local_backend(monkeypatch):
    """Match the upstream load and direct-score signatures, with no GPU dependency."""
    # Hardware selection is tested separately with explicit device fixtures.
    monkeypatch.setattr(
        SemIfDecisionEngine,
        "_runtime",
        lambda self: (
            "mlx" if self.backend == "mlx" else "torch",
            "cpu" if self.device == "auto" else self.device,
            "float32" if self.dtype == "auto" else self.dtype,
        ),
    )
    core = ModuleType("semif_phase1.core")
    direct = ModuleType("semif_phase1.direct")
    mlx = ModuleType("semif_phase1.mlx_backend")
    core.validate_row = MagicMock()
    core.load_causal_model = MagicMock(return_value=("model", "tokenizer", {"revision": "test"}))
    direct.score = MagicMock(
        return_value={
            "option_ids": ["original", "expand"],
            "probabilities": [0.96, 0.04],
            "model": {"revision": "test"},
        }
    )
    mlx.load_model = MagicMock(return_value=core.load_causal_model.return_value)
    mlx.score = direct.score
    shared = ModuleType("semif_phase1.shared")

    def batch(model, tokenizer, rows, metadata, max_tokens):
        return [
            {
                "id": row["id"],
                "option_ids": [o["id"] for o in row["options"]],
                "probabilities": [0.96, 0.04],
                "model": metadata,
            }
            for row in rows
        ], {"batch_size": len(rows)}

    shared.score_shared = MagicMock(side_effect=batch)
    mlx.score_shared = shared.score_shared
    monkeypatch.setitem(sys.modules, "semif_phase1.shared", shared)
    for name, module in [("core", core), ("direct", direct), ("mlx_backend", mlx)]:
        monkeypatch.setitem(sys.modules, f"semif_phase1.{name}", module)
    return core, direct, mlx


def decide(engine):
    return engine.decide(
        state={"query": "What is top_k?"},
        question="Expand?",
        options={"original": "Use the query", "expand": "Expand it"},
    )


@pytest.mark.parametrize("backend", ["torch", "mlx"])
def test_lazy_load_once_and_local_path(temp_dir, local_backend, backend):
    core, direct, mlx = local_backend
    engine = SemIfDecisionEngine(temp_dir, "test", backend=backend, max_tokens=256)
    core.load_causal_model.assert_not_called()
    mlx.load_model.assert_not_called()
    first, second = decide(engine), decide(engine)
    assert first.choice == second.choice == "original"
    assert first.model == {"revision": "test"}
    assert first.elapsed_seconds >= 0
    loader = mlx.load_model if backend == "mlx" else core.load_causal_model
    assert loader.call_count == 1
    assert loader.call_args.args[:2] == (str(Path(temp_dir).resolve()), "test")
    assert direct.score.call_args.args[-1] == 256


def test_shared_engine_loads_once_for_concurrent_calls(temp_dir, local_backend):
    engine = SemIfDecisionEngine(temp_dir, "test")
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: decide(engine), range(8)))
    assert all(r.choice == "original" for r in results)
    local_backend[0].load_causal_model.assert_called_once()


def test_no_remote_model_identifiers():
    with pytest.raises(ValueError, match="existing local"):
        SemIfDecisionEngine("Qwen/Qwen3.5-4B", "test")


def test_removed_checkpoint_never_passed_to_upstream(tmp_path, local_backend):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    engine = SemIfDecisionEngine(str(checkpoint), "test")
    checkpoint.rmdir()
    with pytest.raises(FileNotFoundError):
        decide(engine)
    local_backend[0].load_causal_model.assert_not_called()


@pytest.mark.parametrize("probabilities", [[float("nan"), 0.2], [0.9, 0.9], [-0.1, 1.1], [0.5]])
def test_invalid_backend_scores_rejected(temp_dir, local_backend, probabilities):
    local_backend[1].score.return_value["probabilities"] = probabilities
    with pytest.raises(ValueError):
        decide(SemIfDecisionEngine(temp_dir, "test"))


def test_wrong_options_rejected(temp_dir, local_backend):
    local_backend[1].score.return_value["option_ids"] = ["expand", "original"]
    with pytest.raises(ValueError, match="option set"):
        decide(SemIfDecisionEngine(temp_dir, "test"))


def test_missing_optional_dependency_has_helpful_error(temp_dir):
    engine = SemIfDecisionEngine(temp_dir, "test")
    with patch("piragi.decisions.importlib.import_module", side_effect=ImportError):
        with pytest.raises(ImportError, match="Install SemIf"):
            decide(engine)


@pytest.fixture
def isolated_ragi():
    with (
        patch("piragi.core.create_store"),
        patch("piragi.core.EmbeddingGenerator"),
        patch("piragi.core.Chunker"),
        patch("piragi.llm_client.LLMClient"),
    ):
        yield


def test_ragi_constructs_adapter_without_loading(isolated_ragi, temp_dir, local_backend):
    kb = Ragi(
        config={
            "decision": {"enabled": True, "model_path": temp_dir, "backend": "mlx"},
            "auto_update": {"enabled": False},
        }
    )
    assert isinstance(kb.decision_engine, SemIfDecisionEngine)
    local_backend[2].load_model.assert_not_called()


def test_enabled_decisions_use_default_model(isolated_ragi):
    kb = Ragi(config={"decision": {"enabled": True}, "auto_update": {"enabled": False}})
    assert kb.decision_engine.model_path is None
    assert kb.decision_engine.revision == DEFAULT_REVISION


@pytest.mark.parametrize(
    "settings",
    [
        {"min_score": float("nan")},
        {"min_margin": 1.1},
        {"max_tokens": 0},
        {"backend": "cloud"},
        {"engine": "typo"},
        {"typo": True},
    ],
)
def test_invalid_config(settings):
    with pytest.raises(ValidationError):
        RagiConfig(decision=settings)


@pytest.mark.parametrize(
    "settings,enabled",
    [
        ({}, False),
        ({"engine": "openjev"}, True),
        ({"engine": "openjev", "enabled": False}, False),
        ({"enabled": True}, True),
    ],
)
def test_engine_selection_enables_judging(settings, enabled):
    original = settings.copy()
    config = RagiConfig(decision=settings)
    assert config.decision.enabled is enabled
    assert settings == original
    assert RagiConfig.model_validate(config.model_dump()).decision.enabled is enabled


@pytest.fixture
def hub(monkeypatch, temp_dir):
    module = ModuleType("huggingface_hub")
    module.snapshot_download = MagicMock(return_value=temp_dir)
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    return module


def test_short_config_does_not_load_on_empty_retrieval(isolated_ragi, local_backend, hub):
    kb = Ragi(config={"decision": {"engine": "openjev"}, "auto_update": {"enabled": False}})
    kb.store.search.return_value = []
    answer = kb.ask("What is top_k?")
    hub.snapshot_download.assert_not_called()
    kb.llm_client.complete.assert_not_called()
    assert answer.decision["status"] == "abstained"


def test_short_config_can_be_explicitly_disabled(isolated_ragi, hub):
    kb = Ragi(
        config={
            "decision": {"engine": "openjev", "enabled": False},
            "auto_update": {"enabled": False},
        }
    )
    kb.store.search.return_value = []
    kb.ask("question")
    assert kb.decision_engine is None
    hub.snapshot_download.assert_not_called()
    kb.llm_client.complete.assert_called_once()


def test_local_override_bypasses_download(temp_dir, local_backend, hub):
    engine = SemIfDecisionEngine(model_path=temp_dir, backend="torch")
    decide(engine)
    hub.snapshot_download.assert_not_called()
    assert engine.revision == "local"


def test_cached_only_and_cache_directory_passed_to_hub(temp_dir, local_backend, hub):
    engine = SemIfDecisionEngine(
        backend="torch",
        device="cpu",
        cache_dir=temp_dir,
        local_files_only=True,
    )
    decide(engine)
    assert hub.snapshot_download.call_args.kwargs["local_files_only"] is True
    assert hub.snapshot_download.call_args.kwargs["cache_dir"] == temp_dir


def test_missing_semif_does_not_download_default_model(hub):
    engine = SemIfDecisionEngine()
    real_import = __import__("importlib").import_module

    def missing_semif(name):
        if name.startswith("semif_phase1"):
            raise ImportError("Not installed")
        return real_import(name)

    with patch("piragi.decisions.importlib.import_module", side_effect=missing_semif):
        with pytest.raises(ImportError, match="Install SemIf"):
            decide(engine)
    hub.snapshot_download.assert_not_called()


def test_default_checkpoint_requires_immutable_revision():
    with pytest.raises(ValueError, match="pinned"):
        SemIfDecisionEngine(revision="main")


@pytest.fixture
def device_runtimes(monkeypatch):
    torch = MagicMock()
    torch.cuda.is_available.return_value = False
    torch.backends.mps.is_available.return_value = False
    torch.cuda.is_bf16_supported.return_value = False
    mx = MagicMock()
    mx.metal.is_available.return_value = True
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "mlx.core", mx)
    with (
        patch("piragi.decisions.platform.system", return_value="Linux"),
        patch("piragi.decisions.platform.machine", return_value="x86_64"),
        patch("piragi.decisions.importlib.util.find_spec", return_value=None),
    ):
        yield torch, mx


@pytest.mark.parametrize(
    "device,bf16,expected",
    [
        ("cpu", False, ("torch", "cpu", "float32")),
        ("mps", False, ("torch", "mps", "float16")),
        ("cuda", False, ("torch", "cuda", "float16")),
        ("cuda", True, ("torch", "cuda", "bfloat16")),
    ],
)
def test_automatic_torch_device_and_dtype(device_runtimes, device, bf16, expected):
    torch, _ = device_runtimes
    torch.cuda.is_available.return_value = device == "cuda"
    torch.backends.mps.is_available.return_value = device == "mps"
    torch.cuda.is_bf16_supported.return_value = bf16
    assert SemIfDecisionEngine()._runtime() == expected


@pytest.mark.parametrize(
    "mlx_installed,metal,expected",
    [
        (True, True, "mlx"),
        (False, True, "torch"),
        (True, False, "torch"),
    ],
)
def test_apple_silicon_prefers_available_mlx(device_runtimes, mlx_installed, metal, expected):
    torch, mx = device_runtimes
    torch.backends.mps.is_available.return_value = True
    mx.metal.is_available.return_value = metal
    with (
        patch("piragi.decisions.platform.system", return_value="Darwin"),
        patch("piragi.decisions.platform.machine", return_value="arm64"),
        patch(
            "piragi.decisions.importlib.util.find_spec",
            return_value=object() if mlx_installed else None,
        ),
    ):
        assert SemIfDecisionEngine()._runtime()[0] == expected


def test_explicit_runtime_overrides_auto_detection(device_runtimes):
    torch, _ = device_runtimes
    torch.cuda.is_available.return_value = True
    engine = SemIfDecisionEngine(backend="torch", device="cpu", dtype="float16")
    assert engine._runtime() == ("torch", "cpu", "float16")


@pytest.mark.asyncio
async def test_async_short_config(isolated_ragi, local_backend, hub):
    kb = AsyncRagi(config={"decision": {"engine": "openjev"}, "auto_update": {"enabled": False}})
    kb._sync.store.search.return_value = []
    answer = await kb.ask("What is top_k?")
    hub.snapshot_download.assert_not_called()
    kb._sync.llm_client.complete.assert_not_called()
    assert answer.decision["status"] == "abstained"


@pytest.mark.parametrize("backend", ["torch", "mlx"])
def test_batch_uses_native_shared_scorer(temp_dir, local_backend, backend):
    from piragi import DecisionQuestion

    engine = SemIfDecisionEngine(temp_dir, "test", backend=backend)
    questions = {
        "c0": DecisionQuestion(question="Evidence?", options={"yes": "Yes", "no": "No"}),
        "c1": DecisionQuestion(question="Other evidence?", options={"yes": "Yes", "no": "No"}),
    }
    results = engine.decide_many(state={"query": "question"}, questions=questions)
    assert list(results) == ["c0", "c1"]
    assert results["c0"].choice == "yes"
    assert results["c0"].elapsed_seconds == results["c1"].elapsed_seconds
    local_backend[2].score_shared.assert_called_once()
    local_backend[1].score.assert_not_called()
    if backend == "mlx":
        assert local_backend[2].load_model.call_args.kwargs == {"bits": 4}


@pytest.mark.parametrize("bad", [[], [{"id": "unexpected"}]])
def test_batch_rejects_missing_or_wrong_questions(temp_dir, local_backend, bad):
    from piragi import DecisionQuestion

    local_backend[2].score_shared.side_effect = None
    local_backend[2].score_shared.return_value = bad, {}
    with pytest.raises(ValueError, match="question set"):
        SemIfDecisionEngine(temp_dir).decide_many(
            state="query",
            questions={
                "c0": DecisionQuestion(question="Relevant?", options={"yes": "Yes", "no": "No"})
            },
        )


def test_evidence_preset_and_explicit_overrides():
    config = RagiConfig(decision={"engine": "openjev"})
    assert config.retrieval.use_hybrid_search
    assert not config.llm.enable_query_expansion
    assert config.decision.candidate_k == 20
    config = RagiConfig(
        decision={"engine": "openjev"},
        llm={"enable_query_expansion": True},
        retrieval={"use_hybrid_search": False},
    )
    assert config.llm.enable_query_expansion
    assert not config.retrieval.use_hybrid_search

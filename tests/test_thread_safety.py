import threading
import time
from unittest.mock import patch, MagicMock
from piragi.embeddings import EmbeddingGenerator
from piragi.core import Ragi
import pytest

def test_concurrent_model_loading():
    threads = []
    errors = []
    
    def load_model():
        try:
            embedder = EmbeddingGenerator(model="all-MiniLM-L6-v2")
            assert embedder.model is not None
        except Exception as e:
            errors.append(e)

    # Spawn 5 threads
    for _ in range(5):
        t = threading.Thread(target=load_model)
        threads.append(t)
        t.start()
        
    for t in threads:
        t.join()
        
    assert not errors, "Exceptions occurred during concurrent loading"

def test_cache_eviction_under_contention():
    # clear cache for clean test
    from piragi.embeddings import _model_cache, _cache_lock, _MODEL_CACHE_MAX
    with _cache_lock:
        _model_cache.clear()

    threads = []
    errors = []

    models = [
        "mock-model-a",
        "mock-model-b",
        "mock-model-c",
        "mock-model-d",
        "mock-model-e",
    ]

    def load_different_model(model_name):
        try:
            embedder = EmbeddingGenerator(model=model_name)
        except Exception as e:
            errors.append(e)

    with patch("sentence_transformers.SentenceTransformer") as mock_st:
        mock_st.return_value = MagicMock()

        for m in models:
            t = threading.Thread(target=load_different_model, args=(m,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

    assert not errors, "Errors: {}".format(errors)
    with _cache_lock:
        assert len(_model_cache) <= _MODEL_CACHE_MAX

def test_no_print_in_progress():
    # Use Ragi to add a mock document and ensure logger is used, not print
    ragi = Ragi(store="memory://", config={"auto_update": {"enabled": False}})
    
    with patch("piragi.core.logger") as mock_logger:
        with patch("builtins.print") as mock_print:
            with patch.object(ragi.loader, "load", return_value=[]):
                ragi.add([])
                
            # Assert logger.info was called
            assert mock_logger.info.called
            
            # Assert print was not called with [piragi]
            for call in mock_print.mock_calls:
                args = call[1]
                if args and isinstance(args[0], str) and "[piragi]" in args[0]:
                    pytest.fail("print('[piragi]') was called!")

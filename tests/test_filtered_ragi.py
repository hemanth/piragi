import threading
from unittest.mock import MagicMock
import pytest

from piragi.core import Ragi, FilteredRagi

class DummyRagi:
    def __init__(self):
        self.ask_calls = []
        self.retrieve_calls = []
        
    def _ask_with_filters(self, query, top_k=5, system_prompt=None, filters=None):
        self.ask_calls.append({"query": query, "filters": filters})
        from piragi.types import Answer
        return Answer(text="answer", citations=[], query=query)
        
    def _retrieve_with_filters(self, query, top_k=5, filters=None):
        self.retrieve_calls.append({"query": query, "filters": filters})
        return []
        
    def filter(self, **kwargs):
        return FilteredRagi(self, kwargs)

def test_filter_returns_new_object():
    ragi = DummyRagi()
    filtered = ragi.filter(type="doc")
    assert isinstance(filtered, FilteredRagi)
    assert filtered._ragi is ragi
    assert filtered._filters == {"type": "doc"}

def test_filter_chaining():
    ragi = DummyRagi()
    f1 = ragi.filter(a=1)
    f2 = f1.filter(b=2)
    assert isinstance(f2, FilteredRagi)
    assert f2._filters == {"a": 1, "b": 2}
    assert f1._filters == {"a": 1}

def test_concurrent_filters():
    ragi = DummyRagi()
    
    def worker(i):
        filtered = ragi.filter(worker_id=i)
        import time
        time.sleep(0.01)
        filtered.ask("hello")
        
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
        
    filters_used = [call["filters"]["worker_id"] for call in ragi.ask_calls]
    assert sorted(filters_used) == list(range(10))

def test_original_unmodified():
    ragi = DummyRagi()
    filtered = ragi.filter(a=1)
    assert not hasattr(ragi, "_filters")

def test_filter_passes_to_store():
    ragi = DummyRagi()
    f = ragi.filter(foo="bar")
    f.ask("test")
    assert ragi.ask_calls[-1]["filters"] == {"foo": "bar"}
    
    f.retrieve("test2")
    assert ragi.retrieve_calls[-1]["filters"] == {"foo": "bar"}

import time
import pytest
from piragi.retry import retry_with_backoff

def test_succeeds_first_try():
    calls = 0
    def fn():
        nonlocal calls
        calls += 1
        return "success"
    
    assert retry_with_backoff(fn) == "success"
    assert calls == 1

def test_retries_on_failure(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda x: None)
    calls = 0
    def fn():
        nonlocal calls
        calls += 1
        if calls < 3:
            raise ConnectionError("fail")
        return "success"
    
    assert retry_with_backoff(fn, retryable_exceptions=(ConnectionError,)) == "success"
    assert calls == 3

def test_exhausts_retries(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda x: None)
    calls = 0
    def fn():
        nonlocal calls
        calls += 1
        raise TimeoutError("fail")
    
    with pytest.raises(TimeoutError):
        retry_with_backoff(fn, max_retries=2, retryable_exceptions=(TimeoutError,))
    assert calls == 3

def test_backoff_delays(monkeypatch):
    delays = []
    def mock_sleep(d):
        delays.append(d)
        
    monkeypatch.setattr(time, "sleep", mock_sleep)
    
    def fn():
        raise ConnectionError("fail")
        
    with pytest.raises(ConnectionError):
        retry_with_backoff(fn, max_retries=3, base_delay=1.0, retryable_exceptions=(ConnectionError,))
        
    assert delays == [1.0, 2.0, 4.0]

def test_max_delay_cap(monkeypatch):
    delays = []
    def mock_sleep(d):
        delays.append(d)
        
    monkeypatch.setattr(time, "sleep", mock_sleep)
    
    def fn():
        raise ConnectionError("fail")
        
    with pytest.raises(ConnectionError):
        retry_with_backoff(fn, max_retries=4, base_delay=2.0, max_delay=5.0, retryable_exceptions=(ConnectionError,))
        
    assert delays == [2.0, 4.0, 5.0, 5.0]

def test_non_retryable_exception(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda x: None)
    calls = 0
    def fn():
        nonlocal calls
        calls += 1
        raise ValueError("bad request")
        
    with pytest.raises(ValueError):
        retry_with_backoff(fn, max_retries=3, retryable_exceptions=(ConnectionError,))
        
    assert calls == 1

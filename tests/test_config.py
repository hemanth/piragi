import pytest
from pydantic import ValidationError
from piragi.config import RagiConfig, LLMConfig

def test_default_config():
    config = RagiConfig()
    assert config.llm.model == "llama3.2"
    assert config.chunk.strategy == "fixed"
    assert config.retrieval.use_hyde is False

def test_partial_override():
    config = RagiConfig(llm={"model": "gpt-4"})
    assert config.llm.model == "gpt-4"
    assert config.llm.temperature == 0.1

def test_nested_override():
    config = RagiConfig(
        llm={"model": "gpt-4o", "temperature": 0.5},
        chunk={"size": 1024, "strategy": "semantic"}
    )
    assert config.llm.model == "gpt-4o"
    assert config.llm.temperature == 0.5
    assert config.chunk.size == 1024
    assert config.chunk.strategy == "semantic"

def test_invalid_type_rejected():
    with pytest.raises(ValidationError):
        RagiConfig(llm={"temperature": "not a float"})

def test_extra_keys_rejected():
    with pytest.raises(ValidationError):
        RagiConfig(llm={"typo_key": True})

def test_backward_compat_dict():
    # If a user passes a plain dict to Ragi, it should be wrapped.
    # RagiConfig(**config) handles this.
    config_dict = {"llm": {"model": "gpt-3.5-turbo"}}
    config = RagiConfig(**config_dict)
    assert config.llm.model == "gpt-3.5-turbo"

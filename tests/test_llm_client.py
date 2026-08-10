import pytest
from unittest.mock import MagicMock, patch
from piragi.llm_client import LLMClient
from piragi.core import Ragi
import openai

def test_llm_client_init():
    client = LLMClient()
    assert client.model == "llama3.2"
    assert client.temperature == 0.1
    assert client.client is not None
    assert client.client.base_url == "http://localhost:11434/v1/"
    assert client.client.api_key == "ollama"

def test_llm_client_custom_config():
    client = LLMClient(model="gpt-4", api_key="test-key", base_url="https://api.openai.com/v1", temperature=0.5)
    assert client.model == "gpt-4"
    assert client.temperature == 0.5
    assert client.client.base_url == "https://api.openai.com/v1/"
    assert client.client.api_key == "test-key"

@patch('openai.OpenAI')
def test_complete_text(mock_openai):
    mock_instance = MagicMock()
    mock_openai.return_value = mock_instance
    
    mock_response = MagicMock()
    mock_message = MagicMock()
    mock_message.message.content = "Test response"
    mock_response.choices = [mock_message]
    
    mock_instance.chat.completions.create.return_value = mock_response
    
    client = LLMClient(api_key="test")
    client._client = mock_instance  # inject mock
    
    response_text = client.complete_text("Hello", system_prompt="You are an AI", temperature=0.7)
    
    assert response_text == "Test response"
    mock_instance.chat.completions.create.assert_called_once_with(
        model="llama3.2",
        messages=[
            {"role": "system", "content": "You are an AI"},
            {"role": "user", "content": "Hello"}
        ],
        temperature=0.7
    )

@patch('piragi.llm_client.retry_with_backoff')
@patch('openai.OpenAI')
def test_complete_with_retry(mock_openai, mock_retry):
    # Actually test retry logic is applied
    pass

def test_complete_with_retry_actual():
    client = LLMClient(api_key="test")
    mock_instance = MagicMock()
    client._client = mock_instance
    
    fail_count = [0]
    
    def side_effect(*args, **kwargs):
        if fail_count[0] < 2:
            fail_count[0] += 1
            raise openai.APIConnectionError(request=MagicMock())
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Success"))]
        return mock_response
        
    mock_instance.chat.completions.create.side_effect = side_effect
    
    # Needs to be tested with retry_with_backoff
    # the decorator is already applied inside complete
    
    # Set retry delays to 0 for fast test
    import piragi.retry
    original_sleep = piragi.retry.time.sleep
    piragi.retry.time.sleep = lambda x: None
    
    try:
        response = client.complete([{"role": "user", "content": "hi"}])
        assert response.choices[0].message.content == "Success"
        assert fail_count[0] == 2
        assert mock_instance.chat.completions.create.call_count == 3
    finally:
        piragi.retry.time.sleep = original_sleep

def test_shared_client_in_ragi():
    ragi = Ragi()
    assert ragi.llm_client is not None
    assert ragi.retriever.llm_client is ragi.llm_client
    
    # check others if initialized
    if ragi._hyde:
        assert ragi._hyde.llm_client is ragi.llm_client
    
    # check chunker if it's ContextualChunker
    from piragi.semantic_chunking import ContextualChunker
    if isinstance(ragi.chunker, ContextualChunker):
        assert ragi.chunker.llm_client is ragi.llm_client

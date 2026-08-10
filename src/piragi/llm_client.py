import logging
from typing import Optional, List, Dict, Any
from .retry import retry_with_backoff

logger = logging.getLogger(__name__)

class LLMClient:
    """Unified LLM client with connection pooling and retry logic."""
    
    def __init__(self, model="llama3.2", api_key=None, base_url=None, temperature=0.1):
        import openai
        self.model = model
        self.temperature = temperature
        self._client = openai.OpenAI(
            api_key=api_key or "ollama",
            base_url=base_url or "http://localhost:11434/v1",
        )
    
    def complete(self, messages, temperature=None, model=None, **kwargs):
        """Send a chat completion request with retry."""
        import openai
        
        @retry_with_backoff(exceptions=(
            ConnectionError, TimeoutError,
            openai.APIConnectionError, openai.APITimeoutError,
        ))
        def _do_request():
            return self._client.chat.completions.create(
                model=model or self.model,
                messages=messages,
                temperature=temperature if temperature is not None else self.temperature,
                **kwargs
            )
        return _do_request()
    
    def complete_text(self, prompt, system_prompt=None, temperature=None, **kwargs):
        """Convenience: send a text prompt, get text back."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        response = self.complete(messages, temperature=temperature, **kwargs)
        return response.choices[0].message.content
    
    @property
    def client(self):
        """Access the underlying OpenAI client for direct use."""
        return self._client

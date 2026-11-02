import logging
import threading
import time
from typing import Optional, List, Dict, Any
from .retry import retry_with_backoff
from .telemetry import classify_exception, classify_provider, get_telemetry, outbound_headers

logger = logging.getLogger(__name__)

class LLMClient:
    """Unified LLM client with connection pooling, retry logic, and dynamic ADC token refresh."""
    
    def __init__(
        self,
        model: str = "llama3.2",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.1,
        provider: Optional[str] = None,
        project_id: Optional[str] = None,
        location: Optional[str] = None,
        telemetry: Any = None,
    ):
        import openai
        self.model = model
        self.temperature = temperature
        self.provider = provider
        self.project_id = project_id
        self.location = location or "us-central1"
        self._base_url = base_url or "http://localhost:11434/v1"
        self._provider_label = classify_provider(base_url=base_url, provider=provider)
        self._telemetry = telemetry
        self._token_lock = threading.Lock()
        self._credentials = None
        self._use_adc_refresh = False

        # Support dynamic Google ADC OAuth2 Bearer token refresh for Vertex AI / GCP endpoints
        is_vertex = (
            (provider is not None and provider.lower() in ("vertexai", "vertex", "gcp"))
            or (base_url is not None and "googleapis.com" in base_url)
        )
        if is_vertex:
            self._use_adc_refresh = True
            self._provider_label = "vertexai"
            try:
                import google.auth
                from google.auth.transport.requests import Request
                self._credentials, detected_proj = google.auth.default(
                    scopes=["https://www.googleapis.com/auth/cloud-platform"]
                )
                if not self.project_id and detected_proj:
                    self.project_id = detected_proj
                if not self._credentials.valid:
                    self._credentials.refresh(Request())
                if self._credentials.token and not api_key:
                    api_key = self._credentials.token
                if base_url is None:
                    proj = self.project_id or "hemanth-hm"
                    base_url = "https://{}-aiplatform.googleapis.com/v1beta1/projects/{}/locations/{}/endpoints/openapi".format(
                        self.location, proj, self.location
                    )
                    self._base_url = base_url
            except Exception as e:
                logger.warning("Could not initialize Vertex AI ADC credentials: %s", e)

        self._client = openai.OpenAI(
            api_key=api_key or "ollama",
            base_url=base_url or "http://localhost:11434/v1",
        )

    def _get_telemetry(self):
        return self._telemetry if self._telemetry is not None else get_telemetry()

    def _refresh_token_if_needed(self) -> None:
        """Dynamically refresh OAuth2 Bearer token using Google ADC if needed."""
        if not self._use_adc_refresh:
            return
        with self._token_lock:
            try:
                import google.auth
                from google.auth.transport.requests import Request
                if self._credentials is None:
                    self._credentials, detected_proj = google.auth.default(
                        scopes=["https://www.googleapis.com/auth/cloud-platform"]
                    )
                    if not self.project_id and detected_proj:
                        self.project_id = detected_proj
                if not self._credentials.valid:
                    self._credentials.refresh(Request())
                if self._credentials.token:
                    self._client.api_key = self._credentials.token
            except Exception as e:
                logger.warning("Failed to refresh ADC token: %s", e)
    
    def complete(self, messages, temperature=None, model=None, **kwargs):
        """Send a chat completion request with retry, trace propagation, and RED telemetry."""
        import openai
        self._refresh_token_if_needed()
        tel = self._get_telemetry()
        target_model = model or self.model

        # Propagate correlation ID and W3C traceparent to external LLM provider
        call_kwargs = dict(kwargs)
        prop_headers = outbound_headers()
        if prop_headers:
            existing_headers = dict(call_kwargs.get("extra_headers") or {})
            for hk, hv in prop_headers.items():
                existing_headers.setdefault(hk, hv)
            call_kwargs["extra_headers"] = existing_headers

        attempt_counter = {"count": 0}
        retry_types = (
            ConnectionError, TimeoutError,
            openai.APIConnectionError, openai.APITimeoutError,
            openai.RateLimitError, openai.InternalServerError,
        )

        with tel.span(
            "llm.complete",
            operation="llm_complete",
            provider=self._provider_label,
            modelFamily=str(target_model).split("/")[0][:32],
        ) as span_attrs:
            @retry_with_backoff(exceptions=retry_types)
            def _do_request():
                attempt_counter["count"] += 1
                t0 = time.perf_counter()
                try:
                    return self._client.chat.completions.create(
                        model=target_model,
                        messages=messages,
                        temperature=temperature if temperature is not None else self.temperature,
                        **call_kwargs
                    )
                except retry_types as exc:
                    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)
                    tel.log(
                        "warn",
                        "llm_retry",
                        operation="llm_complete",
                        provider=self._provider_label,
                        statusClass=classify_exception(exc),
                        durationMs=elapsed_ms,
                        attempt=attempt_counter["count"],
                        errorType=type(exc).__name__,
                    )
                    raise

            result = _do_request()
            span_attrs["attempt"] = attempt_counter["count"]
            return result
    
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
        self._refresh_token_if_needed()
        return self._client

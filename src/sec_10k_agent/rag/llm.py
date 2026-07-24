"""Text generation backend.

xAI Grok, OpenRouter, and a local Ollama server all speak the OpenAI
chat-completions API, so a single OpenAI-SDK client covers all three — only
base_url, model, and key differ. That keeps "which provider generates" a config
decision, not a code change (see ADR on provider swap).

The pipeline depends on the small `Generator` protocol, not on this concrete
class, so tests inject a fake and never touch the network.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from sec_10k_agent.config import Settings, get_settings
from sec_10k_agent.rag.models import LLMResponse

if TYPE_CHECKING:
    from openai import OpenAI


@runtime_checkable
class Generator(Protocol):
    """Anything that can turn a (system, user) pair into an LLMResponse."""

    def complete(self, system: str, user: str, *, temperature: float = 0.0) -> LLMResponse: ...


class OpenAICompatGenerator:
    """OpenAI-chat-completions generator. Points at Grok or Ollama by config."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        client: OpenAI | None = None,
        timeout: float = 600.0,
    ) -> None:
        self._model = model
        if client is None:
            from openai import OpenAI as _OpenAI

            # Ollama needs no real key but the SDK requires a non-empty string.
            client = _OpenAI(api_key=api_key or "not-needed", base_url=base_url, timeout=timeout)
        self._client = client

    def complete(self, system: str, user: str, *, temperature: float = 0.0) -> LLMResponse:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
        )
        choice = resp.choices[0].message.content or ""
        usage = resp.usage
        return LLMResponse(
            text=choice,
            model=resp.model or self._model,
            prompt_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
            completion_tokens=getattr(usage, "completion_tokens", None) if usage else None,
        )


def build_generator(settings: Settings | None = None) -> OpenAICompatGenerator:
    """Construct the generator from config.

    Precedence: xAI Grok (`xai_api_key`) -> OpenRouter (`openrouter_api_key`)
    -> local Ollama (`ollama_base_url`). Raises if none is available so the
    failure is loud rather than a confusing API error.
    """
    settings = settings or get_settings()
    if settings.xai_api_key:
        return OpenAICompatGenerator(
            model=settings.xai_model,
            api_key=settings.xai_api_key,
            base_url=settings.xai_base_url,
            timeout=settings.llm_timeout_seconds,
        )
    if settings.openrouter_api_key:
        return OpenAICompatGenerator(
            model=settings.openrouter_model,
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
            timeout=settings.llm_timeout_seconds,
        )
    if settings.ollama_base_url:
        # Ollama's OpenAI-compatible endpoint lives under /v1.
        base = settings.ollama_base_url.rstrip("/")
        if not base.endswith("/v1"):
            base = f"{base}/v1"
        return OpenAICompatGenerator(
            model=settings.ollama_model,
            api_key="ollama",
            base_url=base,
            timeout=settings.llm_timeout_seconds,
        )
    raise RuntimeError(
        "No generator configured. Set XAI_API_KEY (Grok), OPENROUTER_API_KEY, "
        "or OLLAMA_BASE_URL (local) in .env."
    )

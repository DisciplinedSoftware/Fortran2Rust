from __future__ import annotations

from .base import LLMClient


def get_llm_client(provider: str, model: str, **keys) -> LLMClient:
    if provider == "openai":
        from .openai_client import OpenAIClient
        return OpenAIClient(api_key=keys["openai_api_key"], model=model)
    elif provider == "anthropic":
        from .anthropic_client import AnthropicClient
        return AnthropicClient(
            api_key=keys["anthropic_api_key"],
            model=model,
            max_tokens=keys.get("llm_max_tokens", 16384),
        )
    elif provider == "google":
        from .google_client import GoogleClient
        return GoogleClient(api_key=keys["google_api_key"], model=model)
    elif provider == "openrouter":
        from .openrouter_client import OpenRouterClient
        return OpenRouterClient(api_key=keys["openrouter_api_key"], model=model)
    elif provider == "ollama":
        from .ollama_client import OllamaClient
        return OllamaClient(base_url=keys.get("ollama_base_url", "http://localhost:11434"), model=model)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")

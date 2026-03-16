from __future__ import annotations

import anthropic

from .base import LLMClient


class AnthropicClient(LLMClient):
    def __init__(self, api_key: str, model: str = "claude-opus-4-5"):
        super().__init__()
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def _call_llm(self, system: str, user: str) -> str:
        # Mark the system prompt for server-side caching. Anthropic applies the
        # cache when the prompt prefix exceeds the minimum threshold (1024 tokens
        # for most models). Short prompts are silently ignored by the server, so
        # this is safe to include unconditionally.
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=8192,
            system=[
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user}],
        )
        input_tokens = msg.usage.input_tokens
        # cache_read_input_tokens is present when prompt caching is active.
        cache_read = getattr(msg.usage, "cache_read_input_tokens", 0) or 0
        self._record_usage(input_tokens, msg.usage.output_tokens)
        if cache_read:
            from rich.console import Console
            Console(stderr=True).print(
                f"  [dim]  (Anthropic cache hit: {cache_read:,} cached tokens)[/dim]"
            )
        return msg.content[0].text

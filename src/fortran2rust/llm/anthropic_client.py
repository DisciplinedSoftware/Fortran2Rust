from __future__ import annotations

import anthropic

from .base import LLMClient


class AnthropicClient(LLMClient):
    def __init__(self, api_key: str, model: str = "claude-opus-4-5"):
        super().__init__()
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def complete(self, system: str, user: str) -> str:
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=8192,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        self._record_usage(msg.usage.input_tokens, msg.usage.output_tokens)
        return msg.content[0].text

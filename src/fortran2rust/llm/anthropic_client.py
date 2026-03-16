from __future__ import annotations

import anthropic

from .base import LLMClient


class AnthropicClient(LLMClient):
    def __init__(self, api_key: str, model: str = "claude-opus-4-5", max_tokens: int = 16384):
        super().__init__()
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens

    def _call_llm(self, system: str, user: str) -> str:
        # Mark the system prompt for server-side caching. Anthropic applies the
        # cache when the prompt prefix exceeds the minimum threshold (1024 tokens
        # for most models). Short prompts are silently ignored by the server, so
        # this is safe to include unconditionally.
        #
        # When the user message follows the CODE / ERROR / CONTEXT structure
        # produced by LLMClient.repair(), also cache the code section: it is the
        # largest and most stable part of the prompt, while the error + context
        # tail changes on every retry.  Splitting on the ERROR marker preserves
        # the boundary without requiring the caller to pass structured content.
        _SPLIT = "\n\nERROR:\n"
        if _SPLIT in user:
            code_prefix, tail = user.split(_SPLIT, 1)
            user_blocks: list[dict] = [
                {
                    "type": "text",
                    "text": code_prefix + _SPLIT,
                    "cache_control": {"type": "ephemeral"},
                },
                {"type": "text", "text": tail},
            ]
        else:
            user_blocks = [{"type": "text", "text": user}]

        msg = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=[
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user_blocks}],
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

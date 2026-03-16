from __future__ import annotations

from openai import OpenAI

from .base import LLMClient


class GitHubModelsClient(LLMClient):
    """GitHub Models API client using Azure endpoint (OpenAI-compatible)."""

    def __init__(self, api_key: str, model: str = "gpt-4o"):
        super().__init__()
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://models.inference.ai.azure.com",
        )
        self.model = model

    def _call_llm(self, system: str, user: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        if resp.usage:
            self._record_usage(resp.usage.prompt_tokens, resp.usage.completion_tokens)
        return resp.choices[0].message.content or ""

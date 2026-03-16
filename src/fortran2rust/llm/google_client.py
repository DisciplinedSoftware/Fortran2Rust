from __future__ import annotations

import google.generativeai as genai

from .base import LLMClient


class GoogleClient(LLMClient):
    def __init__(self, api_key: str, model: str = "gemini-1.5-pro"):
        super().__init__()
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model, system_instruction=None)
        self._model_name = model

    def _call_llm(self, system: str, user: str) -> str:
        model = genai.GenerativeModel(self._model_name, system_instruction=system)
        resp = model.generate_content(user)
        if resp.usage_metadata:
            self._record_usage(
                resp.usage_metadata.prompt_token_count,
                resp.usage_metadata.candidates_token_count,
            )
        return resp.text

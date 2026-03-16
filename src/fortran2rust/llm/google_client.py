from __future__ import annotations

import google.generativeai as genai

from .base import LLMClient


class GoogleClient(LLMClient):
    def __init__(self, api_key: str, model: str = "gemini-1.5-pro"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model, system_instruction=None)
        self._model_name = model

    def complete(self, system: str, user: str) -> str:
        model = genai.GenerativeModel(self._model_name, system_instruction=system)
        resp = model.generate_content(user)
        return resp.text

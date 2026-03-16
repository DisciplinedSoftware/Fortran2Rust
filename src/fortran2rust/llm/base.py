from __future__ import annotations

from abc import ABC, abstractmethod


class LLMClient(ABC):
    @abstractmethod
    def complete(self, system: str, user: str) -> str:
        """Send a chat completion and return the assistant's text."""
        ...

    def repair(self, context: str, error: str, code: str) -> str:
        """Standard repair prompt for fixing broken code."""
        system = (
            "You are an expert systems programmer. You will be given code and an error. "
            "Return ONLY the corrected complete file(s), no explanations, no markdown fences."
        )
        user = f"ERROR:\n{error}\n\nCODE:\n{code}\n\nCONTEXT:\n{context}"
        return self.complete(system, user)

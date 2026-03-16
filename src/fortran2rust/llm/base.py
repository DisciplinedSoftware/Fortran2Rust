from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from rich.console import Console

_console = Console(stderr=True)


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __str__(self) -> str:
        return (
            f"tokens: [bold]{self.prompt_tokens:,}[/bold] in / "
            f"[bold]{self.completion_tokens:,}[/bold] out "
            f"([bold]{self.total_tokens:,}[/bold] total)"
        )


class LLMClient(ABC):
    def __init__(self) -> None:
        self.last_usage: TokenUsage = TokenUsage()
        self._conversation_log: list[dict] = []

    def _record_usage(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.last_usage = TokenUsage(prompt_tokens, completion_tokens)
        _console.print(f"  [dim]↳ {self.last_usage}[/dim]")

    @abstractmethod
    def _call_llm(self, system: str, user: str) -> str:
        """Call the LLM API and return the assistant's response text."""
        ...

    def complete(self, system: str, user: str) -> str:
        """Send a chat completion, log the full conversation, and return the response."""
        response = self._call_llm(system, user)
        self._conversation_log.append({
            "system": system,
            "user": user,
            "response": response,
            "prompt_tokens": self.last_usage.prompt_tokens,
            "completion_tokens": self.last_usage.completion_tokens,
        })
        return response

    def pop_conversation_log(self) -> list[dict]:
        """Return all conversations recorded since the last call and reset the log."""
        log = self._conversation_log
        self._conversation_log = []
        return log

    def repair(self, context: str, error: str, code: str) -> str:
        """Standard repair prompt for fixing broken code."""
        system = (
            "You are an expert systems programmer. You will be given code and an error. "
            "Return ONLY the corrected complete file(s), no explanations, no markdown fences."
        )
        user = f"ERROR:\n{error}\n\nCODE:\n{code}\n\nCONTEXT:\n{context}"
        return self.complete(system, user)

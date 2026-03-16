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

    def _record_usage(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.last_usage = TokenUsage(prompt_tokens, completion_tokens)
        _console.print(f"  [dim]↳ {self.last_usage}[/dim]")

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

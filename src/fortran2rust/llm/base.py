from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from rich.console import Console

_MAX_ERROR_LINES = 60


def _truncate_error(error: str, max_lines: int = _MAX_ERROR_LINES) -> str:
    """Cap compiler error output so it doesn't dominate the repair prompt."""
    lines = error.splitlines()
    if len(lines) <= max_lines:
        return error
    return "\n".join(lines[:max_lines]) + f"\n[... {len(lines) - max_lines} more lines truncated ...]"

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
        # Thread-local storage so concurrent calls each track their own usage.
        self._local = threading.local()
        self._conversation_log: list[dict] = []
        self._log_lock = threading.Lock()

    @property
    def last_usage(self) -> TokenUsage:
        return getattr(self._local, "last_usage", TokenUsage())

    def _record_usage(self, prompt_tokens: int, completion_tokens: int) -> None:
        usage = TokenUsage(prompt_tokens, completion_tokens)
        self._local.last_usage = usage
        _console.print(f"  [dim]↳ {usage}[/dim]")

    @abstractmethod
    def _call_llm(self, system: str, user: str) -> str:
        """Call the LLM API and return the assistant's response text."""
        ...

    def complete(self, system: str, user: str) -> str:
        """Send a chat completion, log the full conversation, and return the response."""
        response = self._call_llm(system, user)
        # Capture usage immediately after the call (thread-local, safe) before
        # taking the lock so API latency is never held under the lock.
        usage = self.last_usage
        with self._log_lock:
            self._conversation_log.append({
                "system": system,
                "user": user,
                "response": response,
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
            })
        return response

    def pop_conversation_log(self) -> list[dict]:
        """Return all conversations recorded since the last call and reset the log."""
        with self._log_lock:
            log = self._conversation_log
            self._conversation_log = []
        return log

    def repair(self, context: str, error: str, code: str, attempt: int = 0) -> str:
        """Standard repair prompt for fixing broken code.

        *attempt* is zero-indexed; when > 0 a hint is added to the prompt
        telling the model that previous attempts failed so it should try a
        different approach.
        """
        system = (
            "You are an expert systems programmer. You will be given code and an error. "
            "Return ONLY the corrected complete file(s), no explanations, no markdown fences."
        )
        # CODE first: it is the largest stable portion and benefits most from
        # prefix-based prompt caching (e.g. Anthropic's cache_control).
        user = f"CODE:\n{code}\n\nERROR:\n{_truncate_error(error)}\n\nCONTEXT:\n{context}"
        if attempt > 0:
            user += (
                f"\n\nNOTE: {attempt} previous attempt(s) failed to fix this. "
                "Try a different approach and be more thorough."
            )
        return self.complete(system, user)

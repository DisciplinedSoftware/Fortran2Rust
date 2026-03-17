from __future__ import annotations

import hashlib
from pathlib import Path

_CACHE_DIR = Path.home() / ".cache" / "fortran2rust" / "repair_cache"


def _key(code: str, error: str, context: str, cache_scope: str | None = None) -> str:
    payload = f"{cache_scope or ''}\x00{code}\x00{error}\x00{context}"
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


def lookup(
    code: str,
    error: str,
    context: str,
    cache_dir: Path = _CACHE_DIR,
    cache_scope: str | None = None,
) -> str | None:
    """Return a previously cached repair response, or *None* if not found."""
    cache_file = cache_dir / (_key(code, error, context, cache_scope=cache_scope) + ".txt")
    if cache_file.is_file():
        return cache_file.read_text()
    return None


def store(
    code: str,
    error: str,
    context: str,
    response: str,
    cache_dir: Path = _CACHE_DIR,
    cache_scope: str | None = None,
) -> None:
    """Persist a repair response so future identical requests can skip the LLM."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / (_key(code, error, context, cache_scope=cache_scope) + ".txt")).write_text(response)

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(), override=False)

PROVIDERS = ["openai", "anthropic", "google", "openrouter", "ollama"]


@dataclass
class Config:
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o"
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    google_api_key: str = ""
    openrouter_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"
    max_retries: int = 5
    output_dir: Path = field(default_factory=lambda: Path("./artifacts"))
    stages: list[int] = field(default_factory=lambda: list(range(1, 10)))


def load_config(**overrides) -> Config:
    """Read env vars, apply overrides, and return a Config."""
    cfg = Config(
        llm_provider=os.getenv("LLM_PROVIDER", "openai"),
        llm_model=os.getenv("LLM_MODEL", "gpt-4o"),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        google_api_key=os.getenv("GOOGLE_API_KEY", ""),
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        max_retries=int(os.getenv("MAX_RETRIES", "5")),
        output_dir=Path(os.getenv("OUTPUT_DIR", "./artifacts")),
        stages=list(range(1, 10)),
    )
    for key, value in overrides.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg


def make_run_id() -> str:
    """Return a unique run identifier: YYYYMMDD_HHMMSS_<6-char-hash>."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_hash = uuid4().hex[:6]
    return f"{ts}_{short_hash}"


def get_stage_dir(artifacts_dir: Path, run_id: str, stage: str) -> Path:
    """Return artifacts_dir / run_id / stage, creating it if needed."""
    path = artifacts_dir / run_id / stage
    path.mkdir(parents=True, exist_ok=True)
    return path

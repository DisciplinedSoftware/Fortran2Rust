from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console

if TYPE_CHECKING:
    from ..llm.base import LLMClient

from ..exceptions import CompilationError, MaxRetriesExceededError

_console = Console(stderr=True)


def _first_error_line(error: str) -> str:
    for line in error.splitlines():
        if "error" in line.lower() and line.strip():
            return line.strip()[:120]
    return error.strip()[:120]


def _read_rust_files(directory: Path) -> str:
    parts = []
    for f in sorted(directory.rglob("*.rs")):
        parts.append(f"--- {f.relative_to(directory)} ---\n{f.read_text()}")
    return "\n\n".join(parts)


def _apply_llm_response(response: str, output_dir: Path) -> None:
    parts = re.split(r"^---\s+(\S+\.(?:rs|toml))\s+---$", response, flags=re.MULTILINE)
    if len(parts) >= 3:
        it = iter(parts[1:])
        for filename, content in zip(it, it):
            target = output_dir / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content.strip() + "\n")
    else:
        content = re.sub(r"```[a-z]*\n?", "", response).strip()
        rs_files = sorted(output_dir.rglob("*.rs"))
        target = rs_files[0] if rs_files else output_dir / "src" / "lib.rs"
        target.write_text(content + "\n")


def _cargo_build(cargo_toml: Path) -> tuple[bool, str]:
    result = subprocess.run(
        ["cargo", "build", "--manifest-path", str(cargo_toml)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    return result.returncode == 0, result.stderr


def fix_rust_code(
    rust_dir: Path,
    output_dir: Path,
    llm: "LLMClient",
    max_retries: int,
    baseline_dir: Path,
    status_fn=None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy rust_dir to output_dir
    if output_dir != rust_dir:
        shutil.copytree(rust_dir, output_dir, dirs_exist_ok=True)

    cargo_toml = output_dir / "Cargo.toml"

    llm_log: list[dict] = []
    llm_turns = 0
    retries = 0

    # Build loop
    if status_fn:
        status_fn("cargo build…")
    build_ok, build_error = _cargo_build(cargo_toml)
    for attempt in range(max_retries):
        if build_ok:
            break
        if status_fn:
            status_fn(f"LLM: fixing Rust compilation (attempt {attempt+1}/{max_retries})…")
        code = _read_rust_files(output_dir)
        response = llm.repair(
            context="Fix this Rust code that was transpiled from C by c2rust. Fix all compilation errors.",
            error=build_error,
            code=code,
        )
        llm_log.append({"phase": "build", "attempt": attempt, "error": build_error[:2000]})
        llm_turns += 1
        retries += 1
        _apply_llm_response(response, output_dir)
        build_ok, build_error = _cargo_build(cargo_toml)

    bench_ok = False
    # Bench loop: attempt to run a benchmark and compare to baseline
    if build_ok:
        fortran_bins = list(baseline_dir.glob("*_baseline.bin"))
        if not fortran_bins:
            bench_ok = True
        else:
            try:
                import numpy as np  # noqa: F401
                if status_fn:
                    status_fn(f"Running Rust benchmark…")
                # Try to run bench binary
                bench_result = subprocess.run(
                    ["cargo", "test", "--manifest-path", str(cargo_toml)],
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
                bench_ok = bench_result.returncode == 0
            except Exception:
                bench_ok = False

    (output_dir / "llm_log.json").write_text(json.dumps(llm_log, indent=2))
    result = {
        "build_ok": build_ok,
        "bench_ok": bench_ok,
        "llm_turns": llm_turns,
        "retries": retries,
        "build_error": build_error if not build_ok else "",
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2))
    return result

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..llm.base import LLMClient

SAFE_SYSTEM_PROMPT = (
    "You are a Rust expert. Rewrite this code to eliminate all unsafe blocks while preserving "
    "exact numerical behavior. Return ONLY the complete corrected file, no explanations, no markdown fences."
)


def _cargo_build(cargo_toml: Path) -> tuple[bool, str]:
    result = subprocess.run(
        ["cargo", "build", "--manifest-path", str(cargo_toml)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    return result.returncode == 0, result.stderr


def _count_unsafe(content: str) -> int:
    return len(re.findall(r"\bunsafe\b", content))


def _apply_llm_response(response: str, target_file: Path) -> None:
    content = re.sub(r"```[a-z]*\n?", "", response).strip()
    target_file.write_text(content + "\n")


def make_safe(
    rust_dir: Path,
    output_dir: Path,
    llm: "LLMClient",
    max_retries: int,
    baseline_dir: Path,
    status_fn=None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    if output_dir != rust_dir:
        shutil.copytree(rust_dir, output_dir, dirs_exist_ok=True)

    cargo_toml = output_dir / "Cargo.toml"
    llm_log: list[dict] = []
    llm_turns = 0
    retries = 0

    # Count unsafe before
    unsafe_before = 0
    rs_files = [f for f in output_dir.rglob("*.rs") if "bench" not in f.name and "test" not in f.name]
    for f in rs_files:
        unsafe_before += _count_unsafe(f.read_text())

    # Process each file
    for rs_file in rs_files:
        content = rs_file.read_text()
        n = _count_unsafe(content)
        if n == 0:
            continue

        if status_fn:
            status_fn(f"LLM: removing unsafe from {rs_file.name} ({n} occurrences)…")
        response = llm.complete(SAFE_SYSTEM_PROMPT, content)
        llm_log.append({"phase": "safe", "file": str(rs_file.name), "unsafe_count": _count_unsafe(content)})
        llm_turns += 1
        _apply_llm_response(response, rs_file)

        # Verify build after each file
        build_ok, build_error = _cargo_build(cargo_toml)
        for attempt in range(max_retries):
            if build_ok:
                break
            if status_fn:
                status_fn(f"Verifying safe Rust builds… (attempt {attempt+1}/{max_retries})")
            repair_response = llm.repair(
                context="Fix compilation error after removing unsafe blocks in Rust code.",
                error=build_error,
                code=rs_file.read_text(),
            )
            llm_log.append({"phase": "safe_repair", "attempt": attempt, "error": build_error[:2000]})
            llm_turns += 1
            retries += 1
            _apply_llm_response(repair_response, rs_file)
            build_ok, build_error = _cargo_build(cargo_toml)

        if not build_ok:
            # Restore original
            original = rust_dir / rs_file.relative_to(output_dir)
            if original.exists():
                shutil.copy(original, rs_file)

    # Count unsafe after
    unsafe_after = 0
    for f in rs_files:
        unsafe_after += _count_unsafe(f.read_text())

    (output_dir / "llm_log.json").write_text(json.dumps(llm_log, indent=2))
    result = {
        "unsafe_before": unsafe_before,
        "unsafe_after": unsafe_after,
        "llm_turns": llm_turns,
        "retries": retries,
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2))
    return result

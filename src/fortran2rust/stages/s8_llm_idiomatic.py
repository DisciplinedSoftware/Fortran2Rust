from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..llm.base import LLMClient

IDIOMATIC_SYSTEM_PROMPT = (
    "You are a Rust expert. Rewrite this Rust code to be idiomatic: use iterators instead of raw loops, "
    "use Vec/slice indexing instead of raw pointers, improve naming conventions (snake_case), use proper "
    "error handling with Result/Option instead of assertions, and add appropriate documentation comments. "
    "Preserve exact numerical behavior. "
    "Return ONLY the complete corrected file, no explanations, no markdown fences."
)


def _cargo_build(cargo_toml: Path) -> tuple[bool, str]:
    result = subprocess.run(
        ["cargo", "build", "--manifest-path", str(cargo_toml)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    return result.returncode == 0, result.stderr


def _apply_llm_response(response: str, target_file: Path) -> None:
    content = re.sub(r"```[a-z]*\n?", "", response).strip()
    target_file.write_text(content + "\n")


def make_idiomatic(
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

    rs_files = [f for f in output_dir.rglob("*.rs") if "bench" not in f.name and "test" not in f.name]

    for rs_file in rs_files:
        content = rs_file.read_text()
        if status_fn:
            status_fn(f"LLM: making {rs_file.name} idiomatic…")
        response = llm.complete(IDIOMATIC_SYSTEM_PROMPT, content)
        llm_log.append({"phase": "idiomatic", "file": rs_file.name})
        llm_turns += 1
        _apply_llm_response(response, rs_file)

        build_ok, build_error = _cargo_build(cargo_toml)
        for attempt in range(max_retries):
            if build_ok:
                break
            if status_fn:
                status_fn(f"Verifying idiomatic Rust builds… (attempt {attempt+1}/{max_retries})")
            repair_response = llm.repair(
                context="Fix compilation error after making Rust code idiomatic.",
                error=build_error,
                code=rs_file.read_text(),
            )
            llm_log.append({"phase": "idiomatic_repair", "attempt": attempt, "error": build_error[:2000]})
            llm_turns += 1
            retries += 1
            _apply_llm_response(repair_response, rs_file)
            build_ok, build_error = _cargo_build(cargo_toml)

        if not build_ok:
            # Restore from previous stage
            orig = rust_dir / rs_file.relative_to(output_dir)
            if orig.exists():
                shutil.copy(orig, rs_file)

    (output_dir / "llm_log.json").write_text(json.dumps(llm_log, indent=2))
    result = {
        "files_processed": len(rs_files),
        "llm_turns": llm_turns,
        "retries": retries,
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2))
    return result

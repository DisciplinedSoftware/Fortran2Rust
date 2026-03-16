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
from ._bench import _fix_duplicate_no_mangle, _fix_stable_rust_features, print_bench_summary, run_rust_benchmarks
from ._log import make_stage_logger

_console = Console(stderr=True)

IDIOMATIC_SYSTEM_PROMPT = (
    "You are a Rust expert. Rewrite this Rust code to be idiomatic: use iterators instead of raw loops, "
    "use Vec/slice indexing instead of raw pointers, improve naming conventions (snake_case), use proper "
    "error handling with Result/Option instead of assertions, and add appropriate documentation comments. "
    "Preserve exact numerical behavior. "
    "Return ONLY the complete corrected file, no explanations, no markdown fences."
)


def _first_error_line(error: str) -> str:
    for line in error.splitlines():
        if "error" in line.lower() and line.strip():
            return line.strip()[:120]
    return error.strip()[:120]


def _cargo_build(cargo_toml: Path) -> tuple[bool, str]:
    result = subprocess.run(
        ["cargo", "build", "--lib", "--manifest-path", str(cargo_toml)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    return result.returncode == 0, result.stdout + result.stderr


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
    log = make_stage_logger(output_dir)
    log.info(f"make_idiomatic: rust_dir={rust_dir}, max_retries={max_retries}")

    if output_dir != rust_dir:
        shutil.copytree(rust_dir, output_dir, dirs_exist_ok=True)

    cargo_toml = output_dir / "Cargo.toml"
    cargo_build_log = output_dir / "cargo_build.log"
    llm_log: list[dict] = []
    llm_turns = 0
    retries = 0

    rs_files = [f for f in output_dir.rglob("*.rs") if "bench" not in f.name and "test" not in f.name and f.name != "lib.rs"]
    log.info(f"Processing {len(rs_files)} Rust files")

    for rs_file in rs_files:
        content = rs_file.read_text()
        if status_fn:
            status_fn(f"LLM: making {rs_file.name} idiomatic…")
        log.info(f"LLM idiomatic rewrite of {rs_file.name}")
        response = llm.complete(IDIOMATIC_SYSTEM_PROMPT, content)
        llm_log.append({"phase": "idiomatic", "file": rs_file.name})
        llm_turns += 1
        _apply_llm_response(response, rs_file)
        _fix_stable_rust_features(rs_file)  # LLM sometimes re-adds stabilised feature flags

        build_ok, build_output = _cargo_build(cargo_toml)
        with open(cargo_build_log, "a") as fh:
            fh.write(
                f"=== {rs_file.name} (after idiomatic rewrite) ===\n"
                f"{build_output}\n=== EXIT: {'OK' if build_ok else 'FAIL'} ===\n\n"
            )
        for attempt in range(max_retries):
            if build_ok:
                log.info(f"  cargo build OK after idiomatic rewrite of {rs_file.name}")
                break
            # Try deterministic fixes before spending an LLM turn
            if _fix_duplicate_no_mangle(rs_file, build_output):
                log.info(f"  Fixed duplicate #[no_mangle] in {rs_file.name} without LLM")
                build_ok, build_output = _cargo_build(cargo_toml)
                with open(cargo_build_log, "a") as fh:
                    fh.write(
                        f"=== {rs_file.name} dedup fix ===\n"
                        f"{build_output}\n=== EXIT: {'OK' if build_ok else 'FAIL'} ===\n\n"
                    )
                if build_ok:
                    break
            _console.print(
                f"  [yellow]⚠ Idiomatic Rust build failed[/yellow] in [bold]{rs_file.name}[/bold]: "
                f"[dim]{_first_error_line(build_output)}[/dim]"
            )
            log.warning(f"  build failed after idiomatic rewrite of {rs_file.name}, attempt {attempt+1}/{max_retries}")
            if status_fn:
                status_fn(f"LLM: fixing idiomatic Rust build (attempt {attempt+1}/{max_retries})…")
            repair_response = llm.repair(
                context="Fix compilation error after making Rust code idiomatic.",
                error=build_output,
                code=rs_file.read_text(),
            )
            llm_log.append({"phase": "idiomatic_repair", "attempt": attempt, "error": build_output})
            llm_turns += 1
            retries += 1
            _apply_llm_response(repair_response, rs_file)
            _fix_stable_rust_features(rs_file)
            build_ok, build_output = _cargo_build(cargo_toml)
            with open(cargo_build_log, "a") as fh:
                fh.write(
                    f"=== {rs_file.name} repair attempt {attempt+1} ===\n"
                    f"{build_output}\n=== EXIT: {'OK' if build_ok else 'FAIL'} ===\n\n"
                )

        if not build_ok:
            log.warning(f"  reverting {rs_file.name} — could not fix after {max_retries} attempts")
            # Restore from previous stage
            orig = rust_dir / rs_file.relative_to(output_dir)
            if orig.exists():
                shutil.copy(orig, rs_file)

    (output_dir / "llm_log.json").write_text(json.dumps(llm_log, indent=2))
    (output_dir / "llm_conversations.json").write_text(
        json.dumps(llm.pop_conversation_log(), indent=2)
    )
    result = {
        "files_processed": len(rs_files),
        "llm_turns": llm_turns,
        "retries": retries,
    }

    # ── Benchmarks: build bins + run against Fortran baseline ─────────────────
    if status_fn:
        status_fn("Running Rust benchmarks…")
    log.info("Running Rust benchmarks")
    bench_results = run_rust_benchmarks(output_dir, baseline_dir, cargo_toml, log, status_fn)
    print_bench_summary(bench_results, {})
    result["bench_results"] = bench_results

    (output_dir / "result.json").write_text(json.dumps(result, indent=2))
    log.info("Stage complete")
    return result

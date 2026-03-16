from __future__ import annotations

import json
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console

if TYPE_CHECKING:
    from ..llm.base import LLMClient

from ..exceptions import CompilationError, MaxRetriesExceededError
from ._bench import _fix_bench_extern_types, _fix_stable_rust_features, _get_failing_rust_files, print_bench_summary, run_rust_benchmarks
from ._log import make_stage_logger

_console = Console(stderr=True)


def _first_error_line(error: str) -> str:
    for line in error.splitlines():
        if "error" in line.lower() and line.strip():
            return line.strip()[:120]
    return error.strip()[:120]


def _apply_llm_response(response: str, output_dir: Path) -> None:
    parts = re.split(r"^---\s+(\S+\.(?:rs|toml))\s+---$", response, flags=re.MULTILINE)
    if len(parts) >= 3:
        it = iter(parts[1:])
        for filename, content in zip(it, it):
            target = output_dir / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(_strip_fences(content) + "\n")
    else:
        content = _strip_fences(response)
        rs_files = sorted(output_dir.rglob("*.rs"))
        target = rs_files[0] if rs_files else output_dir / "src" / "lib.rs"
        target.write_text(content + "\n")


def _strip_fences(content: str) -> str:
    """Strip markdown code fences from LLM output (handles ``` ```rust, ```Rust, etc.)."""
    content = re.sub(r"^```[a-zA-Z0-9]*\s*\n?", "", content.strip())
    content = re.sub(r"\n?```\s*$", "", content)
    return content.strip()


def _cargo_build(cargo_toml: Path) -> tuple[bool, str]:
    result = subprocess.run(
        ["cargo", "build", "--release", "--lib", "--manifest-path", str(cargo_toml)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    return result.returncode == 0, result.stdout + result.stderr


def fix_rust_code(
    rust_dir: Path,
    output_dir: Path,
    llm: "LLMClient",
    max_retries: int,
    baseline_dir: Path,
    status_fn=None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    log = make_stage_logger(output_dir)
    log.info(f"fix_rust_code: rust_dir={rust_dir}, max_retries={max_retries}")

    if output_dir != rust_dir:
        shutil.copytree(rust_dir, output_dir, dirs_exist_ok=True)

    # Patch c2rust extern types before the first build to avoid a wasted LLM retry.
    # Apply to all .rs files — not just bench_*.rs — because lib files such as xerbla.rs
    # also contain `#![feature(extern_types)]` and opaque extern type declarations.
    src_dir = output_dir / "src"
    for rs in sorted(src_dir.glob("*.rs")):
        _fix_bench_extern_types(rs)
        _fix_stable_rust_features(rs)
        log.info(f"Pre-patched extern types / stable features in {rs.name}")

    cargo_toml = output_dir / "Cargo.toml"
    cargo_build_log = output_dir / "cargo_build.log"
    llm_log: list[dict] = []
    llm_turns = 0
    retries = 0

    # ── Compilation repair loop (LLM gated) ──────────────────────────────────
    if status_fn:
        status_fn("cargo build…")
    build_ok, build_output = _cargo_build(cargo_toml)
    build_error = build_output
    with open(cargo_build_log, "a") as fh:
        fh.write(f"=== INITIAL BUILD ===\n{build_output}\n=== EXIT: {'OK' if build_ok else 'FAIL'} ===\n\n")
    log.info(f"Initial cargo build: {'OK' if build_ok else 'FAILED'}")
    if not build_ok:
        log.warning(build_output[:2000])

    for attempt in range(max_retries):
        if build_ok:
            break
        # Exclude pre-generated bench_*.rs files from LLM repair — the LLM does not
        # understand their structure and tends to corrupt them with explanation text.
        failing = [
            f for f in _get_failing_rust_files(build_output, output_dir)
            if not f.name.startswith("bench_")
        ]
        if not failing:
            # All errors in bench files — re-patch them and retry without LLM
            for rs in sorted(src_dir.glob("bench_*.rs")):
                _fix_bench_extern_types(rs)
            build_ok, build_output = _cargo_build(cargo_toml)
            build_error = build_output
            with open(cargo_build_log, "a") as fh:
                fh.write(f"=== BENCH REPATCH ===\n{build_output}\n=== EXIT: {'OK' if build_ok else 'FAIL'} ===\n\n")
            continue
        _console.print(
            f"  [yellow]⚠ Rust build failed[/yellow] "
            f"({len(failing)} file(s)): "
            f"[dim]{_first_error_line(build_output)}[/dim]"
        )
        if status_fn:
            status_fn(
                f"LLM: fixing {len(failing)} failing Rust file(s) "
                f"(attempt {attempt+1}/{max_retries})…"
            )
        log.info(f"LLM repair attempt {attempt+1}/{max_retries}: {[f.name for f in failing]}")

        def _repair(rs_file: Path) -> tuple:
            code = rs_file.read_text()
            response = llm.repair(
                context=(
                    "Fix this Rust file that was transpiled from C by c2rust. "
                    "Fix all compilation errors shown."
                ),
                error=build_output,
                code=code,
                attempt=attempt,
            )
            return rs_file, response

        with ThreadPoolExecutor(max_workers=len(failing)) as executor:
            repairs = list(executor.map(_repair, failing))

        for rs_file, response in repairs:
            _apply_llm_response(response, output_dir)
            llm_log.append({"phase": "build", "attempt": attempt, "file": rs_file.name, "error": build_output})
            llm_turns += 1
            retries += 1

        build_ok, build_output = _cargo_build(cargo_toml)
        build_error = build_output
        with open(cargo_build_log, "a") as fh:
            fh.write(
                f"=== ATTEMPT {attempt+1} ===\n{build_output}\n"
                f"=== EXIT: {'OK' if build_ok else 'FAIL'} ===\n\n"
            )
        if build_ok:
            log.info(f"cargo build OK after attempt {attempt+1}")
        else:
            log.warning(f"cargo build still failing after attempt {attempt+1}")

    if not build_ok:
        exc = CompilationError("Rust", build_error)
        raise MaxRetriesExceededError("Stage 6 (fix Rust)", exc)

    # ── Benchmarks: build bins + run against Fortran baseline ─────────────────
    if status_fn:
        status_fn("Running Rust benchmarks…")
    log.info("Running Rust benchmarks")
    bench_results = run_rust_benchmarks(output_dir, baseline_dir, cargo_toml, log, status_fn)
    print_bench_summary(bench_results, {})

    (output_dir / "llm_log.json").write_text(json.dumps(llm_log, indent=2))
    (output_dir / "llm_conversations.json").write_text(
        json.dumps(llm.pop_conversation_log(), indent=2)
    )
    result = {
        "llm_turns": llm_turns,
        "retries": retries,
        "bench_results": bench_results,
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2))
    log.info("Stage complete")
    return result


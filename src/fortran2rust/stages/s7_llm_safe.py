from __future__ import annotations

import json
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from rich.console import Console

if TYPE_CHECKING:
    from ..llm.base import LLMClient

from ..exceptions import CompilationError, MaxRetriesExceededError
from ._bench import (
    _fix_bench_extern_types,
    _fix_duplicate_no_mangle,
    _fix_stable_rust_features,
    _get_failing_rust_files,
    print_bench_summary,
    run_rust_benchmarks,
)
from ._llm_cleanup import compact_rust_for_llm, filter_errors_for_file, restore_rust_after_llm, strip_markdown_fences
from ._log import make_stage_logger

_console = Console(stderr=True)

SAFE_SYSTEM_PROMPT = (
    "You are a Rust expert. Rewrite this code to eliminate all unsafe blocks while preserving "
    "exact numerical behavior. Return ONLY the complete corrected file, no explanations, no markdown fences."
)


def _first_error_line(error: str) -> str:
    for line in error.splitlines():
        if "error" in line.lower() and line.strip():
            return line.strip()[:120]
    return error.strip()[:120]


def _cargo_build(cargo_toml: Path) -> tuple[bool, str]:
    result = subprocess.run(
        ["cargo", "build", "--release", "--lib", "--manifest-path", str(cargo_toml)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    return result.returncode == 0, result.stdout + result.stderr


def _count_unsafe(content: str) -> int:
    return len(re.findall(r"\bunsafe\b", content))


def _apply_llm_response(response: str, target_file: Path) -> None:
    content = strip_markdown_fences(response)
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
    log = make_stage_logger(output_dir)
    log.info(f"make_safe: rust_dir={rust_dir}, max_retries={max_retries}")

    if output_dir != rust_dir:
        shutil.copytree(rust_dir, output_dir, dirs_exist_ok=True)

    # Patch c2rust extern types before the first build to avoid a wasted LLM retry.
    # Apply to all .rs files — not just bench_*.rs.
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

    rs_files = [
        f for f in output_dir.rglob("*.rs")
        if "bench" not in f.name and "test" not in f.name and f.name != "lib.rs"
    ]

    # Count unsafe before
    unsafe_before = sum(_count_unsafe(f.read_text()) for f in rs_files)
    log.info(f"unsafe blocks before: {unsafe_before} across {len(rs_files)} files")

    files_to_process = [(f, _count_unsafe(f.read_text())) for f in rs_files]
    files_to_process = [(f, n) for f, n in files_to_process if n > 0]

    if files_to_process:
        if status_fn:
            status_fn(
                f"LLM: removing unsafe blocks from {len(files_to_process)} file(s) (parallel)…"
            )

        # ── Phase 1: parallel initial unsafe removal ──────────────────────────
        def _rewrite(args: tuple) -> tuple:
            rs_file, n = args
            log.info(f"LLM removing {n} unsafe block(s) from {rs_file.name}")
            original = rs_file.read_text()
            compact_code, preserved_prefix = compact_rust_for_llm(original)
            prompt = (
                SAFE_SYSTEM_PROMPT.replace(
                    "Return ONLY the complete corrected file, no explanations, no markdown fences.",
                    "Leading comments were removed before sending to reduce token usage. Return ONLY the complete corrected file, no explanations, no markdown fences.",
                )
            )
            return rs_file, n, llm.complete(prompt, compact_code), preserved_prefix

        with ThreadPoolExecutor(max_workers=len(files_to_process)) as executor:
            rewrites = list(executor.map(_rewrite, files_to_process))

        for rs_file, n, response, preserved_prefix in rewrites:
            _apply_llm_response(response, rs_file)
            rs_file.write_text(restore_rust_after_llm(rs_file.read_text(), preserved_prefix) + "\n")
            _fix_stable_rust_features(rs_file)
            llm_log.append({"phase": "safe", "file": str(rs_file.name), "unsafe_count": n})
            llm_turns += 1

        # ── Phase 2: build once after all rewrites ────────────────────────────
        build_ok, build_output = _cargo_build(cargo_toml)
        with open(cargo_build_log, "a") as fh:
            fh.write(
                f"=== after parallel unsafe removal ===\n"
                f"{build_output}\n=== EXIT: {'OK' if build_ok else 'FAIL'} ===\n\n"
            )

        # ── Phase 3: repair loop ──────────────────────────────────────────────
        for attempt in range(max_retries):
            if build_ok:
                break

            failing = _get_failing_rust_files(build_output, output_dir)
            _console.print(
                f"  [yellow]⚠ Safe Rust build failed[/yellow] "
                f"({len(failing)} file(s)): "
                f"[dim]{_first_error_line(build_output)}[/dim]"
            )

            # Deterministic dedup fix first (no LLM cost)
            dedup_fixed = any(_fix_duplicate_no_mangle(f, build_output) for f in failing)
            if dedup_fixed:
                build_ok, build_output = _cargo_build(cargo_toml)
                with open(cargo_build_log, "a") as fh:
                    fh.write(
                        f"=== dedup fix (attempt {attempt+1}) ===\n"
                        f"{build_output}\n=== EXIT: {'OK' if build_ok else 'FAIL'} ===\n\n"
                    )
                if build_ok:
                    break
                failing = _get_failing_rust_files(build_output, output_dir)

            if status_fn:
                status_fn(
                    f"LLM: fixing safe Rust build — {len(failing)} file(s) "
                    f"(attempt {attempt+1}/{max_retries})…"
                )
            log.warning(
                f"build failed after unsafe removal, attempt {attempt+1}/{max_retries}, "
                f"failing: {[f.name for f in failing]}"
            )

            def _repair(rs_file: Path) -> tuple:
                original = rs_file.read_text()
                compact_code, preserved_prefix = compact_rust_for_llm(original)
                return rs_file, llm.repair(
                    context=(
                        "Fix compilation error after removing unsafe blocks in Rust code. "
                        "Leading comments were removed before sending to reduce token usage."
                    ),
                    error=filter_errors_for_file(build_output, rs_file.name),
                    code=compact_code,
                    attempt=attempt,
                ), preserved_prefix

            with ThreadPoolExecutor(max_workers=len(failing)) as executor:
                repairs = list(executor.map(_repair, failing))

            for rs_file, response, preserved_prefix in repairs:
                _apply_llm_response(response, rs_file)
                rs_file.write_text(restore_rust_after_llm(rs_file.read_text(), preserved_prefix) + "\n")
                _fix_stable_rust_features(rs_file)
                llm_log.append({"phase": "safe_repair", "attempt": attempt, "error": build_output})
                llm_turns += 1
                retries += 1

            build_ok, build_output = _cargo_build(cargo_toml)
            with open(cargo_build_log, "a") as fh:
                fh.write(
                    f"=== repair attempt {attempt+1} ===\n"
                    f"{build_output}\n=== EXIT: {'OK' if build_ok else 'FAIL'} ===\n\n"
                )

        if not build_ok:
            log.warning("Could not fix build after all retries — reverting all rewritten files")
            for rs_file, _ in files_to_process:
                original = rust_dir / rs_file.relative_to(output_dir)
                if original.exists():
                    shutil.copy(original, rs_file)

    # Count unsafe after
    unsafe_after = sum(_count_unsafe(f.read_text()) for f in rs_files)
    log.info(f"unsafe blocks after: {unsafe_after} (removed {unsafe_before - unsafe_after})")

    (output_dir / "llm_log.json").write_text(json.dumps(llm_log, indent=2))
    (output_dir / "llm_conversations.json").write_text(
        json.dumps(llm.pop_conversation_log(), indent=2)
    )
    result = {
        "unsafe_after": unsafe_after,
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

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
from ._bench import (
    _fix_duplicate_no_mangle,
    _fix_stable_rust_features,
    _get_failing_rust_files,
    print_bench_summary,
    run_rust_benchmarks,
)
from ._llm_cleanup import compact_rust_for_llm, restore_rust_after_llm, strip_markdown_fences
from ._log import make_stage_logger

_console = Console(stderr=True)

_MAX_BENCH_SLOWDOWN_RATIO = 1.5
_MIN_REGRESSION_MS = 5.0

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
        ["cargo", "build", "--release", "--lib", "--manifest-path", str(cargo_toml)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    return result.returncode == 0, result.stdout + result.stderr


def _load_previous_bench_results(rust_dir: Path) -> dict[str, dict]:
    result_path = rust_dir / "result.json"
    if not result_path.exists():
        return {}
    try:
        return json.loads(result_path.read_text()).get("bench_results", {})
    except (OSError, json.JSONDecodeError, AttributeError):
        return {}


def _find_bench_regressions(
    previous_results: dict[str, dict],
    current_results: dict[str, dict],
) -> list[dict[str, float | str]]:
    regressions: list[dict[str, float | str]] = []
    for fn_name, current in current_results.items():
        previous = previous_results.get(fn_name)
        if not previous:
            continue
        prev_time = previous.get("time_ms")
        curr_time = current.get("time_ms")
        if not previous.get("run_ok") or not current.get("run_ok"):
            continue
        if prev_time is None or curr_time is None or prev_time <= 0:
            continue
        ratio = curr_time / prev_time
        delta_ms = curr_time - prev_time
        if ratio >= _MAX_BENCH_SLOWDOWN_RATIO and delta_ms >= _MIN_REGRESSION_MS:
            regressions.append({
                "function": fn_name,
                "previous_ms": float(prev_time),
                "current_ms": float(curr_time),
                "ratio": float(ratio),
            })
    return regressions


def _apply_llm_response(response: str, target_file: Path) -> None:
    content = strip_markdown_fences(response)
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
    reverted_for_performance = False
    previous_bench_results = _load_previous_bench_results(rust_dir)

    rs_files = [
        f for f in output_dir.rglob("*.rs")
        if "bench" not in f.name and "test" not in f.name and f.name != "lib.rs"
    ]
    log.info(f"Processing {len(rs_files)} Rust files")

    if rs_files:
        if status_fn:
            status_fn(f"LLM: rewriting {len(rs_files)} file(s) to idiomatic Rust (parallel)…")

        # ── Phase 1: parallel idiomatic rewrite ───────────────────────────────
        def _rewrite(rs_file: Path) -> tuple:
            log.info(f"LLM idiomatic rewrite of {rs_file.name}")
            original = rs_file.read_text()
            compact_code, preserved_prefix = compact_rust_for_llm(original)
            prompt = IDIOMATIC_SYSTEM_PROMPT.replace(
                "Return ONLY the complete corrected file, no explanations, no markdown fences.",
                "Leading comments were removed before sending to reduce token usage. Return ONLY the complete corrected file, no explanations, no markdown fences.",
            )
            return rs_file, llm.complete(prompt, compact_code), preserved_prefix

        with ThreadPoolExecutor(max_workers=len(rs_files)) as executor:
            rewrites = list(executor.map(_rewrite, rs_files))

        for rs_file, response, preserved_prefix in rewrites:
            _apply_llm_response(response, rs_file)
            rs_file.write_text(restore_rust_after_llm(rs_file.read_text(), preserved_prefix) + "\n")
            _fix_stable_rust_features(rs_file)
            llm_log.append({"phase": "idiomatic", "file": rs_file.name})
            llm_turns += 1

        # ── Phase 2: build once after all rewrites ────────────────────────────
        build_ok, build_output = _cargo_build(cargo_toml)
        with open(cargo_build_log, "a") as fh:
            fh.write(
                f"=== after parallel idiomatic rewrite ===\n"
                f"{build_output}\n=== EXIT: {'OK' if build_ok else 'FAIL'} ===\n\n"
            )

        # ── Phase 3: repair loop ──────────────────────────────────────────────
        for attempt in range(max_retries):
            if build_ok:
                break

            failing = _get_failing_rust_files(build_output, output_dir)
            _console.print(
                f"  [yellow]⚠ Idiomatic Rust build failed[/yellow] "
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
                    f"LLM: fixing idiomatic Rust build — {len(failing)} file(s) "
                    f"(attempt {attempt+1}/{max_retries})…"
                )
            log.warning(
                f"build failed after idiomatic rewrite, attempt {attempt+1}/{max_retries}, "
                f"failing: {[f.name for f in failing]}"
            )

            def _repair(rs_file: Path) -> tuple:
                original = rs_file.read_text()
                compact_code, preserved_prefix = compact_rust_for_llm(original)
                return rs_file, llm.repair(
                    context=(
                        "Fix compilation error after making Rust code idiomatic. "
                        "Leading comments were removed before sending to reduce token usage."
                    ),
                    error=build_output,
                    code=compact_code,
                    attempt=attempt,
                ), preserved_prefix

            with ThreadPoolExecutor(max_workers=len(failing)) as executor:
                repairs = list(executor.map(_repair, failing))

            for rs_file, response, preserved_prefix in repairs:
                _apply_llm_response(response, rs_file)
                rs_file.write_text(restore_rust_after_llm(rs_file.read_text(), preserved_prefix) + "\n")
                _fix_stable_rust_features(rs_file)
                llm_log.append({"phase": "idiomatic_repair", "attempt": attempt, "error": build_output})
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
            for rs_file in rs_files:
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
    regressions = _find_bench_regressions(previous_bench_results, bench_results)
    if regressions:
        reverted_for_performance = True
        log.warning(
            "Idiomatic rewrite caused benchmark regressions; reverting rewritten files: "
            f"{regressions}"
        )
        _console.print(
            "  [yellow]⚠ Idiomatic rewrite regressed benchmark performance[/yellow]: "
            + ", ".join(
                f"{r['function']} {r['current_ms']:.1f}ms vs {r['previous_ms']:.1f}ms ({r['ratio']:.2f}x)"
                for r in regressions
            )
        )
        if status_fn:
            status_fn("Reverting idiomatic rewrites after benchmark regression…")
        for rs_file in rs_files:
            orig = rust_dir / rs_file.relative_to(output_dir)
            if orig.exists():
                shutil.copy(orig, rs_file)
        build_ok, build_output = _cargo_build(cargo_toml)
        with open(cargo_build_log, "a") as fh:
            fh.write(
                "=== reverted after benchmark regression ===\n"
                f"{build_output}\n=== EXIT: {'OK' if build_ok else 'FAIL'} ===\n\n"
            )
        if build_ok:
            bench_results = run_rust_benchmarks(output_dir, baseline_dir, cargo_toml, log, status_fn)
        else:
            log.warning("Build failed after reverting idiomatic rewrites")
    print_bench_summary(bench_results, {})
    result["bench_results"] = bench_results
    result["reverted_for_performance"] = reverted_for_performance
    if regressions:
        result["performance_regressions"] = regressions

    (output_dir / "result.json").write_text(json.dumps(result, indent=2))
    log.info("Stage complete")
    return result

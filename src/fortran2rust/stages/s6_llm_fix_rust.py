from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from rich.console import Console

if TYPE_CHECKING:
    from ..llm.base import LLMClient

from ..exceptions import CompilationError, MaxRetriesExceededError
from ._log import make_stage_logger

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
    return result.returncode == 0, result.stdout + result.stderr


def _run_bench(cargo_toml: Path, baseline_dir: Path) -> dict:
    """Run the compiled Rust benchmark, capture timing and numerical output. No LLM gating."""
    result: dict = {"time_ms": None, "max_abs_diff": None, "run_ok": False, "run_error": ""}
    try:
        run = subprocess.run(
            ["cargo", "run", "--manifest-path", str(cargo_toml)],
            capture_output=True, text=True, cwd=str(baseline_dir), timeout=300,
        )
        result["run_ok"] = run.returncode == 0
        result["run_error"] = run.stderr + run.stdout if run.returncode != 0 else ""
        match = re.search(r"(?:RUST|C)_TIME_MS=\s*([\d.]+)", run.stdout)
        if match:
            result["time_ms"] = float(match.group(1))

        # Compare output against Fortran baseline
        for rust_bin in baseline_dir.glob("bench_*_output.bin"):
            fn_name = re.sub(r"^bench_|_output\.bin$", "", rust_bin.name)
            fortran_bin = baseline_dir / f"bench_{fn_name}_output.bin"
            if rust_bin != fortran_bin and fortran_bin.exists():
                r_data = np.fromfile(str(rust_bin), dtype=np.float64)
                f_data = np.fromfile(str(fortran_bin), dtype=np.float64)
                if r_data.shape == f_data.shape:
                    result["max_abs_diff"] = float(np.max(np.abs(r_data - f_data)))
    except Exception as e:
        result["run_error"] = str(e)
    return result


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
        _console.print(
            f"  [yellow]⚠ Rust build failed[/yellow]: "
            f"[dim]{_first_error_line(build_output)}[/dim]"
        )
        if status_fn:
            status_fn(f"LLM: fixing Rust compilation (attempt {attempt+1}/{max_retries})…")
        log.info(f"LLM repair attempt {attempt+1}/{max_retries}")
        code = _read_rust_files(output_dir)
        response = llm.repair(
            context="Fix this Rust code that was transpiled from C by c2rust. Fix all compilation errors.",
            error=build_output,
            code=code,
        )
        llm_log.append({"phase": "build", "attempt": attempt, "error": build_output})
        llm_turns += 1
        retries += 1
        _apply_llm_response(response, output_dir)
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

    # ── Timing + numerical accuracy capture (report only, no LLM gating) ─────
    if status_fn:
        status_fn("Running Rust benchmark (for report)…")
    log.info("Running Rust benchmark")
    bench_data = _run_bench(cargo_toml, baseline_dir)
    if bench_data["max_abs_diff"] is not None:
        _console.print(
            f"  [dim]↳ numerical diff vs Fortran: max abs = {bench_data['max_abs_diff']:.3e}[/dim]"
        )
        log.info(f"  numerical diff vs Fortran: max abs = {bench_data['max_abs_diff']:.3e}")

    (output_dir / "llm_log.json").write_text(json.dumps(llm_log, indent=2))
    result = {
        "build_ok": build_ok,
        "llm_turns": llm_turns,
        "retries": retries,
        **bench_data,
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2))
    log.info("Stage complete")
    return result


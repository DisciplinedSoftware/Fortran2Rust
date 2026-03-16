from __future__ import annotations

import logging
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
from rich.console import Console

_console = Console(stderr=True)

# Must match the package name in the Cargo.toml template (s5_c2rust.py)
_CRATE_NAME = "fortran2rust_output"


def run_rust_benchmarks(
    output_dir: Path,
    baseline_dir: Path,
    cargo_toml: Path,
    log: logging.Logger,
    status_fn=None,
) -> dict[str, dict]:
    """
    Build Rust bench binaries from transpiled bench_*.rs modules and run them
    against the Fortran baseline. Purely informational — never raises.

    For each src/bench_{fn}.rs in output_dir:
      - Generates a thin safe wrapper  src/main_bench_{fn}.rs
      - Adds a [[bin]] entry to Cargo.toml
      - Builds with `cargo build --bins`
      - Copies dataset_*.bin files from baseline_dir into output_dir
      - Runs the binary from output_dir, parses C_TIME_MS / RUST_TIME_MS
      - Compares bench_{fn}_output.bin against baseline_dir/bench_{fn}_output.bin
      - Saves rust_bench_{fn}.log

    Returns {fn_name: {run_ok, time_ms, max_abs_diff, run_error}}.
    """
    src_dir = output_dir / "src"
    bench_rs_files = sorted(src_dir.glob("bench_*.rs")) if src_dir.is_dir() else []
    if not bench_rs_files:
        return {}

    # Append [[bin]] entries + wrapper files for each bench module
    cargo_content = cargo_toml.read_text()
    for bench_rs in bench_rs_files:
        stem = bench_rs.stem  # e.g. bench_dgemm
        if f'name = "{stem}"' in cargo_content:
            continue
        wrapper = src_dir / f"main_{stem}.rs"
        if not wrapper.exists():
            # Thin safe wrapper — calls the (possibly unsafe) transpiled main()
            wrapper.write_text(
                f"fn main() {{\n"
                f"    #[allow(unsafe_code)]\n"
                f"    unsafe {{ {_CRATE_NAME}::{stem}::main(); }}\n"
                f"}}\n"
            )
        cargo_content += (
            f'\n[[bin]]\nname = "{stem}"\n'
            f'path = "src/main_{stem}.rs"\n'
        )
    cargo_toml.write_text(cargo_content)

    # Build all bench binaries
    if status_fn:
        status_fn("Building Rust bench binaries…")
    log.info("cargo build --bins")
    br = subprocess.run(
        ["cargo", "build", "--bins", "--manifest-path", str(cargo_toml)],
        capture_output=True, text=True, timeout=300,
    )
    if br.returncode != 0:
        log.warning(f"cargo build --bins failed:\n{br.stderr[:1000]}")
        return {}

    # Copy dataset files so the bench binary can read them from output_dir
    for ds in sorted(baseline_dir.glob("dataset_*.bin")):
        dest = output_dir / ds.name
        if not dest.exists():
            shutil.copy(ds, dest)

    results: dict[str, dict] = {}
    for bench_rs in bench_rs_files:
        stem = bench_rs.stem  # e.g. bench_dgemm
        fn_name = re.sub(r"^bench_", "", stem)  # e.g. dgemm

        fortran_bin = baseline_dir / f"bench_{fn_name}_output.bin"
        if not fortran_bin.exists():
            log.info(f"  No Fortran baseline for {fn_name}, skipping")
            continue

        binary = output_dir / "target" / "debug" / stem
        if not binary.exists():
            log.warning(f"  Bench binary not found: {binary}")
            continue

        if status_fn:
            status_fn(f"Running Rust benchmark: {stem}…")
        log.info(f"Running Rust benchmark: {stem}")

        run = subprocess.run(
            [str(binary)], capture_output=True, text=True,
            cwd=str(output_dir), timeout=300,
        )
        (output_dir / f"rust_bench_{stem}.log").write_text(
            f"=== BINARY ===\n{binary}\n\n"
            f"=== STDOUT ===\n{run.stdout}\n"
            f"=== STDERR ===\n{run.stderr}\n"
            f"=== EXIT CODE: {run.returncode} ===\n"
        )

        entry: dict = {
            "run_ok": run.returncode == 0,
            "time_ms": None,
            "max_abs_diff": None,
            "run_error": "",
        }
        if run.returncode != 0:
            entry["run_error"] = run.stdout + run.stderr
            log.warning(f"  {stem} failed (exit {run.returncode}): {run.stderr[:300]}")
        else:
            m = re.search(r"(?:RUST|C)_TIME_MS=([\d.]+)", run.stdout)
            if m:
                entry["time_ms"] = float(m.group(1))

            rust_out = output_dir / f"bench_{fn_name}_output.bin"
            if rust_out.exists():
                r_data = np.fromfile(str(rust_out), dtype=np.float64)
                f_data = np.fromfile(str(fortran_bin), dtype=np.float64)
                if r_data.shape == f_data.shape:
                    entry["max_abs_diff"] = float(np.max(np.abs(r_data - f_data)))
                    log.info(
                        f"  {fn_name}: max_abs_diff={entry['max_abs_diff']:.3e}"
                        f", time_ms={entry['time_ms']}"
                    )
                else:
                    log.warning(f"  {fn_name}: shape mismatch {r_data.shape} vs {f_data.shape}")
            else:
                log.warning(f"  {fn_name}: output binary not found after run")

        results[fn_name] = entry

    return results


def print_bench_summary(bench_results: dict[str, dict], fortran_times: dict[str, float | None]) -> None:
    """Print a one-liner bench summary to the Rich console (stderr)."""
    if not bench_results:
        return
    parts = []
    for fn_name, br in bench_results.items():
        diff = br.get("max_abs_diff")
        diff_str = f"Δ={diff:.2e}" if diff is not None else "Δ=?"
        r_ms = br.get("time_ms")
        f_ms = fortran_times.get(fn_name)
        if r_ms is not None and f_ms:
            timing_str = f"Rust:{r_ms:.1f}ms, {r_ms/f_ms:.2f}x Fortran"
        elif r_ms is not None:
            timing_str = f"Rust:{r_ms:.1f}ms"
        else:
            timing_str = None
        detail = f"{diff_str}, {timing_str}" if timing_str else diff_str
        if br.get("run_ok"):
            parts.append(f"[green]{fn_name}[/green] ✓ ({detail})")
        else:
            parts.append(f"[red]{fn_name}[/red] ✗ (run failed)")
    _console.print(f"  Benchmarks: {' | '.join(parts)}")

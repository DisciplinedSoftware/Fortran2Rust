from __future__ import annotations

import logging
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
from rich.console import Console

from .s2_benchmarks import _blas_precision

_console = Console(stderr=True)

# Must match the package name in the Cargo.toml template (s5_c2rust.py)
_CRATE_NAME = "fortran2rust_output"


def _get_failing_rust_files(error: str, workspace_dir: Path) -> list[Path]:
    """Parse ``cargo build`` stderr to find source files that contain errors.

    Rust error messages contain lines like::

        error[E0412]: cannot find type ...
          --> src/dgemm.rs:42:27

    Paths are relative to the workspace root (*workspace_dir*).  Returns a
    deduplicated list of existing ``Path`` objects, or all ``.rs`` files in
    *workspace_dir* when no specific files can be identified.
    """
    found: list[Path] = []
    seen: set[str] = set()
    for match in re.finditer(r"-->\s+(\S+\.rs):", error):
        rel = match.group(1)
        if rel not in seen:
            seen.add(rel)
            candidate = (workspace_dir / rel).resolve()
            if candidate.exists():
                found.append(candidate)
    return found if found else list(workspace_dir.rglob("*.rs"))


# Features that c2rust emits but that have since been stabilised.
# Keeping them on a stable toolchain raises E0554.
_STABLE_FEATURES: frozenset[str] = frozenset({"raw_ref_op"})


def _fix_stable_rust_features(rs_file: Path) -> None:
    """
    Strip known-stabilised feature flags from a c2rust-generated .rs file.

    c2rust emits ``#![feature(raw_ref_op)]`` (stabilised in Rust 1.82) and
    potentially others.  Using ``#![feature(...)]`` on a stable toolchain
    raises E0554.  Entries in ``_STABLE_FEATURES`` are removed; the entire
    attribute is dropped when the list becomes empty.

    The transformation is idempotent.
    """
    text = rs_file.read_text()

    def _strip(m: re.Match) -> str:
        flags = [f.strip() for f in m.group(1).split(",")
                 if f.strip() and f.strip() not in _STABLE_FEATURES]
        return f'#![feature({", ".join(flags)})]' if flags else ""

    new_text = re.sub(r"#!\[feature\(([^)]*)\)\]", _strip, text)
    if new_text != text:
        rs_file.write_text(new_text)


def _fix_duplicate_no_mangle(rs_file: Path, build_output: str) -> bool:
    """
    Remove ``#[no_mangle]`` from functions in *rs_file* that cause duplicate-
    symbol linker errors because the real definition lives in another module.

    When an LLM rewrites a file it sometimes copies helper functions (e.g.
    ``lsame_``, ``xerbla_``) inline and marks them ``#[no_mangle]``.  That
    conflicts with the original ``#[no_mangle]`` definition in their own
    module (``lsame.rs``, ``xerbla.rs``).  Dropping ``#[no_mangle]`` from
    the duplicate turns it into a private Rust helper with a mangled name,
    eliminating the linker conflict while preserving the call semantics.

    Handles both linker error formats:
    - ``error: symbol `X` is already defined``
    - ``multiple definition of `X```

    Returns True if the file was modified.
    """
    symbols: set[str] = set()
    for pattern in (
        r"symbol [`']([^`'\s]+)[`'] is already defined",
        r"multiple definition of [`']([^`'\s]+)[`']",
    ):
        symbols.update(re.findall(pattern, build_output))
    if not symbols:
        return False

    lines = rs_file.read_text().splitlines(keepends=True)
    new_lines: list[str] = []
    changed = False
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped == "#[no_mangle]":
            # Look past any additional attribute lines to find the fn signature
            j = i + 1
            while j < len(lines) and lines[j].strip().startswith("#["):
                j += 1
            if j < len(lines) and any(
                re.search(r"\bfn\s+" + re.escape(sym) + r"\b", lines[j])
                for sym in symbols
            ):
                changed = True  # drop this #[no_mangle] line
                i += 1
                continue
        new_lines.append(lines[i])
        i += 1

    if changed:
        rs_file.write_text("".join(new_lines))
    return changed


def _fix_bench_extern_types(bench_rs: Path) -> None:
    """
    Patch a c2rust-transpiled bench_*.rs for stable-Rust compatibility.

    c2rust emits ``extern_types`` (an unstable feature) for opaque C types
    (``_IO_wide_data``, ``_IO_codecvt``, ``_IO_marker`` …).  This function
    replaces them with ``#[repr(C)] struct`` declarations so the file
    compiles on stable Rust, and removes ``extern_types`` from the feature
    list (leaving any remaining unstable features intact).

    The transformation is idempotent — if the file was already patched it
    returns immediately.
    """
    text = bench_rs.read_text()

    # Find all ``pub type NAME;`` lines inside extern "C" blocks
    extern_type_names = re.findall(
        r"^\s{4}pub type ([A-Za-z_]\w*);\s*$", text, re.MULTILINE
    )
    if not extern_type_names:
        return  # Already patched or no extern types present

    # Stable opaque-type replacements — sized, non-constructible, correct ABI
    stable_defs = "\n".join(
        f"#[repr(C)] pub struct {n} {{ "
        f"_priv: [u8; 0], "
        f"_marker: ::core::marker::PhantomData<(*mut u8, ::core::marker::PhantomPinned)> }}"
        for n in extern_type_names
    )

    # Remove the ``pub type NAME;`` lines from extern "C" blocks
    for name in extern_type_names:
        text = re.sub(
            rf"^\s+pub type {re.escape(name)};\s*\n", "", text, flags=re.MULTILINE
        )

    # Remove ``extern_types`` from any ``#![feature(...)]`` attribute,
    # dropping the whole attribute if it becomes empty.
    def _strip(m: re.Match) -> str:
        flags = [f.strip() for f in m.group(1).split(",")
                 if f.strip() and f.strip() != "extern_types"]
        return f'#![feature({", ".join(flags)})]' if flags else ""

    text = re.sub(r"#!\[feature\(([^)]*)\)\]", _strip, text)

    # Insert stable defs right after the last ``#![...]`` crate attribute
    last_end = 0
    for m in re.finditer(r"^#!\[.*\]\n", text, re.MULTILINE):
        last_end = m.end()
    text = text[:last_end] + stable_defs + "\n" + text[last_end:]

    bench_rs.write_text(text)


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
      - Removes the ``pub mod bench_{fn};`` declaration from lib.rs so the
        file can be compiled as a standalone binary (not a library module).
        This is necessary for the ``#![feature(...)]`` attributes inside the
        file to apply at the crate level.
      - Patches extern-type declarations to stable-Rust ``#[repr(C)]`` structs.
      - Adds a ``[[bin]]`` entry to Cargo.toml pointing directly at the file.
      - Builds with ``cargo build --bins``
      - Copies dataset_*.bin files from baseline_dir into output_dir
      - Runs the binary from output_dir, parses C_TIME_MS / RUST_TIME_MS
      - Compares bench_{fn}_output.bin against baseline_dir/bench_{fn}_output.bin
      - Saves rust_bench_{fn}.log

    Returns {fn_name: {run_ok, time_ms, max_abs_diff, run_error}}.
    """
    src_dir = output_dir / "src"

    # Copy pre-generated Rust bench files from baseline_dir (stage 2) to src_dir,
    # replacing any c2rust-generated counterparts.  The stage-2 files include
    # no-op stubs for f2c Fortran I/O symbols (s_wsfe, do_fio, …) that are
    # undefined in the rlib, which would otherwise cause binary link failures.
    if baseline_dir and baseline_dir.is_dir():
        src_dir.mkdir(parents=True, exist_ok=True)
        for rs in sorted(baseline_dir.glob("bench_*.rs")):
            dest = src_dir / rs.name
            shutil.copy(rs, dest)
            log.info(f"Using stage-2 pre-generated Rust bench: {rs.name}")

    bench_rs_files = sorted(src_dir.glob("bench_*.rs")) if src_dir.is_dir() else []
    if not bench_rs_files:
        return {}

    # Remove bench_* module declarations from lib.rs so the files can be used
    # directly as binary crate roots (c2rust already emits `pub fn main()`).
    lib_rs = src_dir / "lib.rs"
    if lib_rs.exists():
        lib_text = lib_rs.read_text()
        cleaned = re.sub(
            r"^[^\S\n]*pub mod bench_[^\n]*\n", "", lib_text, flags=re.MULTILINE
        )
        if cleaned != lib_text:
            lib_rs.write_text(cleaned)
            log.info("Removed pub mod bench_* declarations from lib.rs")

    # Add [[bin]] entries that point directly at each bench_*.rs file.
    # No thin-wrapper is needed — c2rust generates `pub fn main()` for them.
    cargo_content = cargo_toml.read_text()
    # Ensure rlib is in crate-type so [[bin]] targets can link against the lib.
    # Without rlib, extern "C" calls to #[no_mangle] functions are unresolved.
    if 'crate-type' in cargo_content and '"rlib"' not in cargo_content:
        cargo_content = re.sub(
            r'(crate-type\s*=\s*\[)',
            r'\1"rlib", ',
            cargo_content,
        )
        log.info("Added rlib to crate-type so bench binaries can link against the lib")
    for bench_rs in bench_rs_files:
        stem = bench_rs.stem  # e.g. bench_dgemm
        if f'name = "{stem}"' in cargo_content:
            continue
        _fix_bench_extern_types(bench_rs)
        log.info(f"Adding [[bin]] for {stem} (direct, no wrapper)")
        cargo_content += (
            f'\n[[bin]]\nname = "{stem}"\n'
            f'path = "src/{stem}.rs"\n'
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
        log.warning(f"cargo build --bins failed:\n{br.stderr[:3000]}")
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
            "max_rel_diff": None,
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
                dtype = np.dtype(_blas_precision(fn_name).numpy_dtype)
                r_data = np.fromfile(str(rust_out), dtype=dtype)
                f_data = np.fromfile(str(fortran_bin), dtype=dtype)
                if r_data.shape == f_data.shape:
                    abs_diff = np.abs(r_data - f_data)
                    entry["max_abs_diff"] = float(np.max(abs_diff))
                    entry["max_rel_diff"] = float(
                        np.max(abs_diff / np.maximum(np.abs(f_data), 1e-10))
                    )
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

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from ._log import make_stage_logger
from ._bench import _fix_stable_rust_features
from ..exceptions import ConversionError

CARGO_TOML_TEMPLATE = """\
[package]
name = "fortran2rust_output"
version = "0.1.0"
edition = "2021"

[lib]
name = "fortran2rust_output"
crate-type = ["rlib", "cdylib", "staticlib"]
"""


def ensure_c2rust(status_fn=None) -> Path:
    from rich.progress import Progress, SpinnerColumn, TextColumn

    path = shutil.which("c2rust")
    if path:
        return Path(path)

    if status_fn:
        status_fn("Installing c2rust via cargo (this may take several minutes)…")
        result = subprocess.run(
            ["cargo", "install", "c2rust"],
            capture_output=True,
            text=True,
            timeout=1800,
        )
    else:
        with Progress(SpinnerColumn(), TextColumn("[bold blue]Installing c2rust (this may take a while)..."), transient=True) as p:
            p.add_task("install", total=None)
            result = subprocess.run(
                ["cargo", "install", "c2rust"],
                capture_output=True,
                text=True,
                timeout=1800,
            )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to install c2rust:\n{result.stderr}")

    path = shutil.which("c2rust")
    if not path:
        raise RuntimeError("c2rust installed but not found in PATH")
    return Path(path)


def transpile_to_rust(c_dir: Path, compile_commands: Path, output_dir: Path, status_fn=None) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    log = make_stage_logger(output_dir)
    # c2rust requires absolute paths — relative paths cause a canonicalize panic
    compile_commands = compile_commands.resolve()
    output_dir_abs = output_dir.resolve()
    log.info(f"transpile_to_rust: compile_commands={compile_commands}, output_dir={output_dir_abs}")

    if status_fn:
        status_fn("Transpiling C → Rust with c2rust…")
    log.info("Running c2rust transpile")
    result = subprocess.run(
        ["c2rust", "transpile", str(compile_commands), "--output-dir", str(output_dir_abs)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    ok = result.returncode == 0

    # Save full c2rust output as a standalone log file
    (output_dir / "c2rust.log").write_text(
        f"=== COMMAND ===\nc2rust transpile {compile_commands} --output-dir {output_dir_abs}\n\n"
        f"=== STDOUT ===\n{result.stdout}\n"
        f"=== STDERR ===\n{result.stderr}\n"
        f"=== EXIT CODE: {result.returncode} ===\n"
    )

    rust_files = sorted(output_dir.glob("**/*.rs"))
    rust_file_strs = [str(f) for f in rust_files]

    if ok:
        log.info(f"c2rust OK — generated {len(rust_files)} Rust files")
    else:
        log.warning(f"c2rust exited with non-zero code; generated {len(rust_files)} Rust files")
        log.warning(result.stderr)

    # c2rust producing no .rs files is always a hard failure regardless of exit code
    if not rust_files:
        first_error = next(
            (line for line in result.stderr.splitlines() if line.strip()),
            result.stderr[:200] or "c2rust produced no Rust files",
        )
        raise ConversionError("c2rust", first_error)

    src_dir = output_dir / "src"
    lib_modules = [
        f for f in rust_files
        if f.parent == src_dir and f.name != "lib.rs" and not f.name.startswith("bench_")
    ]
    if not lib_modules:
        raise ConversionError(
            "c2rust",
            "No library Rust modules were generated (only benchmark/transient files).",
        )

    if status_fn:
        status_fn(f"Generated {len(rust_files)} Rust files")

    # Generate Cargo.toml
    cargo_toml_path = output_dir / "Cargo.toml"
    cargo_toml_path.write_text(CARGO_TOML_TEMPLATE)

    # Generate src/lib.rs that re-exports all modules.
    # c2rust places .rs files in src/ (not in the output root), so compare against src_dir.
    src_dir.mkdir(exist_ok=True)
    lib_rs = src_dir / "lib.rs"
    modules = sorted(
        f.stem for f in rust_files
        if f.parent == src_dir and f.name != "lib.rs"
    )
    mod_lines = "\n".join(f"pub mod {m};" for m in modules)
    lib_rs.write_text(
        f"#![allow(unused)]\n#![allow(non_snake_case)]\n#![allow(non_camel_case_types)]\n\n{mod_lines}\n"
    )
    log.info(f"Wrote src/lib.rs with modules: {modules}")

    # Strip known-stable feature flags (e.g. raw_ref_op) to avoid E0554 on stable Rust.
    for rs in rust_files:
        _fix_stable_rust_features(rs)

    (output_dir / "c2rust_result.json").write_text(json.dumps({
        "ok": ok,
        "stderr": result.stderr,
        "stdout": result.stdout,
    }, indent=2))

    log.info("Stage complete")
    return {
        "rust_files": rust_file_strs,
        "cargo_toml": str(cargo_toml_path),
        "ok": ok,
        "stderr": result.stderr,
    }

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

CARGO_TOML_TEMPLATE = """\
[package]
name = "fortran2rust_output"
version = "0.1.0"
edition = "2021"

[lib]
name = "fortran2rust_output"
crate-type = ["cdylib", "staticlib"]
"""


def ensure_c2rust() -> Path:
    from rich.progress import Progress, SpinnerColumn, TextColumn

    path = shutil.which("c2rust")
    if path:
        return Path(path)

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


def transpile_to_rust(c_dir: Path, compile_commands: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        ["c2rust", "transpile", str(compile_commands), "--output-dir", str(output_dir)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    ok = result.returncode == 0

    rust_files = sorted(output_dir.glob("**/*.rs"))
    rust_file_strs = [str(f) for f in rust_files]

    # Generate Cargo.toml
    cargo_toml_path = output_dir / "Cargo.toml"
    cargo_toml_path.write_text(CARGO_TOML_TEMPLATE)

    # Generate src/lib.rs that re-exports all modules
    src_dir = output_dir / "src"
    src_dir.mkdir(exist_ok=True)
    lib_rs = src_dir / "lib.rs"
    if not lib_rs.exists():
        modules = [f.stem for f in rust_files if f.parent == output_dir]
        mod_lines = "\n".join(f"pub mod {m};" for m in modules)
        lib_rs.write_text(
            f"#![allow(unused)]\n#![allow(non_snake_case)]\n#![allow(non_camel_case_types)]\n{mod_lines}\n"
        )

    (output_dir / "c2rust_result.json").write_text(json.dumps({
        "ok": ok,
        "stderr": result.stderr,
        "stdout": result.stdout,
    }, indent=2))

    return {
        "rust_files": rust_file_strs,
        "cargo_toml": str(cargo_toml_path),
        "ok": ok,
        "stderr": result.stderr,
    }

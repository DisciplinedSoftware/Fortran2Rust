from __future__ import annotations

import logging
from pathlib import Path

from fortran2rust.stages import _bench


class _FakeCompletedProcess:
    def __init__(self, returncode: int, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_run_rust_benchmarks_uses_wrapper_bins_and_lib_modules(tmp_path, monkeypatch) -> None:
    output_dir = tmp_path / "s6_llm_fix_rust"
    src_dir = output_dir / "src"
    baseline_dir = tmp_path / "s2_benchmark_generation"
    src_dir.mkdir(parents=True)
    baseline_dir.mkdir(parents=True)

    cargo_toml = output_dir / "Cargo.toml"
    cargo_toml.write_text(
        "[package]\n"
        "name = \"fortran2rust_output\"\n"
        "version = \"0.1.0\"\n"
        "edition = \"2021\"\n\n"
        "[lib]\n"
        "name = \"fortran2rust_output\"\n"
        "crate-type = [\"cdylib\", \"staticlib\"]\n"
    )

    (src_dir / "lib.rs").write_text("pub mod dasum;\n")
    (src_dir / "dasum.rs").write_text(
        "pub type integer = i32;\n"
        "pub type doublereal = f64;\n"
        "#[no_mangle]\n"
        "pub unsafe extern \"C\" fn dasum_(n: *mut integer, dx: *mut doublereal, incx: *mut integer) -> doublereal {\n"
        "    let _ = (n, dx, incx);\n"
        "    0.0\n"
        "}\n"
    )

    (baseline_dir / "bench_dasum.rs").write_text("pub fn main() {}\n")

    def fake_run(cmd, capture_output, text, timeout):
        assert cmd[:4] == ["cargo", "build", "--release", "--bins"]
        return _FakeCompletedProcess(1, stderr="link failed")

    monkeypatch.setattr(_bench.subprocess, "run", fake_run)

    result = _bench.run_rust_benchmarks(
        output_dir=output_dir,
        baseline_dir=baseline_dir,
        cargo_toml=cargo_toml,
        log=logging.getLogger("test"),
        status_fn=None,
    )

    assert result == {}

    updated_cargo = cargo_toml.read_text()
    assert 'crate-type = ["rlib", "cdylib", "staticlib"]' in updated_cargo
    assert '[[bin]]\nname = "bench_dasum"\npath = "src/__bench_bins/bench_dasum_main.rs"' in updated_cargo

    wrapper = src_dir / "__bench_bins" / "bench_dasum_main.rs"
    assert wrapper.exists()
    assert "fortran2rust_output::bench_dasum::main();" in wrapper.read_text()

    lib_rs = (src_dir / "lib.rs").read_text()
    assert "pub mod bench_dasum;" in lib_rs

"""
Regression tests ensuring that all-zero benchmark outputs are detected and
rejected rather than silently passing as a false numerical match.

Scenario: a broken driver (or a stub that never calls the function) writes
zeros to the output .bin file.  Without these guards every downstream stage
would see max_abs_diff == 0 and mark the result as PASS — which would mean
we never verify the function was actually executed.
"""
from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fortran2rust.stages import _bench, s2_benchmarks
from fortran2rust.exceptions import BenchmarkRuntimeError


# ── helpers ────────────────────────────────────────────────────────────────────

def _write_f64(path: Path, values) -> None:
    """Write a list of float64 values as a raw binary file."""
    np.array(values, dtype=np.float64).tofile(str(path))


def _minimal_rust_workspace(tmp_path: Path):
    """Scaffold a minimal Stage-6-style workspace with a bench_dasum module."""
    output_dir = tmp_path / "s6"
    src_dir = output_dir / "src"
    baseline_dir = tmp_path / "s2"
    src_dir.mkdir(parents=True)
    baseline_dir.mkdir(parents=True)

    (src_dir / "lib.rs").write_text("pub mod bench_dasum;\n")
    (src_dir / "bench_dasum.rs").write_text("pub fn main() {}\n")
    cargo_toml = output_dir / "Cargo.toml"
    cargo_toml.write_text(
        '[package]\nname = "fortran2rust_output"\nversion = "0.1.0"\nedition = "2021"\n'
        '[lib]\nname = "fortran2rust_output"\ncrate-type = ["rlib"]\n'
    )

    # Create a fake release binary so the file-exists check in run_rust_benchmarks passes.
    release_dir = output_dir / "target" / "release"
    release_dir.mkdir(parents=True)
    (release_dir / "bench_dasum").write_bytes(b"fake-elf")

    return output_dir, baseline_dir, cargo_toml


# ── Stage 6 (_bench.run_rust_benchmarks) ──────────────────────────────────────

def test_rust_all_zero_output_is_rejected(tmp_path, monkeypatch) -> None:
    """
    When the Rust binary writes an all-zero output the benchmark must be marked
    as failed, not as a pass with diff == 0.
    """
    output_dir, baseline_dir, cargo_toml = _minimal_rust_workspace(tmp_path)

    # Non-zero Fortran baseline — the function clearly returned real values.
    _write_f64(baseline_dir / "bench_dasum_output.bin", [12.5, 7.3, 99.1])

    def fake_run(cmd, capture_output, text, cwd=None, timeout=None):
        if cmd[0] == "cargo":
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        # Bench binary runs but writes all-zero output — simulates a stub.
        _write_f64(Path(cwd) / "bench_dasum_output.bin", [0.0, 0.0, 0.0])
        return SimpleNamespace(returncode=0, stdout="RUST_TIME_MS=1.0\n", stderr="")

    monkeypatch.setattr(_bench.subprocess, "run", fake_run)

    result = _bench.run_rust_benchmarks(
        output_dir=output_dir,
        baseline_dir=baseline_dir,
        cargo_toml=cargo_toml,
        log=logging.getLogger("test"),
    )

    assert "dasum" in result
    assert result["dasum"]["pass"] is False
    assert "all-zeros" in result["dasum"].get("run_error", "").lower()


def test_fortran_all_zero_baseline_is_rejected_in_stage6(tmp_path, monkeypatch) -> None:
    """
    When the Stage-2 Fortran baseline is all-zeros any Rust output (even matching
    zeros) must be rejected — we cannot trust a zero-vs-zero comparison.
    """
    output_dir, baseline_dir, cargo_toml = _minimal_rust_workspace(tmp_path)

    # All-zero Fortran baseline — Stage 2 driver never called the function.
    _write_f64(baseline_dir / "bench_dasum_output.bin", [0.0, 0.0, 0.0])

    def fake_run(cmd, capture_output, text, cwd=None, timeout=None):
        if cmd[0] == "cargo":
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        # Rust writes the same zeros — would be diff=0 without the guard.
        _write_f64(Path(cwd) / "bench_dasum_output.bin", [0.0, 0.0, 0.0])
        return SimpleNamespace(returncode=0, stdout="RUST_TIME_MS=1.0\n", stderr="")

    monkeypatch.setattr(_bench.subprocess, "run", fake_run)

    result = _bench.run_rust_benchmarks(
        output_dir=output_dir,
        baseline_dir=baseline_dir,
        cargo_toml=cargo_toml,
        log=logging.getLogger("test"),
    )

    assert "dasum" in result
    assert result["dasum"]["pass"] is False
    assert "all-zeros" in result["dasum"].get("run_error", "").lower()


# ── Stage 2 (s2_benchmarks.generate_benchmarks) ───────────────────────────────

def test_stage2_all_zero_fortran_baseline_raises(tmp_path, monkeypatch) -> None:
    """
    If the Fortran benchmark driver compiles and runs but writes an all-zero
    output, Stage 2 must raise BenchmarkRuntimeError so the pipeline aborts
    rather than propagating a useless baseline.
    """
    source_dir = tmp_path / "src"
    output_dir = tmp_path / "out"
    source_dir.mkdir()
    (source_dir / "dgemm.f").write_text("      END\n")
    dep_files = [source_dir / "dgemm.f"]

    def _fake_run(cmd, capture_output, text, timeout, cwd=None):
        if cmd[0] == "gfortran":
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        exe_name = Path(cmd[0]).name
        run_dir = Path(cwd) if cwd else output_dir
        if exe_name == "bench_dgemm":
            # Driver "succeeds" but writes all-zero output — broken driver.
            np.array([0.0], dtype=np.float64).tofile(str(run_dir / "bench_dgemm_output.bin"))
            return SimpleNamespace(returncode=0, stdout="FORTRAN_TIME_MS=1.0\n", stderr="")
        # Calibration and precision runs are allowed to silently fail.
        raise RuntimeError(f"fake subprocess for: {exe_name}")

    monkeypatch.setattr(s2_benchmarks.subprocess, "run", _fake_run)

    with pytest.raises(BenchmarkRuntimeError) as exc_info:
        s2_benchmarks.generate_benchmarks(
            source_dir=source_dir,
            entry_points=["dgemm"],
            dep_files=dep_files,
            output_dir=output_dir,
            call_graph={},
        )
    # The all-zeros detail is in the snippet attribute, not the main message.
    assert "all-zeros" in exc_info.value.snippet.lower()


def test_stage2_all_zero_scalar_baseline_is_discarded_not_raised(tmp_path, monkeypatch) -> None:
    """
    Scalar/test-style entry points with no numeric array parameters may produce
    all-zero output even with non-zero inputs (e.g. a THRESH-style test helper
    that never transforms its arguments).  Stage 2 should discard the entry point
    — recording it in discarded_entry_points — rather than raising or silently
    accepting a zero baseline.
    """
    source_dir = tmp_path / "src"
    output_dir = tmp_path / "out"
    source_dir.mkdir()
    (source_dir / "db1nrm2.f").write_text("      END\n")
    dep_files = [source_dir / "db1nrm2.f"]

    def _fake_parse(_fn_name, _source_dir):
        return {
            "is_function": False,
            "params": [
                {"name": "N", "type": "INTEGER", "is_array": False},
                {"name": "INCX", "type": "INTEGER", "is_array": False},
                {"name": "THRESH", "type": "DOUBLE PRECISION", "is_array": False},
            ],
            "return_type": None,
        }

    def _fake_run(cmd, capture_output, text, timeout, cwd=None):
        if cmd[0] == "gfortran":
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        exe_name = Path(cmd[0]).name
        run_dir = Path(cwd) if cwd else output_dir
        if exe_name == "bench_db1nrm2":
            # Still writes zero — function ignores its inputs.
            np.array([0.0], dtype=np.float64).tofile(str(run_dir / "bench_db1nrm2_output.bin"))
            return SimpleNamespace(returncode=0, stdout="FORTRAN_TIME_MS=1.0\n", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(s2_benchmarks, "_parse_fn_signature", _fake_parse)
    monkeypatch.setattr(s2_benchmarks.subprocess, "run", _fake_run)

    result = s2_benchmarks.generate_benchmarks(
        source_dir=source_dir,
        entry_points=["db1nrm2"],
        dep_files=dep_files,
        output_dir=output_dir,
        call_graph={},
    )

    # Entry point is not in benchmarks — it was discarded.
    assert "db1nrm2" not in result["benchmarks"]
    assert "db1nrm2" in result["discarded_entry_points"]

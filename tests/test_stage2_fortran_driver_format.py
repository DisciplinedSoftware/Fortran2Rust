from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from fortran2rust.stages import s2_benchmarks


def test_stage2_generates_free_form_fortran_drivers(monkeypatch, tmp_path: Path) -> None:
    source_dir = tmp_path / "src"
    output_dir = tmp_path / "out"
    source_dir.mkdir()

    for name in ("dgemm.f", "lsame.f", "xerbla.f"):
        (source_dir / name).write_text("      END\n")

    dep_files = [source_dir / "dgemm.f", source_dir / "lsame.f", source_dir / "xerbla.f"]
    call_graph = {"DGEMM": ["LSAME", "XERBLA"]}

    def _fake_run(cmd, capture_output, text, timeout, cwd=None):
        if cmd[0] == "gfortran":
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        exe_name = Path(cmd[0]).name
        run_dir = Path(cwd) if cwd else output_dir

        if exe_name == "bench_dgemm":
            (run_dir / "bench_dgemm_output.bin").write_bytes(b"\0" * 8)
            return SimpleNamespace(returncode=0, stdout="FORTRAN_TIME_MS=1.234\n", stderr="")

        if exe_name == "bench_dgemm_precision":
            (run_dir / "bench_dgemm_precision_output.bin").write_bytes(b"\0" * 8)
            return SimpleNamespace(returncode=0, stdout="PRECISION_TEST_DONE\n", stderr="")

        raise AssertionError(f"Unexpected command in test: {cmd}")

    monkeypatch.setattr(s2_benchmarks.subprocess, "run", _fake_run)

    result = s2_benchmarks.generate_benchmarks(
        source_dir=source_dir,
        entry_points=["dgemm"],
        dep_files=dep_files,
        output_dir=output_dir,
        call_graph=call_graph,
    )

    driver_path = output_dir / "bench_dgemm.f90"
    precision_driver_path = output_dir / "bench_dgemm_precision.f90"

    assert driver_path.exists()
    assert precision_driver_path.exists()
    assert not (output_dir / "bench_dgemm.f").exists()

    driver_text = driver_path.read_text()
    precision_driver_text = precision_driver_path.read_text()

    assert "\n     $" not in driver_text
    assert "\n     $" not in precision_driver_text
    assert "ACCESS='STREAM'" in driver_text

    compile_cmd = result["benchmarks"]["dgemm"]["compile_cmd"]
    assert any(str(arg).endswith("bench_dgemm.f90") for arg in compile_cmd)


def test_stage2_generates_precision_for_generic_entrypoint(monkeypatch, tmp_path: Path) -> None:
    source_dir = tmp_path / "src"
    output_dir = tmp_path / "out"
    source_dir.mkdir()

    for name in ("dasum.f", "lsame.f", "xerbla.f"):
        (source_dir / name).write_text("      END\n")

    dep_files = [source_dir / "dasum.f", source_dir / "lsame.f", source_dir / "xerbla.f"]
    call_graph = {"DASUM": ["LSAME", "XERBLA"]}

    def _fake_parse(_fn_name, _source_dir):
        return {
            "is_function": True,
            "return_type": "DOUBLE PRECISION",
            "params": [
                {"name": "N", "type": "INTEGER", "is_array": False},
                {"name": "DX", "type": "DOUBLE PRECISION", "is_array": True},
                {"name": "INCX", "type": "INTEGER", "is_array": False},
            ],
        }

    def _fake_run(cmd, capture_output, text, timeout, cwd=None):
        if cmd[0] == "gfortran":
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        exe_name = Path(cmd[0]).name
        run_dir = Path(cwd) if cwd else output_dir

        if exe_name == "bench_dasum":
            (run_dir / "bench_dasum_output.bin").write_bytes(b"\0" * 8)
            return SimpleNamespace(returncode=0, stdout="FORTRAN_TIME_MS=1.234\n", stderr="")

        if exe_name == "bench_dasum_precision":
            (run_dir / "bench_dasum_precision_output.bin").write_bytes(b"\0" * 8)
            return SimpleNamespace(returncode=0, stdout="FORTRAN_TIME_MS=1.111\n", stderr="")

        raise AssertionError(f"Unexpected command in test: {cmd}")

    monkeypatch.setattr(s2_benchmarks, "_parse_fn_signature", _fake_parse)
    monkeypatch.setattr(s2_benchmarks.subprocess, "run", _fake_run)

    result = s2_benchmarks.generate_benchmarks(
        source_dir=source_dir,
        entry_points=["dasum"],
        dep_files=dep_files,
        output_dir=output_dir,
        call_graph=call_graph,
    )

    assert (output_dir / "bench_dasum_precision.f90").exists()
    assert (output_dir / "dataset_dasum_precision_A.bin").exists()
    assert (output_dir / "dataset_dasum_precision_B.bin").exists()
    assert (output_dir / "dataset_dasum_expected.bin").exists()
    assert (output_dir / "dataset_dasum_precision_expected.bin").exists()
    assert "expected" in result["datasets"]["dasum"]
    assert "dasum_precision" in result["datasets"]
    assert "expected" in result["datasets"]["dasum_precision"]

    c_driver_text = (output_dir / "bench_dasum.c").read_text()
    assert "extern doublereal dasum_(integer *bench_n, doublereal *bench_dx, integer *bench_incx);" in c_driver_text
    assert "doublereal bench_result;" in c_driver_text
    assert "bench_result = dasum_(&bench_n, bench_dx, &bench_incx);" in c_driver_text
    assert "fwrite(&bench_result, sizeof(doublereal), 1, out);" in c_driver_text


def test_stage2_c_driver_is_generic_without_specialized_kernel_paths(monkeypatch, tmp_path: Path) -> None:
    source_dir = tmp_path / "src"
    output_dir = tmp_path / "out"
    source_dir.mkdir()

    for name in ("daxpy.f", "lsame.f", "xerbla.f"):
        (source_dir / name).write_text("      END\n")

    dep_files = [source_dir / "daxpy.f", source_dir / "lsame.f", source_dir / "xerbla.f"]
    call_graph = {"DAXPY": ["LSAME", "XERBLA"]}

    def _fake_run(cmd, capture_output, text, timeout, cwd=None):
        if cmd[0] == "gfortran":
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        exe_name = Path(cmd[0]).name
        run_dir = Path(cwd) if cwd else output_dir

        if exe_name == "bench_daxpy":
            (run_dir / "bench_daxpy_output.bin").write_bytes(b"\0" * 8)
            return SimpleNamespace(returncode=0, stdout="FORTRAN_TIME_MS=1.234\n", stderr="")

        if exe_name == "bench_daxpy_precision":
            (run_dir / "bench_daxpy_precision_output.bin").write_bytes(b"\0" * 8)
            return SimpleNamespace(returncode=0, stdout="FORTRAN_TIME_MS=1.111\n", stderr="")

        raise AssertionError(f"Unexpected command in test: {cmd}")

    monkeypatch.setattr(s2_benchmarks.subprocess, "run", _fake_run)

    s2_benchmarks.generate_benchmarks(
        source_dir=source_dir,
        entry_points=["daxpy"],
        dep_files=dep_files,
        output_dir=output_dir,
        call_graph=call_graph,
    )

    c_driver_text = (output_dir / "bench_daxpy.c").read_text()
    assert "/* TODO: verify function signature and call against the f2c output */" in c_driver_text
    assert 'read_bin("dataset_daxpy_A.bin"' in c_driver_text
    assert 'read_bin("dataset_daxpy_B.bin"' in c_driver_text
    assert "memcpy(" not in c_driver_text


def test_stage2_dbeg_c_driver_declares_logical_param(monkeypatch, tmp_path: Path) -> None:
    source_dir = tmp_path / "src"
    output_dir = tmp_path / "out"
    source_dir.mkdir()

    for name in ("dbeg.f", "lsame.f", "xerbla.f"):
        (source_dir / name).write_text("      END\n")

    dep_files = [source_dir / "dbeg.f", source_dir / "lsame.f", source_dir / "xerbla.f"]
    call_graph = {"DBEG": ["LSAME", "XERBLA"]}

    def _fake_parse(_fn_name, _source_dir):
        return {
            "is_function": True,
            "return_type": "DOUBLE PRECISION",
            "params": [
                {"name": "RESET", "type": "LOGICAL", "is_array": False},
            ],
        }

    def _fake_run(cmd, capture_output, text, timeout, cwd=None):
        if cmd[0] == "gfortran":
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        exe_name = Path(cmd[0]).name
        run_dir = Path(cwd) if cwd else output_dir

        if exe_name == "bench_dbeg":
            (run_dir / "bench_dbeg_output.bin").write_bytes(b"\0" * 8)
            return SimpleNamespace(returncode=0, stdout="FORTRAN_TIME_MS=1.234\n", stderr="")

        if exe_name == "bench_dbeg_precision":
            (run_dir / "bench_dbeg_precision_output.bin").write_bytes(b"\0" * 8)
            return SimpleNamespace(returncode=0, stdout="FORTRAN_TIME_MS=1.111\n", stderr="")

        raise AssertionError(f"Unexpected command in test: {cmd}")

    monkeypatch.setattr(s2_benchmarks, "_parse_fn_signature", _fake_parse)
    monkeypatch.setattr(s2_benchmarks.subprocess, "run", _fake_run)

    s2_benchmarks.generate_benchmarks(
        source_dir=source_dir,
        entry_points=["dbeg"],
        dep_files=dep_files,
        output_dir=output_dir,
        call_graph=call_graph,
    )

    c_driver_text = (output_dir / "bench_dbeg.c").read_text()
    assert "logical bench_reset;" in c_driver_text
    assert "bench_reset = 1;" in c_driver_text
    assert "extern doublereal dbeg_(logical *bench_reset);" in c_driver_text
    assert "dbeg_(&bench_reset);" in c_driver_text

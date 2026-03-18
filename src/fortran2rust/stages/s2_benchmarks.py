from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from jinja2 import BaseLoader, Environment

from ..exceptions import BenchmarkRuntimeError, CompilationError
from ._log import make_stage_logger

N_DEFAULT = 500  # matrix size giving ~1ms per DGEMM call
N_MIN = 64
N_MAX = 2000
TARGET_TIME_MS_MIN = 1.0
TARGET_TIME_MS_MAX = 100.0
MAX_CALIBRATION_STEPS = 6
VECTOR_N_MAX = 2_048_000
VECTOR_CALIBRATION_STEPS = 13


# ── Precision type system ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class _Precision:
    """Numeric type info used when generating typed benchmark code."""
    fortran_type: str   # e.g. "DOUBLE PRECISION"
    fortran_zero: str   # e.g. "0.0D0"
    fortran_one: str    # e.g. "1.0D0"
    c_type: str         # f2c typedef name, e.g. "doublereal"
    numpy_dtype: str    # numpy dtype string, e.g. "float64"
    is_complex: bool


_PREC_D = _Precision("DOUBLE PRECISION",  "0.0D0",          "1.0D0",          "doublereal",    "float64",    False)
_PREC_S = _Precision("REAL",              "0.0E0",          "1.0E0",          "real",          "float32",    False)
_PREC_Z = _Precision("COMPLEX*16",        "(0.0D0,0.0D0)",  "(1.0D0,0.0D0)", "doublecomplex", "complex128",  True)
_PREC_C = _Precision("COMPLEX",           "(0.0E0,0.0E0)",  "(1.0E0,0.0E0)", "complex",       "complex64",   True)

_BLAS_PREFIX_TO_PREC: dict[str, _Precision] = {
    "D": _PREC_D,
    "S": _PREC_S,
    "Z": _PREC_Z,
    "C": _PREC_C,
}

# Map normalised Fortran type strings (upper-case) → _Precision
_FORTRAN_TYPE_TO_PREC: dict[str, _Precision] = {
    "DOUBLE PRECISION": _PREC_D,
    "REAL*8":           _PREC_D,
    "REAL":             _PREC_S,
    "REAL*4":           _PREC_S,
    "COMPLEX*16":       _PREC_Z,
    "DOUBLE COMPLEX":   _PREC_Z,
    "COMPLEX*8":        _PREC_C,
    "COMPLEX":          _PREC_C,
}


def _blas_precision(fn_name: str) -> _Precision:
    """Return the _Precision for a BLAS function based on its first-letter convention."""
    return _BLAS_PREFIX_TO_PREC.get(fn_name[0].upper(), _PREC_D)


def _split_entity_list(s: str) -> list[str]:
    """Split a Fortran entity-declaration list by commas, respecting nested parentheses."""
    parts: list[str] = []
    depth = 0
    cur: list[str] = []
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    if cur:
        parts.append("".join(cur))
    return parts


def _parse_fn_signature(fn_name: str, source_dir: Path) -> dict | None:
    """
    Parse Fortran source files in *source_dir* to extract the signature of *fn_name*.

    Returns ``{'is_function': bool, 'params': [{'name', 'type', 'is_array'}]}``
    where every *type* is a normalised upper-case Fortran type string, or ``None``
    when the function cannot be found / parsed.
    """
    try:
        from fparser.two import Fortran2003
        from fparser.two.utils import walk
        from fparser.two.parser import ParserFactory
        from fparser.common.readfortran import FortranFileReader
    except ImportError:
        return None

    fn_upper = fn_name.upper()
    parser = ParserFactory().create(std="f2003")

    fortran_sources = sorted(
        [p for p in source_dir.iterdir() if p.is_file() and p.suffix.lower() in {".f", ".f90"}]
    )

    for f in fortran_sources:
        try:
            reader = FortranFileReader(str(f), ignore_comments=True)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tree = parser(reader)
        except Exception:
            continue

        for subprog in walk(tree, (Fortran2003.Subroutine_Subprogram,
                                   Fortran2003.Function_Subprogram)):
            is_function = isinstance(subprog, Fortran2003.Function_Subprogram)
            stmts = (walk(subprog, Fortran2003.Subroutine_Stmt)
                     or walk(subprog, Fortran2003.Function_Stmt))
            if not stmts:
                continue
            stmt = stmts[0]
            if str(stmt.items[1]).upper() != fn_upper:
                continue

            # Parameter names in declaration order
            dummy_list = stmt.items[2]
            param_names: list[str] = []
            if dummy_list is not None:
                param_names = [
                    p.strip().upper()
                    for p in str(dummy_list).split(",")
                    if p.strip()
                ]

            # Build name → {type, is_array} from Type_Declaration_Stmt nodes
            type_map: dict[str, dict] = {}
            for decl in walk(subprog, Fortran2003.Type_Declaration_Stmt):
                raw_type = re.sub(r"\s+", " ", str(decl.items[0]).upper().strip())
                entity_str = str(decl.items[2]) if decl.items[2] is not None else ""
                for var in _split_entity_list(entity_str):
                    var = var.strip()
                    if not var:
                        continue
                    m = re.match(r"^([A-Z_][A-Z0-9_$]*)\s*(\(.*)?$", var.upper())
                    if m:
                        vname = m.group(1)
                        is_array = m.group(2) is not None
                        type_map[vname] = {"type": raw_type, "is_array": is_array}

            return_type = None
            if is_function:
                raw_ret = re.sub(r"\s+", " ", str(stmt.items[0]).upper().strip())
                if raw_ret and raw_ret != "NONE":
                    return_type = raw_ret

            if is_function and return_type is None:
                fn_decl = type_map.get(fn_upper)
                if fn_decl and fn_decl.get("type"):
                    return_type = fn_decl["type"]

            params = [
                {
                    "name": pname,
                    "type": type_map.get(pname, {}).get("type", "DOUBLE PRECISION"),
                    "is_array": type_map.get(pname, {}).get("is_array", False),
                }
                for pname in param_names
            ]
            return {
                "is_function": is_function,
                "params": params,
                "return_type": return_type,
            }

    return None


def _dominant_precision(sig: dict | None) -> _Precision:
    """Return the _Precision of the first numeric (non-integer, non-char) parameter."""
    if sig is None:
        return _PREC_D
    for p in sig["params"]:
        prec = _FORTRAN_TYPE_TO_PREC.get(p["type"])
        if prec is not None:
            return prec
    return _PREC_D

KNOWN_BLAS = {
    "DGEMM", "SGEMM", "ZGEMM", "CGEMM",
    "DGEMV", "SGEMV", "ZGEMV", "CGEMV",
    "DDOT", "SDOT", "DNRM2", "SNRM2",
    "DAXPY", "SAXPY", "DCOPY", "SCOPY",
    "DSCAL", "SSCAL", "IDAMAX", "ISAMAX",
    "DTRSM", "STRSM", "DTRMM", "STRMM",
    "DSYMM", "SSYMM", "DSYRK", "SSYRK",
    "DSYR2K", "SSYR2K", "DTRMV", "STRMV",
}

KNOWN_GEMM = {"DGEMM", "SGEMM", "ZGEMM", "CGEMM"}
KNOWN_VECTOR_BLAS = {
    "DASUM", "SASUM", "DNRM2", "SNRM2", "DDOT", "SDOT",
    "IDAMAX", "ISAMAX", "DAXPY", "SAXPY", "DCOPY", "SCOPY", "DSCAL", "SSCAL",
    "DB1NRM2",
}


def _is_vector_blas(fn_name: str) -> bool:
    return fn_name.upper() in KNOWN_VECTOR_BLAS


def _resolve_fortran_deps(
    source_dir: Path,
    dep_files: list[Path],
    call_graph: dict[str, list[str]] | None,
    ep_upper: str,
) -> list[str]:
    """Resolve Fortran dependency source files for compiling a benchmark driver."""
    fortran_dep_paths: list[Path] = [Path(f).resolve() for f in dep_files if Path(f).exists()]
    extra_symbols = set((call_graph or {}).get(ep_upper, []))
    extra_symbols.add(ep_upper)
    for sym in extra_symbols:
        sym_lower = sym.lower()
        for ext in (".f", ".f90"):
            candidate = source_dir / f"{sym_lower}{ext}"
            if candidate.exists():
                fortran_dep_paths.append(candidate.resolve())
    return [str(p) for p in sorted(set(fortran_dep_paths))]


def _make_fortran_driver_source(fn_name: str, n: int, sig: dict | None, prec: _Precision) -> str:
    """Return the standard Fortran benchmark driver source for *fn_name* at size *n*."""
    fn_upper = fn_name.upper()
    if fn_upper in KNOWN_GEMM:
        return _make_dgemm_driver(fn_name, n, prec)
    if fn_upper.endswith("AXPY") and not prec.is_complex:
        return _make_axpy_driver(fn_name, n, prec)
    return _make_generic_driver(fn_name, n, sig, prec)


def _calibrate_benchmark_size(
    fn_name: str,
    source_dir: Path,
    dep_files: list[Path],
    output_dir: Path,
    call_graph: dict[str, list[str]] | None,
    sig: dict | None,
    prec: _Precision,
    log,
    matrix_n_max: int,
    timing_max_runs: int,
    dataset_reuse_every: int,
    status_fn=None,
) -> tuple[int, int]:
    """
    Choose a dataset size N for benchmarking by iterating until measured runtime
    for a single call lands in [TARGET_TIME_MS_MIN, TARGET_TIME_MS_MAX].
    """
    fn_upper = fn_name.upper()
    fn_lower = fn_name.lower()
    n = N_DEFAULT
    n_max = VECTOR_N_MAX if _is_vector_blas(fn_name) else min(N_MAX, matrix_n_max)
    max_steps = VECTOR_CALIBRATION_STEPS if _is_vector_blas(fn_name) else MAX_CALIBRATION_STEPS
    fortran_deps = _resolve_fortran_deps(source_dir, dep_files, call_graph, fn_upper)

    for step in range(max_steps):
        if status_fn:
            status_fn(f"Calibrating benchmark size for {fn_name} (N={n})…")
        log.info(f"Calibrating {fn_name}: attempt {step + 1}/{max_steps}, N={n}")

        calib_seed = 42 + (step // max(1, dataset_reuse_every))
        generate_dataset(fn_name, n, output_dir, prec=prec, seed=calib_seed)
        driver_src = _make_fortran_driver_source(fn_name, n, sig, prec)
        driver_file = output_dir / f"bench_{fn_lower}_calib.f90"
        driver_file.write_text(driver_src)
        exe_path = output_dir / f"bench_{fn_lower}_calib"
        compile_cmd = [
            "gfortran", "-O2",
            "-ffunction-sections", "-fdata-sections",
            str(driver_file.resolve()),
            *fortran_deps,
            "-Wl,--gc-sections", "-Wl,--allow-multiple-definition", "-lm",
            "-o", str(exe_path.resolve()),
        ]

        try:
            result = subprocess.run(compile_cmd, capture_output=True, text=True, timeout=120)
            if result.returncode != 0:
                log.warning(f"  Calibration compile failed for {fn_name} at N={n}; using default N={N_DEFAULT}")
                return N_DEFAULT, 1

            run_result = subprocess.run(
                [str(exe_path.resolve())],
                capture_output=True,
                text=True,
                cwd=str(output_dir.resolve()),
                timeout=300,
            )
            if run_result.returncode != 0:
                log.warning(f"  Calibration run failed for {fn_name} at N={n}; using default N={N_DEFAULT}")
                return N_DEFAULT, 1

            match = re.search(r"FORTRAN_TIME_MS=\s*([\d.]+)", run_result.stdout)
            if not match:
                log.warning(f"  Calibration could not parse timing for {fn_name} at N={n}; using current N")
                return n, 1

            elapsed_ms = float(match.group(1))
            log.info(f"  Calibration timing for {fn_name}: N={n}, t={elapsed_ms:.4f} ms")

            if TARGET_TIME_MS_MIN <= elapsed_ms <= TARGET_TIME_MS_MAX:
                return n, 1
            if elapsed_ms < TARGET_TIME_MS_MIN:
                if n >= n_max:
                    est_runs = int((TARGET_TIME_MS_MIN + max(elapsed_ms, 1e-6) - 1e-9) / max(elapsed_ms, 1e-6))
                    timing_runs = max(1, min(timing_max_runs, est_runs))
                    log.info(
                        "  Calibration %s: N=%s below %.1fms; using %s timing run(s)",
                        fn_name,
                        n,
                        TARGET_TIME_MS_MIN,
                        timing_runs,
                    )
                    return n, timing_runs
                n = min(n_max, n * 2)
            else:
                if n <= N_MIN:
                    break
                n = max(N_MIN, n // 2)
        except Exception as e:
            log.warning(f"  Calibration exception for {fn_name} at N={n}: {e}; using default N={N_DEFAULT}")
            return N_DEFAULT, 1

    log.warning(
        f"  Calibration for {fn_name} ended at N={n} outside target "
        f"[{TARGET_TIME_MS_MIN:.1f}, {TARGET_TIME_MS_MAX:.1f}] ms"
    )
    return n, 1


def generate_precision_dataset(fn_name: str, N: int, output_dir: Path,
                               prec: _Precision = _PREC_D) -> dict[str, Path]:
    """Generate near-cancellation dataset for precision testing (fixed seed for reproducibility)."""
    rng = np.random.default_rng(43)  # distinct seed from main dataset
    fn_lo = fn_name.lower()
    dtype = np.dtype(prec.numpy_dtype)
    EPS = 1e-8
    if _is_vector_blas(fn_name):
        if prec.is_complex:
            A = (1.0 + EPS * (rng.random(N) - 0.5) + 1j * EPS * (rng.random(N) - 0.5)).astype(dtype)
            B = (1.0 + EPS * (rng.random(N) - 0.5) + 1j * EPS * (rng.random(N) - 0.5)).astype(dtype)
        else:
            A = (1.0 + EPS * (rng.random(N).astype(dtype) - 0.5))
            B = (1.0 + EPS * (rng.random(N).astype(dtype) - 0.5))
    else:
        if prec.is_complex:
            A = (1.0 + EPS * (rng.random((N, N)) - 0.5) + 1j * EPS * (rng.random((N, N)) - 0.5)).astype(dtype)
            B = (1.0 + EPS * (rng.random((N, N)) - 0.5) + 1j * EPS * (rng.random((N, N)) - 0.5)).astype(dtype)
        else:
            A = (1.0 + EPS * (rng.random((N, N)).astype(dtype) - 0.5))
            B = (1.0 + EPS * (rng.random((N, N)).astype(dtype) - 0.5))
    a_path = output_dir / f"dataset_{fn_lo}_precision_A.bin"
    b_path = output_dir / f"dataset_{fn_lo}_precision_B.bin"
    A.flatten(order="F").tofile(str(a_path))
    B.flatten(order="F").tofile(str(b_path))
    return {"A": a_path, "B": b_path}


def generate_dataset(fn_name: str, N: int, output_dir: Path,
                     prec: _Precision = _PREC_D, seed: int = 42) -> dict[str, Path]:
    """Generate common input dataset files (typed binary) shared by Fortran/C/Rust."""
    rng = np.random.default_rng(seed)
    fn_upper = fn_name.upper()
    dtype = np.dtype(prec.numpy_dtype)

    if fn_upper in KNOWN_BLAS:
        if _is_vector_blas(fn_name):
            if prec.is_complex:
                A = (rng.random(N) + 1j * rng.random(N)).astype(dtype)
                B = (rng.random(N) + 1j * rng.random(N)).astype(dtype)
                params = np.array([1.0 + 0j, 0.0 + 0j], dtype=dtype)
            else:
                A = rng.random(N).astype(dtype)
                B = rng.random(N).astype(dtype)
                params = np.array([1.0, 0.0], dtype=dtype)
        else:
            if prec.is_complex:
                A = (rng.random((N, N)) + 1j * rng.random((N, N))).astype(dtype)
                B = (rng.random((N, N)) + 1j * rng.random((N, N))).astype(dtype)
                params = np.array([1.0 + 0j, 0.0 + 0j], dtype=dtype)
            else:
                A = rng.random((N, N)).astype(dtype)
                B = rng.random((N, N)).astype(dtype)
                params = np.array([1.0, 0.0], dtype=dtype)

        a_path = output_dir / f"dataset_{fn_name.lower()}_A.bin"
        b_path = output_dir / f"dataset_{fn_name.lower()}_B.bin"
        p_path = output_dir / f"dataset_{fn_name.lower()}_params.bin"

        # Column-major (Fortran order) for A and B so Fortran reads them directly
        A.flatten(order="F").tofile(str(a_path))
        B.flatten(order="F").tofile(str(b_path))
        params.tofile(str(p_path))

        return {"A": a_path, "B": b_path, "params": p_path}
    else:
        # Generic: write two NxN matrices using the dominant precision
        if prec.is_complex:
            A = (rng.random((N, N)) + 1j * rng.random((N, N))).astype(dtype)
            B = (rng.random((N, N)) + 1j * rng.random((N, N))).astype(dtype)
        else:
            A = rng.random((N, N)).astype(dtype)
            B = rng.random((N, N)).astype(dtype)
        a_path = output_dir / f"dataset_{fn_name.lower()}_A.bin"
        b_path = output_dir / f"dataset_{fn_name.lower()}_B.bin"
        A.flatten(order="F").tofile(str(a_path))
        B.flatten(order="F").tofile(str(b_path))
        return {"A": a_path, "B": b_path}


def _make_dgemm_driver(fn_name: str, N: int, prec: _Precision) -> str:
    fn_up = fn_name.upper()
    fn_lo = fn_name.lower()
    ftype = prec.fortran_type
    zero  = prec.fortran_zero
    return f"""\
      PROGRAM BENCH_{fn_up}
      IMPLICIT NONE
      INTEGER, PARAMETER :: N = {N}
      {ftype} :: A(N,N), B(N,N), C(N,N)
      {ftype} :: ALPHA, BETA
            DOUBLE PRECISION :: ELAPSED, T1, T2
      INTEGER :: I, J

    ! Load shared dataset (raw binary, column-major)
    OPEN(10, FILE='dataset_{fn_lo}_A.bin', FORM='UNFORMATTED', ACCESS='STREAM', STATUS='OLD')
      READ(10) A
      CLOSE(10)

    OPEN(11, FILE='dataset_{fn_lo}_B.bin', FORM='UNFORMATTED', ACCESS='STREAM', STATUS='OLD')
      READ(11) B
      CLOSE(11)

    OPEN(12, FILE='dataset_{fn_lo}_params.bin', FORM='UNFORMATTED', ACCESS='STREAM', STATUS='OLD')
      READ(12) ALPHA
      READ(12) BETA
      CLOSE(12)

      DO I = 1, N
        DO J = 1, N
          C(I,J) = {zero}
        END DO
      END DO

      ! Time ONLY the computation, not I/O
            CALL CPU_TIME(T1)
            CALL {fn_up}('N','N',N,N,N,ALPHA,A,N,B,N,BETA,C,N)
            CALL CPU_TIME(T2)
            ELAPSED = (T2-T1) * 1000.0D0

    ! Write output (column-major raw binary for numpy comparison)
    OPEN(13, FILE='bench_{fn_lo}_output.bin', FORM='UNFORMATTED', ACCESS='STREAM', STATUS='REPLACE')
      WRITE(13) C
      CLOSE(13)

    WRITE(*,'(A,F16.8)') 'FORTRAN_TIME_MS=', ELAPSED
      END PROGRAM
"""


def _make_dgemm_precision_driver(fn_name: str, N: int, prec: _Precision) -> str:
    """Near-cancellation precision test driver — reads shared dataset for cross-language reproducibility."""
    fn_up = fn_name.upper()
    fn_lo = fn_name.lower()
    ftype = prec.fortran_type
    zero  = prec.fortran_zero
    one   = prec.fortran_one
    return f"""\
      PROGRAM BENCH_{fn_up}_PREC
      IMPLICIT NONE
      INTEGER, PARAMETER :: N = {N}
      {ftype} :: A(N,N), B(N,N), C(N,N)
      {ftype} :: ALPHA, BETA
      INTEGER :: I, J

      ALPHA = {one}
      BETA = {zero}

    ! Load shared near-cancellation dataset (same data used by C and Rust)
    OPEN(10, FILE='dataset_{fn_lo}_precision_A.bin', FORM='UNFORMATTED', ACCESS='STREAM', STATUS='OLD')
      READ(10) A
      CLOSE(10)
    OPEN(11, FILE='dataset_{fn_lo}_precision_B.bin', FORM='UNFORMATTED', ACCESS='STREAM', STATUS='OLD')
      READ(11) B
      CLOSE(11)

      DO I = 1, N
        DO J = 1, N
          C(I,J) = {zero}
        END DO
      END DO

      CALL {fn_up}('N','N',N,N,N,ALPHA,A,N,B,N,BETA,C,N)

    OPEN(13, FILE='bench_{fn_lo}_precision_output.bin', FORM='UNFORMATTED', ACCESS='STREAM', STATUS='REPLACE')
      WRITE(13) C
      CLOSE(13)

      WRITE(*,*) 'PRECISION_TEST_DONE'
      END PROGRAM
"""


def _make_axpy_driver(fn_name: str, N: int, prec: _Precision) -> str:
    fn_up = fn_name.upper()
    fn_lo = fn_name.lower()
    ftype = prec.fortran_type
    return f"""\
PROGRAM BENCH_{fn_up}
IMPLICIT NONE
INTEGER, PARAMETER :: N = {N}
{ftype} :: X(N), Y(N), ALPHA
INTEGER :: INCX, INCY
DOUBLE PRECISION :: ELAPSED, T1, T2

OPEN(10, FILE='dataset_{fn_lo}_A.bin', FORM='UNFORMATTED', ACCESS='STREAM', STATUS='OLD')
READ(10) X
CLOSE(10)

OPEN(11, FILE='dataset_{fn_lo}_B.bin', FORM='UNFORMATTED', ACCESS='STREAM', STATUS='OLD')
READ(11) Y
CLOSE(11)

OPEN(12, FILE='dataset_{fn_lo}_params.bin', FORM='UNFORMATTED', ACCESS='STREAM', STATUS='OLD')
READ(12) ALPHA
CLOSE(12)

INCX = 1
INCY = 1

CALL CPU_TIME(T1)
CALL {fn_up}(N, ALPHA, X, INCX, Y, INCY)
CALL CPU_TIME(T2)
ELAPSED = (T2-T1) * 1000.0D0

OPEN(13, FILE='bench_{fn_lo}_output.bin', FORM='UNFORMATTED', ACCESS='STREAM', STATUS='REPLACE')
WRITE(13) Y
CLOSE(13)

WRITE(*,'(A,F16.8)') 'FORTRAN_TIME_MS=', ELAPSED
END PROGRAM
"""


def _make_axpy_precision_driver(fn_name: str, N: int, prec: _Precision) -> str:
    fn_up = fn_name.upper()
    fn_lo = fn_name.lower()
    ftype = prec.fortran_type
    one = prec.fortran_one
    return f"""\
PROGRAM BENCH_{fn_up}_PREC
IMPLICIT NONE
INTEGER, PARAMETER :: N = {N}
{ftype} :: X(N), Y(N), ALPHA
INTEGER :: INCX, INCY

OPEN(10, FILE='dataset_{fn_lo}_precision_A.bin', FORM='UNFORMATTED', ACCESS='STREAM', STATUS='OLD')
READ(10) X
CLOSE(10)

OPEN(11, FILE='dataset_{fn_lo}_precision_B.bin', FORM='UNFORMATTED', ACCESS='STREAM', STATUS='OLD')
READ(11) Y
CLOSE(11)

ALPHA = {one}
INCX = 1
INCY = 1
CALL {fn_up}(N, ALPHA, X, INCX, Y, INCY)

OPEN(13, FILE='bench_{fn_lo}_precision_output.bin', FORM='UNFORMATTED', ACCESS='STREAM', STATUS='REPLACE')
WRITE(13) Y
CLOSE(13)

WRITE(*,*) 'PRECISION_TEST_DONE'
END PROGRAM
"""


def _make_generic_driver(fn_name: str, N: int,
                         sig: dict | None, prec: _Precision) -> str:
    """
    Generate a Fortran benchmark driver for a non-BLAS function.

    When *sig* is available (parsed from source), produces a type-correct driver
    with a proper typed call.  Falls back to a DGEMM-shaped placeholder using
    *prec* when the signature cannot be parsed.
    """
    fn_up = fn_name.upper()
    fn_lo = fn_name.lower()
    vector_mode = _is_vector_blas(fn_name)

    if sig is None:
        # Fallback: typed placeholder with the dominant precision
        ftype = prec.fortran_type
        zero  = prec.fortran_zero
        one   = prec.fortran_one
        return f"""\
      PROGRAM BENCH_{fn_up}
      IMPLICIT NONE
      INTEGER, PARAMETER :: N = {N}
      {ftype} :: A(N,N), B(N,N), C(N,N)
      {ftype} :: ALPHA, BETA
    DOUBLE PRECISION :: ELAPSED, T1, T2
      INTEGER :: I, J

    OPEN(10, FILE='dataset_{fn_lo}_A.bin', FORM='UNFORMATTED', ACCESS='STREAM', STATUS='OLD')
      READ(10) A
      CLOSE(10)
    OPEN(11, FILE='dataset_{fn_lo}_B.bin', FORM='UNFORMATTED', ACCESS='STREAM', STATUS='OLD')
      READ(11) B
      CLOSE(11)

      ALPHA = {one}
      BETA = {zero}
      DO I = 1, N
        DO J = 1, N
          C(I,J) = {zero}
        END DO
      END DO

            CALL CPU_TIME(T1)
            ! TODO: replace with the correct {fn_up}(...) call
            CALL {fn_up}('N','N',N,N,N,ALPHA,A,N,B,N,BETA,C,N)
            CALL CPU_TIME(T2)
            ELAPSED = (T2-T1) * 1000.0D0

    OPEN(13, FILE='bench_{fn_lo}_output.bin', FORM='UNFORMATTED', ACCESS='STREAM', STATUS='REPLACE')
      WRITE(13) C
      CLOSE(13)

    WRITE(*,'(A,F16.8)') 'FORTRAN_TIME_MS=', ELAPSED
      END PROGRAM
"""
    # ── Signature-aware driver ────────────────────────────────────────────────
    params     = sig["params"]
    is_fn      = sig["is_function"]
    decls:      list[str] = []
    inits:      list[str] = []
    file_opens: list[str] = []
    call_args:  list[str] = []
    file_n  = 10
    arr_idx = 0   # index into dataset_A / dataset_B files

    for p in params:
        pname  = p["name"]
        ptype  = p["type"]
        bvar   = f"BENCH_{pname}"
        p_prec = _FORTRAN_TYPE_TO_PREC.get(ptype)

        if "CHARACTER" in ptype:
            decls.append(f"      CHARACTER*1 :: {bvar}")
            inits.append(f"      {bvar} = 'N'")
            call_args.append(bvar)

        elif "INTEGER" in ptype:
            decls.append(f"      INTEGER :: {bvar}")
            pu = pname.upper()
            if pu.startswith("LD") or pu in ("LDA", "LDB", "LDC", "LDQ", "LDT"):
                inits.append(f"      {bvar} = N")
            elif pu.startswith("INC") or pu in ("INCX", "INCY"):
                inits.append(f"      {bvar} = 1")
            else:
                inits.append(f"      {bvar} = N")
            call_args.append(bvar)

        elif "LOGICAL" in ptype:
            decls.append(f"      LOGICAL :: {bvar}")
            inits.append(f"      {bvar} = .TRUE.")
            call_args.append(bvar)

        elif p_prec is not None:
            if p["is_array"]:
                if vector_mode:
                    decls.append(f"      {ptype} :: {bvar}(N)")
                else:
                    decls.append(f"      {ptype} :: {bvar}(N,N)")
                ds = "A" if arr_idx == 0 else "B"
                file_opens.append(
                    f"      OPEN({file_n}, FILE='dataset_{fn_lo}_{ds}.bin', FORM='UNFORMATTED', ACCESS='STREAM', STATUS='OLD')\n"
                    f"      READ({file_n}) {bvar}\n"
                    f"      CLOSE({file_n})"
                )
                file_n  += 1
                arr_idx += 1
                call_args.append(bvar)
            else:
                decls.append(f"      {ptype} :: {bvar}")
                pu = pname.upper()
                if pu in ("ALPHA", "SCALE", "DA", "SA", "ZA", "CA", "A", "W"):
                    inits.append(f"      {bvar} = {p_prec.fortran_one}")
                else:
                    inits.append(f"      {bvar} = {p_prec.fortran_zero}")
                call_args.append(bvar)
        else:
            # Unknown / unsupported type — declare as a comment placeholder
            decls.append(f"      ! {ptype} :: {bvar}  ! TODO: unsupported type")
            call_args.append(bvar)

    # Result variable for FUNCTIONs
    result_decl = ""
    func_decl = ""
    call_stmt   = ""
    if is_fn:
        ret_prec = _FORTRAN_TYPE_TO_PREC.get(str(sig.get("return_type", "")).upper(), prec)
        result_decl = f"      {ret_prec.fortran_type} :: BENCH_RESULT\n"
        func_decl = f"      {ret_prec.fortran_type} :: {fn_up}\n      EXTERNAL {fn_up}\n"
        call_stmt   = f"BENCH_RESULT = {fn_up}({', '.join(call_args)})"
        out_var     = "BENCH_RESULT"
    else:
        call_stmt = f"CALL {fn_up}({', '.join(call_args)})"
        # Write the last float array as output, or first if none
        float_arrays = [p for p in params if p["is_array"] and p["type"] in _FORTRAN_TYPE_TO_PREC]
        if float_arrays:
            out_var = f"BENCH_{float_arrays[-1]['name']}"
        else:
            float_scalars = [p for p in params if (not p["is_array"]) and p["type"] in _FORTRAN_TYPE_TO_PREC]
            out_var = f"BENCH_{float_scalars[-1]['name']}" if float_scalars else None

    decl_block      = "\n".join(decls)
    init_block      = "\n".join(inits)
    file_open_block = "\n".join(file_opens)

    output_block = ""
    if out_var:
        output_block = (
            f"      OPEN(13, FILE='bench_{fn_lo}_output.bin', FORM='UNFORMATTED', ACCESS='STREAM', STATUS='REPLACE')\n"
            f"      WRITE(13) {out_var}\n"
            f"      CLOSE(13)\n"
        )

    return f"""\
      PROGRAM BENCH_{fn_up}
      IMPLICIT NONE
      INTEGER, PARAMETER :: N = {N}
{result_decl}{func_decl}{decl_block}
            DOUBLE PRECISION :: ELAPSED, T1, T2

{file_open_block}
{init_block}

            CALL CPU_TIME(T1)
            {call_stmt}
            CALL CPU_TIME(T2)
            ELAPSED = (T2-T1) * 1000.0D0

{output_block}
    WRITE(*,'(A,F16.8)') 'FORTRAN_TIME_MS=', ELAPSED
      END PROGRAM
"""


def _make_generic_precision_driver(fn_name: str, N: int,
                                   sig: dict | None, prec: _Precision) -> str:
    """
    Generate a near-cancellation precision Fortran benchmark driver for a
    non-specialized entry point by adapting the generic driver template to
    read precision datasets and emit a precision output artifact.
    """
    fn_up = fn_name.upper()
    fn_lo = fn_name.lower()
    src = _make_generic_driver(fn_name, N, sig, prec)
    src = src.replace(f"PROGRAM BENCH_{fn_up}", f"PROGRAM BENCH_{fn_up}_PREC", 1)
    src = src.replace(f"dataset_{fn_lo}_A.bin", f"dataset_{fn_lo}_precision_A.bin")
    src = src.replace(f"dataset_{fn_lo}_B.bin", f"dataset_{fn_lo}_precision_B.bin")
    src = src.replace(f"bench_{fn_lo}_output.bin", f"bench_{fn_lo}_precision_output.bin")
    return src


def _make_c_blas_driver(fn_name: str, N: int, prec: _Precision) -> str:
    """
    C benchmark driver for a BLAS dgemm-style function.
    Reads the numpy-generated binary dataset, calls the f2c-transpiled function,
    writes the output binary, and prints timing. Does NOT go through f2c.
    """
    fn_lo  = fn_name.lower()
    fn_ext = fn_lo + "_"  # f2c appends underscore
    ct     = prec.c_type
    return f"""\
/* C benchmark driver for {fn_name} — generated by Fortran2Rust */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "f2c.h"

#define BENCH_N {N}

/* f2c signature: char* args are followed by their ftnlen lengths at the end */
extern int {fn_ext}(char *transa, char *transb,
                    integer *m, integer *n, integer *k,
                    {ct} *alpha, {ct} *a, integer *lda,
                    {ct} *b, integer *ldb, {ct} *beta,
                    {ct} *c__, integer *ldc,
                    ftnlen transa_len, ftnlen transb_len);

static {ct} bench_A[BENCH_N * BENCH_N];
static {ct} bench_B[BENCH_N * BENCH_N];
static {ct} bench_C[BENCH_N * BENCH_N];

static void read_bin(const char *path, {ct} *buf, size_t count) {{
    FILE *fp = fopen(path, "rb");
    if (!fp) {{ perror(path); exit(1); }}
    if (fread(buf, sizeof({ct}), count, fp) != count) {{
        fprintf(stderr, "Short read: %s\\n", path); exit(1);
    }}
    fclose(fp);
}}

int main(void) {{
    {ct} params[2];
    read_bin("dataset_{fn_lo}_A.bin", bench_A, (size_t)BENCH_N * BENCH_N);
    read_bin("dataset_{fn_lo}_B.bin", bench_B, (size_t)BENCH_N * BENCH_N);
    read_bin("dataset_{fn_lo}_params.bin", params, 2);
    {ct} alpha = params[0], beta = params[1];

    integer m = BENCH_N, n = BENCH_N, k = BENCH_N;
    integer lda = BENCH_N, ldb = BENCH_N, ldc = BENCH_N;
    char transa = 'N', transb = 'N';

    memset(bench_C, 0, sizeof(bench_C));

    /* Warm up */
    {fn_ext}(&transa, &transb, &m, &n, &k, &alpha, bench_A, &lda, bench_B, &ldb, &beta, bench_C, &ldc, 1, 1);
    memset(bench_C, 0, sizeof(bench_C));

    struct timespec t1, t2;
    clock_gettime(CLOCK_MONOTONIC, &t1);
    for (int iter = 0; iter < 10; iter++) {{
        {fn_ext}(&transa, &transb, &m, &n, &k, &alpha, bench_A, &lda, bench_B, &ldb, &beta, bench_C, &ldc, 1, 1);
    }}
    clock_gettime(CLOCK_MONOTONIC, &t2);
    double elapsed_ms = ((t2.tv_sec - t1.tv_sec) * 1e9 + (t2.tv_nsec - t1.tv_nsec)) / 1e6 / 10.0;

    FILE *out = fopen("bench_{fn_lo}_output.bin", "wb");
    if (!out) {{ perror("bench_{fn_lo}_output.bin"); exit(1); }}
    fwrite(bench_C, sizeof({ct}), (size_t)BENCH_N * BENCH_N, out);
    fclose(out);

    printf("C_TIME_MS=%.4f\\n", elapsed_ms);
    return 0;
}}
"""


def _make_c_blas_precision_driver(fn_name: str, N: int, prec: _Precision) -> str:
    """
    C near-cancellation precision benchmark driver for a BLAS dgemm-style function.
    Reads the shared precision dataset (generated with fixed seed 43), calls the
    f2c-transpiled function, and writes output binary. No timing — precision only.
    """
    fn_lo  = fn_name.lower()
    fn_ext = fn_lo + "_"
    ct     = prec.c_type
    return f"""\
/* C precision benchmark driver for {fn_name} — generated by Fortran2Rust */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "f2c.h"

#define BENCH_N {N}

/* f2c signature: char* args are followed by their ftnlen lengths at the end */
extern int {fn_ext}(char *transa, char *transb,
                    integer *m, integer *n, integer *k,
                    {ct} *alpha, {ct} *a, integer *lda,
                    {ct} *b, integer *ldb, {ct} *beta,
                    {ct} *c__, integer *ldc,
                    ftnlen transa_len, ftnlen transb_len);

static {ct} prec_A[BENCH_N * BENCH_N];
static {ct} prec_B[BENCH_N * BENCH_N];
static {ct} prec_C[BENCH_N * BENCH_N];

static void read_bin(const char *path, {ct} *buf, size_t count) {{
    FILE *fp = fopen(path, "rb");
    if (!fp) {{ perror(path); exit(1); }}
    if (fread(buf, sizeof({ct}), count, fp) != count) {{
        fprintf(stderr, "Short read: %s\\n", path); exit(1);
    }}
    fclose(fp);
}}

int main(void) {{
    read_bin("dataset_{fn_lo}_precision_A.bin", prec_A, (size_t)BENCH_N * BENCH_N);
    read_bin("dataset_{fn_lo}_precision_B.bin", prec_B, (size_t)BENCH_N * BENCH_N);

    {ct} alpha, beta;
    memset(&alpha, 0, sizeof(alpha)); memset(&beta, 0, sizeof(beta));
    /* Set alpha=1, beta=0 in a type-safe way */
    *((double *)&alpha) = 1.0;
    integer m = BENCH_N, n = BENCH_N, k = BENCH_N;
    integer lda = BENCH_N, ldb = BENCH_N, ldc = BENCH_N;
    char transa = 'N', transb = 'N';

    memset(prec_C, 0, sizeof(prec_C));
    {fn_ext}(&transa, &transb, &m, &n, &k, &alpha, prec_A, &lda, prec_B, &ldb, &beta, prec_C, &ldc, 1, 1);

    FILE *out = fopen("bench_{fn_lo}_precision_output.bin", "wb");
    if (!out) {{ perror("bench_{fn_lo}_precision_output.bin"); exit(1); }}
    fwrite(prec_C, sizeof({ct}), (size_t)BENCH_N * BENCH_N, out);
    fclose(out);

    printf("C_PRECISION_DONE\\n");
    return 0;
}}
"""


def _make_c_axpy_driver(fn_name: str, N: int, prec: _Precision) -> str:
    fn_lo = fn_name.lower()
    fn_ext = fn_lo + "_"
    ct = prec.c_type
    return f"""\
/* C benchmark driver for {fn_name} (AXPY) — generated by Fortran2Rust */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "f2c.h"

#define BENCH_N {N}

extern int {fn_ext}(integer *n, {ct} *da, {ct} *dx, integer *incx, {ct} *dy, integer *incy);

static {ct} bench_X[BENCH_N];
static {ct} bench_Y0[BENCH_N];
static {ct} bench_Y[BENCH_N];

static void read_bin(const char *path, {ct} *buf, size_t count) {{
    FILE *fp = fopen(path, "rb");
    if (!fp) {{ perror(path); exit(1); }}
    if (fread(buf, sizeof({ct}), count, fp) != count) {{
        fprintf(stderr, "Short read: %s\\n", path); exit(1);
    }}
    fclose(fp);
}}

int main(void) {{
    {ct} params[2];
    read_bin("dataset_{fn_lo}_A.bin", bench_X, BENCH_N);
    read_bin("dataset_{fn_lo}_B.bin", bench_Y0, BENCH_N);
    read_bin("dataset_{fn_lo}_params.bin", params, 2);

    {ct} alpha = params[0];
    integer n = BENCH_N, incx = 1, incy = 1;

    struct timespec t1, t2;
    clock_gettime(CLOCK_MONOTONIC, &t1);
    for (int iter = 0; iter < 10; iter++) {{
        memcpy(bench_Y, bench_Y0, sizeof(bench_Y));
        {fn_ext}(&n, &alpha, bench_X, &incx, bench_Y, &incy);
    }}
    clock_gettime(CLOCK_MONOTONIC, &t2);
    double elapsed_ms = ((t2.tv_sec - t1.tv_sec) * 1e9 + (t2.tv_nsec - t1.tv_nsec)) / 1e6 / 10.0;

    memcpy(bench_Y, bench_Y0, sizeof(bench_Y));
    {fn_ext}(&n, &alpha, bench_X, &incx, bench_Y, &incy);

    FILE *out = fopen("bench_{fn_lo}_output.bin", "wb");
    if (!out) {{ perror("bench_{fn_lo}_output.bin"); exit(1); }}
    fwrite(bench_Y, sizeof({ct}), BENCH_N, out);
    fclose(out);

    printf("C_TIME_MS=%.4f\\n", elapsed_ms);
    return 0;
}}
"""


def _make_c_axpy_precision_driver(fn_name: str, N: int, prec: _Precision) -> str:
    fn_lo = fn_name.lower()
    fn_ext = fn_lo + "_"
    ct = prec.c_type
    return f"""\
/* C precision benchmark driver for {fn_name} (AXPY) — generated by Fortran2Rust */
#include <stdio.h>
#include <stdlib.h>
#include "f2c.h"

#define BENCH_N {N}

extern int {fn_ext}(integer *n, {ct} *da, {ct} *dx, integer *incx, {ct} *dy, integer *incy);

static {ct} prec_X[BENCH_N];
static {ct} prec_Y[BENCH_N];

static void read_bin(const char *path, {ct} *buf, size_t count) {{
    FILE *fp = fopen(path, "rb");
    if (!fp) {{ perror(path); exit(1); }}
    if (fread(buf, sizeof({ct}), count, fp) != count) {{
        fprintf(stderr, "Short read: %s\\n", path); exit(1);
    }}
    fclose(fp);
}}

int main(void) {{
    read_bin("dataset_{fn_lo}_precision_A.bin", prec_X, BENCH_N);
    read_bin("dataset_{fn_lo}_precision_B.bin", prec_Y, BENCH_N);

    {ct} alpha;
    memset(&alpha, 0, sizeof(alpha));
    *((double *)&alpha) = 1.0;
    integer n = BENCH_N, incx = 1, incy = 1;
    {fn_ext}(&n, &alpha, prec_X, &incx, prec_Y, &incy);

    FILE *out = fopen("bench_{fn_lo}_precision_output.bin", "wb");
    if (!out) {{ perror("bench_{fn_lo}_precision_output.bin"); exit(1); }}
    fwrite(prec_Y, sizeof({ct}), BENCH_N, out);
    fclose(out);

    printf("C_PRECISION_DONE\\n");
    return 0;
}}
"""


def _precision_to_rust_type(prec: _Precision) -> str:
    """Map _Precision to the Rust scalar type used in generated bench files."""
    return {"float64": "f64", "float32": "f32"}.get(prec.numpy_dtype, "f64")


# f2c Fortran I/O runtime stub bodies included in every generated Rust bench binary.
# xerbla_ (the BLAS error handler compiled into the library) calls these functions
# when Fortran WRITE / STOP is executed.  Providing no-op stubs satisfies the
# binary linker without needing libf2c at runtime; they are never reached during
# a correct benchmark run.
_RUST_F2C_STUBS = """\
// f2c Fortran I/O runtime stubs — satisfy linker when xerbla_ is in the lib.
#[no_mangle] pub extern "C" fn s_wsfe(_ci: *mut core::ffi::c_void) -> i32 { 0 }
#[no_mangle] pub extern "C" fn do_fio(_n: *mut i32, _s: *mut i8, _len: i32) -> i32 { 0 }
#[no_mangle] pub extern "C" fn e_wsfe() -> i32 { 0 }
#[no_mangle] pub extern "C" fn s_wslu(_ci: *mut core::ffi::c_void) -> i32 { 0 }
#[no_mangle] pub extern "C" fn e_wsle() -> i32 { 0 }
#[no_mangle] pub extern "C" fn do_lio(
    _ty: *mut i32, _n: *mut i32, _s: *mut i8, _len: i32,
) -> i32 { 0 }
#[no_mangle] pub unsafe extern "C" fn s_stop(_s: *mut i8, _n: i32) -> i32 {
    std::process::exit(1)
}
"""


def _make_rust_blas_driver(fn_name: str, N: int, prec: _Precision) -> str:
    """
    Generate a clean Rust benchmark binary for a real-precision BLAS dgemm-style
    function.  Includes f2c I/O runtime stubs so the binary links correctly
    when the lib contains xerbla_ (which calls f2c I/O from Fortran WRITE/STOP).

    Only real (float64 / float32) precision is supported; complex variants are
    skipped at the call site.
    """
    fn_lo  = fn_name.lower()
    fn_ext = fn_lo + "_"
    rtype  = _precision_to_rust_type(prec)
    stubs  = _RUST_F2C_STUBS
    return f"""\
/* Auto-generated Rust benchmark for {fn_name} — do NOT edit */
/* Generated by Fortran2Rust stage 2; replaces c2rust output in stages 6-8.  */
#![allow(non_snake_case, non_upper_case_globals, dead_code, static_mut_refs)]
use std::fs::File;
use std::io::{{Read, Write}};
use std::time::Instant;

{stubs}
// Reference the transpiled implementation via its Rust module path so that
// Cargo correctly links the rlib into this binary target.
use fortran2rust_output::{fn_lo}::{fn_ext};

const BENCH_N: usize = {N};
static mut BENCH_A: [{rtype}; BENCH_N * BENCH_N] = [0.0; BENCH_N * BENCH_N];
static mut BENCH_B: [{rtype}; BENCH_N * BENCH_N] = [0.0; BENCH_N * BENCH_N];
static mut BENCH_C: [{rtype}; BENCH_N * BENCH_N] = [0.0; BENCH_N * BENCH_N];

fn read_data(path: &str, buf: &mut [{rtype}]) {{
    let mut f = File::open(path).unwrap_or_else(|e| {{
        eprintln!("{{}}: {{}}", path, e); std::process::exit(1);
    }});
    let nb = buf.len() * std::mem::size_of::<{rtype}>();
    let mut raw = vec![0u8; nb];
    f.read_exact(&mut raw).unwrap_or_else(|_| {{
        eprintln!("Short read: {{}}", path); std::process::exit(1);
    }});
    for (dst, chunk) in buf.iter_mut().zip(raw.chunks_exact(std::mem::size_of::<{rtype}>())) {{
        *dst = {rtype}::from_ne_bytes(chunk.try_into().unwrap());
    }}
}}

fn main() {{
    unsafe {{
        read_data("dataset_{fn_lo}_A.bin", &mut BENCH_A);
        read_data("dataset_{fn_lo}_B.bin", &mut BENCH_B);
        let mut params: [{rtype}; 2] = [0.0; 2];
        read_data("dataset_{fn_lo}_params.bin", &mut params);
        let mut alpha = params[0];
        let mut beta  = params[1];
        let mut m:   i32 = BENCH_N as i32;
        let mut n:   i32 = BENCH_N as i32;
        let mut k:   i32 = BENCH_N as i32;
        let mut lda: i32 = BENCH_N as i32;
        let mut ldb: i32 = BENCH_N as i32;
        let mut ldc: i32 = BENCH_N as i32;
        let mut transa: i8 = b'N' as i8;
        let mut transb: i8 = b'N' as i8;

        // Warm up
        BENCH_C.fill(0.0);
        {fn_ext}(&mut transa, &mut transb,
                  &mut m, &mut n, &mut k,
                  &mut alpha, BENCH_A.as_mut_ptr(), &mut lda,
                  BENCH_B.as_mut_ptr(), &mut ldb,
                  &mut beta, BENCH_C.as_mut_ptr(), &mut ldc, 1, 1);
        BENCH_C.fill(0.0);

        // Benchmark — 10 iterations, report per-iteration average
        let start = Instant::now();
        for _ in 0..10 {{
            {fn_ext}(&mut transa, &mut transb,
                      &mut m, &mut n, &mut k,
                      &mut alpha, BENCH_A.as_mut_ptr(), &mut lda,
                      BENCH_B.as_mut_ptr(), &mut ldb,
                      &mut beta, BENCH_C.as_mut_ptr(), &mut ldc, 1, 1);
        }}
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0 / 10.0;

        // Write output binary for comparison with the Fortran baseline
        let out_bytes: &[u8] = std::slice::from_raw_parts(
            BENCH_C.as_ptr() as *const u8,
            BENCH_N * BENCH_N * std::mem::size_of::<{rtype}>(),
        );
        File::create("bench_{fn_lo}_output.bin")
            .and_then(|mut f| f.write_all(out_bytes))
            .expect("failed to write bench_{fn_lo}_output.bin");

        println!("RUST_TIME_MS={{:.4}}", elapsed_ms);
    }}
}}
"""


def _make_rust_blas_precision_driver(fn_name: str, N: int, prec: _Precision) -> str:
    """
    Generate a Rust near-cancellation precision benchmark for a real-precision
    BLAS dgemm-style function.  No timing — just produces the output binary
    for comparison with the Fortran precision baseline (seed 43 dataset).
    """
    fn_lo  = fn_name.lower()
    fn_ext = fn_lo + "_"
    rtype  = _precision_to_rust_type(prec)
    stubs  = _RUST_F2C_STUBS
    return f"""\
/* Auto-generated Rust precision benchmark for {fn_name} — do NOT edit */
/* Generated by Fortran2Rust stage 2; replaces c2rust output in stages 6-8.  */
#![allow(non_snake_case, non_upper_case_globals, dead_code, static_mut_refs)]
use std::fs::File;
use std::io::{{Read, Write}};

{stubs}
// Reference the transpiled implementation via its Rust module path so that
// Cargo correctly links the rlib into this binary target.
use fortran2rust_output::{fn_lo}::{fn_ext};

const BENCH_N: usize = {N};
static mut PREC_A: [{rtype}; BENCH_N * BENCH_N] = [0.0; BENCH_N * BENCH_N];
static mut PREC_B: [{rtype}; BENCH_N * BENCH_N] = [0.0; BENCH_N * BENCH_N];
static mut PREC_C: [{rtype}; BENCH_N * BENCH_N] = [0.0; BENCH_N * BENCH_N];

fn read_data(path: &str, buf: &mut [{rtype}]) {{
    let mut f = File::open(path).unwrap_or_else(|e| {{
        eprintln!("{{}}: {{}}", path, e); std::process::exit(1);
    }});
    let nb = buf.len() * std::mem::size_of::<{rtype}>();
    let mut raw = vec![0u8; nb];
    f.read_exact(&mut raw).unwrap_or_else(|_| {{
        eprintln!("Short read: {{}}", path); std::process::exit(1);
    }});
    for (dst, chunk) in buf.iter_mut().zip(raw.chunks_exact(std::mem::size_of::<{rtype}>())) {{
        *dst = {rtype}::from_ne_bytes(chunk.try_into().unwrap());
    }}
}}

fn main() {{
    unsafe {{
        read_data("dataset_{fn_lo}_precision_A.bin", &mut PREC_A);
        read_data("dataset_{fn_lo}_precision_B.bin", &mut PREC_B);
        let mut alpha: {rtype} = 1.0;
        let mut beta:  {rtype} = 0.0;
        let mut m:   i32 = BENCH_N as i32;
        let mut n:   i32 = BENCH_N as i32;
        let mut k:   i32 = BENCH_N as i32;
        let mut lda: i32 = BENCH_N as i32;
        let mut ldb: i32 = BENCH_N as i32;
        let mut ldc: i32 = BENCH_N as i32;
        let mut transa: i8 = b'N' as i8;
        let mut transb: i8 = b'N' as i8;

        PREC_C.fill(0.0);
        {fn_ext}(&mut transa, &mut transb,
                  &mut m, &mut n, &mut k,
                  &mut alpha, PREC_A.as_mut_ptr(), &mut lda,
                  PREC_B.as_mut_ptr(), &mut ldb,
                  &mut beta, PREC_C.as_mut_ptr(), &mut ldc, 1, 1);

        // Write output binary for comparison with the Fortran precision baseline
        let out_bytes: &[u8] = std::slice::from_raw_parts(
            PREC_C.as_ptr() as *const u8,
            BENCH_N * BENCH_N * std::mem::size_of::<{rtype}>(),
        );
        File::create("bench_{fn_lo}_precision_output.bin")
            .and_then(|mut f| f.write_all(out_bytes))
            .expect("failed to write bench_{fn_lo}_precision_output.bin");

        println!("RUST_PRECISION_DONE");
    }}
}}
"""


def _make_rust_axpy_driver(fn_name: str, N: int, prec: _Precision) -> str:
    fn_lo = fn_name.lower()
    fn_ext = fn_lo + "_"
    rtype = _precision_to_rust_type(prec)
    stubs = _RUST_F2C_STUBS
    return f"""\
/* Auto-generated Rust benchmark for {fn_name} (AXPY) — do NOT edit */
#![allow(non_snake_case, non_upper_case_globals, dead_code, static_mut_refs)]
use std::fs::File;
use std::io::{{Read, Write}};
use std::time::Instant;

{stubs}
use fortran2rust_output::{fn_lo}::{fn_ext};

const BENCH_N: usize = {N};
static mut BENCH_X: [{rtype}; BENCH_N] = [0.0; BENCH_N];
static mut BENCH_Y: [{rtype}; BENCH_N] = [0.0; BENCH_N];

fn read_data(path: &str, buf: &mut [{rtype}]) {{
    let mut f = File::open(path).unwrap_or_else(|e| {{
        eprintln!("{{}}: {{}}", path, e); std::process::exit(1);
    }});
    let nb = buf.len() * std::mem::size_of::<{rtype}>();
    let mut raw = vec![0u8; nb];
    f.read_exact(&mut raw).unwrap_or_else(|_| {{
        eprintln!("Short read: {{}}", path); std::process::exit(1);
    }});
    for (dst, chunk) in buf.iter_mut().zip(raw.chunks_exact(std::mem::size_of::<{rtype}>())) {{
        *dst = {rtype}::from_ne_bytes(chunk.try_into().unwrap());
    }}
}}

fn main() {{
    unsafe {{
        read_data("dataset_{fn_lo}_A.bin", &mut BENCH_X);
        read_data("dataset_{fn_lo}_B.bin", &mut BENCH_Y);
        let mut params: [{rtype}; 2] = [0.0; 2];
        read_data("dataset_{fn_lo}_params.bin", &mut params);
        let mut alpha = params[0];

        let mut n: i32 = BENCH_N as i32;
        let mut incx: i32 = 1;
        let mut incy: i32 = 1;

        let start = Instant::now();
        for _ in 0..10 {{
            {fn_ext}(&mut n, &mut alpha, BENCH_X.as_mut_ptr(), &mut incx, BENCH_Y.as_mut_ptr(), &mut incy);
        }}
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0 / 10.0;

        let out_bytes: &[u8] = std::slice::from_raw_parts(
            BENCH_Y.as_ptr() as *const u8,
            BENCH_N * std::mem::size_of::<{rtype}>(),
        );
        File::create("bench_{fn_lo}_output.bin")
            .and_then(|mut f| f.write_all(out_bytes))
            .expect("failed to write bench_{fn_lo}_output.bin");

        println!("RUST_TIME_MS={{:.4}}", elapsed_ms);
    }}
}}
"""


def _make_rust_axpy_precision_driver(fn_name: str, N: int, prec: _Precision) -> str:
    fn_lo = fn_name.lower()
    fn_ext = fn_lo + "_"
    rtype = _precision_to_rust_type(prec)
    stubs = _RUST_F2C_STUBS
    return f"""\
/* Auto-generated Rust precision benchmark for {fn_name} (AXPY) — do NOT edit */
#![allow(non_snake_case, non_upper_case_globals, dead_code, static_mut_refs)]
use std::fs::File;
use std::io::{{Read, Write}};

{stubs}
use fortran2rust_output::{fn_lo}::{fn_ext};

const BENCH_N: usize = {N};
static mut PREC_X: [{rtype}; BENCH_N] = [0.0; BENCH_N];
static mut PREC_Y: [{rtype}; BENCH_N] = [0.0; BENCH_N];

fn read_data(path: &str, buf: &mut [{rtype}]) {{
    let mut f = File::open(path).unwrap_or_else(|e| {{
        eprintln!("{{}}: {{}}", path, e); std::process::exit(1);
    }});
    let nb = buf.len() * std::mem::size_of::<{rtype}>();
    let mut raw = vec![0u8; nb];
    f.read_exact(&mut raw).unwrap_or_else(|_| {{
        eprintln!("Short read: {{}}", path); std::process::exit(1);
    }});
    for (dst, chunk) in buf.iter_mut().zip(raw.chunks_exact(std::mem::size_of::<{rtype}>())) {{
        *dst = {rtype}::from_ne_bytes(chunk.try_into().unwrap());
    }}
}}

fn main() {{
    unsafe {{
        read_data("dataset_{fn_lo}_precision_A.bin", &mut PREC_X);
        read_data("dataset_{fn_lo}_precision_B.bin", &mut PREC_Y);
        let mut alpha: {rtype} = 1.0;
        let mut n: i32 = BENCH_N as i32;
        let mut incx: i32 = 1;
        let mut incy: i32 = 1;

        {fn_ext}(&mut n, &mut alpha, PREC_X.as_mut_ptr(), &mut incx, PREC_Y.as_mut_ptr(), &mut incy);

        let out_bytes: &[u8] = std::slice::from_raw_parts(
            PREC_Y.as_ptr() as *const u8,
            BENCH_N * std::mem::size_of::<{rtype}>(),
        );
        File::create("bench_{fn_lo}_precision_output.bin")
            .and_then(|mut f| f.write_all(out_bytes))
            .expect("failed to write bench_{fn_lo}_precision_output.bin");

        println!("RUST_PRECISION_DONE");
    }}
}}
"""


def _make_c_generic_driver(fn_name: str, N: int,
                           sig: dict | None, prec: _Precision) -> str:
    """
    Generic C benchmark driver.  Uses the dominant precision from the parsed
    signature for array/scalar declarations.  The LLM (stage 4) will fix the
    actual function call if it requires further adjustments.
    """
    fn_lo  = fn_name.lower()
    fn_ext = fn_lo + "_"
    ct     = prec.c_type
    vector_mode = _is_vector_blas(fn_name)
    array_count_expr = "BENCH_N" if vector_mode else "(size_t)BENCH_N * BENCH_N"
    fn_proto_decl = ""

    # Build typed variable declarations and the function call from the signature
    if sig is not None:
        params     = sig["params"]
        char_params: list[str] = []
        decls:      list[str] = []
        inits:      list[str] = []
        reads:      list[str] = []
        call_args:  list[str] = []
        proto_args: list[str] = []
        scalar_outputs: list[tuple[str, str]] = []
        fn_result_output: tuple[str, str] | None = None
        arr_idx = 0

        for p in params:
            pname  = p["name"].lower()
            ptype  = p["type"]
            p_prec = _FORTRAN_TYPE_TO_PREC.get(ptype)
            cname  = f"bench_{pname}"

            if "CHARACTER" in ptype:
                decls.append(f"    char {cname} = 'N';")
                call_args.append(f"&{cname}")
                proto_args.append(f"char *{cname}")
                char_params.append(cname)

            elif "INTEGER" in ptype:
                decls.append(f"    integer {cname};")
                pu = pname.upper()
                if pu.startswith("LD") or pu in ("LDA", "LDB", "LDC", "LDQ", "LDT"):
                    inits.append(f"    {cname} = BENCH_N;")
                elif pu.startswith("INC") or pu in ("INCX", "INCY"):
                    inits.append(f"    {cname} = 1;")
                else:
                    inits.append(f"    {cname} = BENCH_N;")
                call_args.append(f"&{cname}")
                proto_args.append(f"integer *{cname}")

            elif "LOGICAL" in ptype:
                decls.append(f"    logical {cname};")
                inits.append(f"    {cname} = 1;")
                call_args.append(f"&{cname}")
                proto_args.append(f"logical *{cname}")

            elif p_prec is not None:
                pctype = p_prec.c_type
                if p["is_array"]:
                    ds = "A" if arr_idx == 0 else "B"
                    if vector_mode:
                        decls.append(f"    static {pctype} {cname}[BENCH_N];")
                    else:
                        decls.append(f"    static {pctype} {cname}[BENCH_N * BENCH_N];")
                    reads.append(
                        f'    read_bin("dataset_{fn_lo}_{ds}.bin",'
                        f" {cname}, {array_count_expr});"
                    )
                    arr_idx += 1
                    call_args.append(cname)
                    proto_args.append(f"{pctype} *{cname}")
                else:
                    pu = pname.upper()
                    decls.append(f"    {pctype} {cname};")
                    inits.append(f"    memset(&{cname}, 0, sizeof({cname}));")
                    if pu in ("ALPHA", "SCALE", "DA", "SA", "ZA", "CA", "A", "W"):
                        inits.append(f"    *((double *)&{cname}) = 1.0;  /* alpha = 1 */")
                    scalar_outputs.append((cname, pctype))
                    call_args.append(f"&{cname}")
                    proto_args.append(f"{pctype} *{cname}")
            else:
                decls.append(f"    integer {cname}; /* TODO: {ptype} */")
                inits.append(f"    {cname} = 0;")
                call_args.append(f"&{cname}")
                proto_args.append(f"integer *{cname}")

        # Append ftnlen=1 for each CHARACTER parameter
        ftnlen_args = ["1"] * len(char_params)
        proto_args.extend(f"ftnlen {name}_len" for name in char_params)

        if sig.get("is_function"):
            return_type = (sig.get("return_type") or "").upper()
            return_prec = _FORTRAN_TYPE_TO_PREC.get(return_type)
            if return_prec is not None:
                fn_ret_ctype = return_prec.c_type
            elif "INTEGER" in return_type:
                fn_ret_ctype = "integer"
            else:
                fn_ret_ctype = "int"

            ret_var = "bench_result"
            decls.append(f"    {fn_ret_ctype} {ret_var};")
            all_args = call_args + ftnlen_args
            call_line = f"    {ret_var} = {fn_ext}({', '.join(all_args)});"
            fn_result_output = (ret_var, fn_ret_ctype)
            fn_proto_decl = f"extern {fn_ret_ctype} {fn_ext}({', '.join(proto_args)});"
        else:
            all_args = call_args + ftnlen_args
            call_line = f"    {fn_ext}({', '.join(all_args)});"
            fn_proto_decl = f"extern int {fn_ext}({', '.join(proto_args)});"

        # Determine output array (last float array in params)
        float_arrays = [
            p["name"].lower()
            for p in params
            if p["is_array"] and p["type"] in _FORTRAN_TYPE_TO_PREC
        ]
        if fn_result_output is not None:
            out_arr_expr = f"&{fn_result_output[0]}"
            out_ctype = fn_result_output[1]
            out_count = "1"
        elif float_arrays:
            out_arr_expr = f"bench_{float_arrays[-1]}"
            out_ctype = ct
            out_count = array_count_expr
        elif scalar_outputs:
            out_arr_expr = f"&{scalar_outputs[-1][0]}"
            out_ctype = scalar_outputs[-1][1]
            out_count = "1"
        else:
            out_arr_expr = "bench_A"
            out_ctype = ct
            out_count = "(size_t)BENCH_N * BENCH_N"

        decl_block = "\n".join(decls)
        init_block = "\n".join(inits)
        read_block = "\n".join(reads)
    else:
        decl_block = (
            f"    static {ct} bench_A[BENCH_N * BENCH_N];\n"
            f"    static {ct} bench_B[BENCH_N * BENCH_N];\n"
            f"    static {ct} bench_C[BENCH_N * BENCH_N];"
        )
        init_block  = ""
        read_block  = (
            f'    read_bin("dataset_{fn_lo}_A.bin", bench_A, {array_count_expr});\n'
            f'    read_bin("dataset_{fn_lo}_B.bin", bench_B, {array_count_expr});'
        )
        call_line   = f"    /* TODO: {fn_ext}(...); */"
        out_arr_expr = "bench_C"
        out_ctype   = ct
        out_count   = array_count_expr

    return f"""\
/* C benchmark driver for {fn_name} — generated by Fortran2Rust */
/* TODO: verify function signature and call against the f2c output */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "f2c.h"

#define BENCH_N {N}

{fn_proto_decl}

static void read_bin(const char *path, {ct} *buf, size_t count) {{
    FILE *fp = fopen(path, "rb");
    if (!fp) {{ perror(path); exit(1); }}
    if (fread(buf, sizeof({ct}), count, fp) != count) {{
        fprintf(stderr, "Short read: %s\\n", path); exit(1);
    }}
    fclose(fp);
}}

int main(void) {{
{decl_block}
{read_block}
{init_block}

    struct timespec t1, t2;
    clock_gettime(CLOCK_MONOTONIC, &t1);
{call_line}
    clock_gettime(CLOCK_MONOTONIC, &t2);
    double elapsed_ms = ((t2.tv_sec - t1.tv_sec) * 1e9 + (t2.tv_nsec - t1.tv_nsec)) / 1e6;

    FILE *out = fopen("bench_{fn_lo}_output.bin", "wb");
    if (!out) {{ perror("bench_{fn_lo}_output.bin"); exit(1); }}
    fwrite({out_arr_expr}, sizeof({out_ctype}), {out_count}, out);
    fclose(out);

    printf("C_TIME_MS=%.4f\\n", elapsed_ms);
    return 0;
}}
"""

# ── Stage-2 report templates ───────────────────────────────────────────────────

_S2_REPORT_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Stage 2 — Fortran Benchmark Results</title>
<style>
:root {
  --primary: #007AC3;
  --navy: #1B3C6E;
  --bg: #F4F7FB;
  --success: #00A550;
  --danger: #E31937;
  --text: #1A1A1A;
}
body { font-family: Inter, system-ui, sans-serif; background: var(--bg); color: var(--text); margin: 0; }
header { background: var(--navy); color: white; padding: 1.5rem 2rem; }
header h1 { margin: 0 0 0.25rem 0; font-size: 1.75rem; }
header p { margin: 0; opacity: 0.8; font-size: 0.9rem; }
.card { background: white; border-radius: 8px; padding: 1.5rem; margin: 1rem 2rem; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
.card h2 { margin-top: 0; color: var(--navy); border-bottom: 2px solid var(--primary); padding-bottom: 0.5rem; }
table { width: 100%; border-collapse: collapse; }
th { background: var(--navy); color: white; padding: 0.75rem; text-align: left; }
td { padding: 0.6rem 0.75rem; border-bottom: 1px solid #e0e8f0; }
tr:last-child td { border-bottom: none; }
.summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 1rem; }
.metric { text-align: center; padding: 1rem; background: var(--bg); border-radius: 6px; }
.metric .value { font-size: 2rem; font-weight: bold; color: var(--primary); }
.metric .label { font-size: 0.8rem; color: #666; margin-top: 0.25rem; }
.status-pass { background: #e8f7ef; color: var(--success); padding: 0.2rem 0.6rem; border-radius: 4px; font-weight: bold; }
.status-fail { background: #fde8eb; color: var(--danger); padding: 0.2rem 0.6rem; border-radius: 4px; font-weight: bold; }
.na { color: #aaa; }
</style>
</head>
<body>
<header>
  <h1>📊 Stage 2 — Fortran Benchmark Results</h1>
  <p>{{ timestamp }}</p>
</header>

<div class="card">
  <h2>Summary</h2>
  <div class="summary-grid">
    <div class="metric"><div class="value">{{ summary.total }}</div><div class="label">Entry Points</div></div>
    <div class="metric"><div class="value">{{ summary.compiled }}</div><div class="label">Compiled</div></div>
    <div class="metric"><div class="value">{{ summary.ran }}</div><div class="label">Ran Successfully</div></div>
    <div class="metric"><div class="value">{{ summary.precision_ok }}</div><div class="label">Precision Output Valid</div></div>
  </div>
</div>

<div class="card">
  <h2>Entry-Point Results</h2>
  <table>
    <tr>
      <th>Function</th>
      <th>Calibrated N</th>
      <th>Fortran Time (ms)</th>
      <th>Compiled</th>
      <th>Ran</th>
      <th>Precision Output</th>
    </tr>
    {% for row in rows %}
    <tr>
      <td><strong>{{ row.function }}</strong></td>
      <td>{{ row.n if row.n is not none else '<span class="na">—</span>' }}</td>
      <td>{{ "%.3f" | format(row.time_ms) if row.time_ms is not none else '<span class="na">—</span>' }}</td>
      <td>{% if row.compile_ok %}<span class="status-pass">YES</span>{% else %}<span class="status-fail">NO</span>{% endif %}</td>
      <td>{% if row.run_ok %}<span class="status-pass">YES</span>{% else %}<span class="status-fail">NO</span>{% endif %}</td>
      <td>{% if row.precision_ok %}<span class="status-pass">VALID</span>{% else %}<span class="status-fail">MISSING</span>{% endif %}</td>
    </tr>
    {% endfor %}
  </table>
</div>

</body>
</html>"""

_S2_REPORT_MD = """# Stage 2 — Fortran Benchmark Results

**Generated:** {{ timestamp }}

## Summary

| Metric | Value |
|--------|-------|
| Entry Points | {{ summary.total }} |
| Compiled | {{ summary.compiled }} |
| Ran Successfully | {{ summary.ran }} |
| Precision Output Valid | {{ summary.precision_ok }} |

## Entry-Point Results

| Function | Calibrated N | Fortran Time (ms) | Compiled | Ran | Precision Output |
|----------|-------------|-------------------|----------|-----|-----------------|
{% for row in rows %}| {{ row.function }} | {{ row.n if row.n is not none else "—" }} | {{ "%.3f" | format(row.time_ms) if row.time_ms is not none else "—" }} | {{ "YES" if row.compile_ok else "NO" }} | {{ "YES" if row.run_ok else "NO" }} | {{ "VALID" if row.precision_ok else "MISSING" }} |
{% endfor %}
"""


def _generate_benchmark_report(
    output_dir: Path,
    benchmarks: dict,
    ep_n: dict[str, int],
    all_datasets: dict,
    entry_points: list[str],
) -> None:
    """Write benchmark_results.{html,md} into output_dir."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    rows = []
    for ep in entry_points:
        info = benchmarks.get(ep, {})
        precision_ok = bool(all_datasets.get(ep + "_precision", {}).get("expected"))
        rows.append({
            "function": ep.lower(),
            "n": ep_n.get(ep),
            "time_ms": info.get("time_ms"),
            "compile_ok": bool(info.get("compile_ok")),
            "run_ok": bool(info.get("run_ok")),
            "precision_ok": precision_ok,
        })

    summary = {
        "total": len(rows),
        "compiled": sum(1 for r in rows if r["compile_ok"]),
        "ran": sum(1 for r in rows if r["run_ok"]),
        "precision_ok": sum(1 for r in rows if r["precision_ok"]),
    }

    ctx = {"timestamp": timestamp, "summary": summary, "rows": rows}
    env = Environment(loader=BaseLoader())
    (output_dir / "benchmark_results.html").write_text(env.from_string(_S2_REPORT_HTML).render(**ctx))
    (output_dir / "benchmark_results.md").write_text(env.from_string(_S2_REPORT_MD).render(**ctx))


def generate_benchmarks(
    source_dir: Path,
    entry_points: list[str],
    dep_files: list[Path],
    output_dir: Path,
    call_graph: dict[str, list[str]] | None = None,
    max_parallel: int = 2,
    matrix_n_max: int = 768,
    timing_max_runs: int = 12,
    dataset_reuse_every: int = 3,
    status_fn=None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    log = make_stage_logger(output_dir)
    log.info(f"generate_benchmarks: entry_points={entry_points}, N_default={N_DEFAULT}")

    # ── Phase 1: datasets (always first, unconditionally) ────────────────────
    # Datasets are the shared ground truth for Fortran, C, and Rust.
    # They are generated before any compilation so they're always present.
    if status_fn:
        status_fn(f"Generating shared input datasets for {len(entry_points)} function(s)…")
    log.info(f"Generating input datasets for {len(entry_points)} function(s)")

    # Pre-compute precision and parsed signatures for every entry point so
    # both dataset generation and driver generation use consistent types.
    ep_prec: dict[str, _Precision] = {}
    ep_sig:  dict[str, dict | None] = {}
    ep_n:    dict[str, int] = {}
    ep_timing_runs: dict[str, int] = {}
    for ep in entry_points:
        ep_upper = ep.upper()
        if ep_upper in KNOWN_BLAS:
            prec = _blas_precision(ep)
            sig  = None
        else:
            if status_fn:
                status_fn(f"Parsing Fortran signature for {ep}…")
            sig  = _parse_fn_signature(ep, source_dir)
            prec = _dominant_precision(sig)
            if sig is None:
                log.warning(f"  Could not parse signature for {ep}; defaulting to DOUBLE PRECISION")
            else:
                log.info(f"  Parsed signature for {ep}: {prec.fortran_type}")
        ep_prec[ep] = prec
        ep_sig[ep]  = sig

    # Calibrate all entry points in parallel — each uses its own file paths so
    # there are no write conflicts.  Python's logging module is thread-safe.
    # status_fn updates are serialised through a lock to avoid interleaved text.
    _status_lock = threading.Lock()

    def _safe_status(msg: str) -> None:
        if status_fn:
            with _status_lock:
                status_fn(msg)

    max_workers = min(len(entry_points), max(1, max_parallel))
    if status_fn:
        status_fn(f"Calibrating benchmark sizes for {len(entry_points)} function(s) "
                  f"(up to {max_workers} parallel workers)…")

    def _calibrate_one(ep: str) -> tuple[str, int, int]:
        n, timing_runs = _calibrate_benchmark_size(
            fn_name=ep,
            source_dir=source_dir,
            dep_files=dep_files,
            output_dir=output_dir,
            call_graph=call_graph,
            sig=ep_sig[ep],
            prec=ep_prec[ep],
            log=log,
            matrix_n_max=matrix_n_max,
            timing_max_runs=timing_max_runs,
            dataset_reuse_every=dataset_reuse_every,
            status_fn=_safe_status,
        )
        log.info(f"  Selected benchmark size for {ep}: N={n}, timing_runs={timing_runs}")
        return ep, n, timing_runs

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_calibrate_one, ep): ep for ep in entry_points}
        for fut in as_completed(futures):
            ep, n, timing_runs = fut.result()   # re-raises any exception from the worker
            ep_n[ep] = n
            ep_timing_runs[ep] = timing_runs

    all_datasets: dict[str, dict] = {}
    for ep in entry_points:
        prec = ep_prec[ep]
        N = ep_n[ep]
        dataset = generate_dataset(ep, N, output_dir, prec=prec)
        all_datasets[ep] = {k: str(v) for k, v in dataset.items()}
        log.info(f"  dataset for {ep} ({prec.numpy_dtype}, N={N}): {list(dataset.keys())}")
        prec_dataset = generate_precision_dataset(ep, N, output_dir, prec=prec)
        all_datasets[ep + "_precision"] = {k: str(v) for k, v in prec_dataset.items()}
    (output_dir / "datasets.json").write_text(json.dumps(all_datasets, indent=2))

    # ── Phase 2: benchmark drivers (Fortran + C + Rust) ──────────────────────
    benchmarks: dict[str, dict] = {}
    bench_files:    list[str] = []   # Fortran .f drivers (gfortran, modern syntax)
    bench_c_files:  list[str] = []   # C drivers (written directly, no f2c needed)
    bench_rs_files: list[str] = []   # Rust drivers (clean, no f2c contamination)

    for ep in entry_points:
        ep_upper = ep.upper()
        ep_lower = ep.lower()
        N = ep_n[ep]
        prec = ep_prec[ep]
        sig  = ep_sig[ep]
        dataset_paths = {k: Path(v) for k, v in all_datasets[ep].items()}

        driver_ext = ".f90"
        precision_ext = ".f90"

        if ep_upper in KNOWN_GEMM:
            driver_src      = _make_dgemm_driver(ep, N, prec)
            precision_src   = _make_dgemm_precision_driver(ep, N, prec)
            c_driver_src    = _make_c_blas_driver(ep, N, prec)
            c_precision_src = _make_c_blas_precision_driver(ep, N, prec)
        elif ep_upper.endswith("AXPY") and not prec.is_complex:
            driver_src      = _make_axpy_driver(ep, N, prec)
            precision_src   = _make_axpy_precision_driver(ep, N, prec)
            c_driver_src    = _make_c_axpy_driver(ep, N, prec)
            c_precision_src = _make_c_axpy_precision_driver(ep, N, prec)
            driver_ext = ".f90"
            precision_ext = ".f90"
        else:
            driver_src      = _make_generic_driver(ep, N, sig, prec)
            precision_src   = _make_generic_precision_driver(ep, N, sig, prec)
            c_driver_src    = _make_c_generic_driver(ep, N, sig, prec)
            c_precision_src = _make_c_generic_driver(ep, N, sig, prec).replace(
                f"dataset_{ep_lower}_A.bin", f"dataset_{ep_lower}_precision_A.bin"
            ).replace(
                f"dataset_{ep_lower}_B.bin", f"dataset_{ep_lower}_precision_B.bin"
            ).replace(
                f"bench_{ep_lower}_output.bin", f"bench_{ep_lower}_precision_output.bin"
            )

        driver_file = output_dir / f"bench_{ep_lower}{driver_ext}"
        driver_file.write_text(driver_src)
        bench_files.append(str(driver_file))

        # Write C driver directly — no f2c conversion needed
        c_driver_file = output_dir / f"bench_{ep_lower}.c"
        c_driver_file.write_text(c_driver_src)
        bench_c_files.append(str(c_driver_file))

        if c_precision_src:
            c_prec_file = output_dir / f"bench_{ep_lower}_precision.c"
            c_prec_file.write_text(c_precision_src)
            bench_c_files.append(str(c_prec_file))

        if precision_src:
            prec_file = output_dir / f"bench_{ep_lower}_precision{precision_ext}"
            prec_file.write_text(precision_src)
            bench_files.append(str(prec_file))

        # Write Rust bench drivers for real-precision BLAS functions.
        # These are used by stages 6-8 instead of the c2rust-generated bench files,
        # because c2rust bench files reference f2c I/O symbols (s_wsfe, do_fio, …)
        # that are undefined at binary link time.  The generated files include
        # no-op stubs for those symbols so the bench binary links cleanly.
        if ep_upper in KNOWN_GEMM and not prec.is_complex:
            rs_file = output_dir / f"bench_{ep_lower}.rs"
            rs_file.write_text(_make_rust_blas_driver(ep, N, prec))
            bench_rs_files.append(str(rs_file))
            log.info(f"  Wrote Rust bench driver: {rs_file.name}")

            rs_prec_file = output_dir / f"bench_{ep_lower}_precision.rs"
            rs_prec_file.write_text(_make_rust_blas_precision_driver(ep, N, prec))
            bench_rs_files.append(str(rs_prec_file))
            log.info(f"  Wrote Rust precision bench driver: {rs_prec_file.name}")
        elif ep_upper.endswith("AXPY") and not prec.is_complex:
            rs_file = output_dir / f"bench_{ep_lower}.rs"
            rs_file.write_text(_make_rust_axpy_driver(ep, N, prec))
            bench_rs_files.append(str(rs_file))
            log.info(f"  Wrote Rust bench driver: {rs_file.name}")

            rs_prec_file = output_dir / f"bench_{ep_lower}_precision.rs"
            rs_prec_file.write_text(_make_rust_axpy_precision_driver(ep, N, prec))
            bench_rs_files.append(str(rs_prec_file))
            log.info(f"  Wrote Rust precision bench driver: {rs_prec_file.name}")

        # Compile and run Fortran benchmark to produce the reference output.
        fortran_deps = _resolve_fortran_deps(source_dir, dep_files, call_graph, ep_upper)
        exe_path = (output_dir / f"bench_{ep_lower}").resolve()
        compile_cmd = (
            [
                "gfortran", "-O2",
                "-ffunction-sections", "-fdata-sections",
                str(driver_file.resolve()),
            ]
            + fortran_deps
            + ["-Wl,--gc-sections", "-Wl,--allow-multiple-definition", "-lm", "-o", str(exe_path)]
        )

        bench_info: dict = {
            "driver_file": str(driver_file),
            "n": N,
            "timing_runs": ep_timing_runs.get(ep, 1),
            "numpy_dtype": prec.numpy_dtype,
            "dataset": {k: str(v) for k, v in dataset_paths.items()},
            "compile_cmd": compile_cmd,
            "compile_ok": False,
            "compile_stdout": "",
            "compile_stderr": "",
            "run_ok": False,
            "run_exit_code": None,
            "run_stdout": "",
            "run_stderr": "",
            "output_ok": False,
            "time_ms": None,
            "output_file": None,
        }

        try:
            if status_fn:
                status_fn(f"Compiling Fortran benchmark for {ep}…")
            log.info(f"Compiling Fortran benchmark for {ep}: {' '.join(compile_cmd)}")
            result = subprocess.run(
                compile_cmd, capture_output=True, text=True,
                timeout=120,
            )
            bench_info["compile_stdout"] = result.stdout
            bench_info["compile_stderr"] = result.stderr
            compile_log = output_dir / f"gfortran_{ep_lower}.log"
            compile_log.write_text(
                f"=== COMMAND ===\n{' '.join(compile_cmd)}\n\n"
                f"=== STDOUT ===\n{result.stdout}\n"
                f"=== STDERR ===\n{result.stderr}\n"
                f"=== EXIT CODE: {result.returncode} ===\n"
            )
            if result.returncode == 0:
                bench_info["compile_ok"] = True
                log.info(f"  gfortran OK for {ep}")
                if status_fn:
                    status_fn(f"Running Fortran baseline for {ep}…")
                log.info(f"  Running Fortran baseline for {ep}")
                run_times: list[float] = []
                baseline_runs = max(1, ep_timing_runs.get(ep, 1))
                run_result = None
                for run_idx in range(baseline_runs):
                    if run_idx == 0 or run_idx % max(1, dataset_reuse_every) == 0:
                        seed = 42 + (run_idx // max(1, dataset_reuse_every))
                        rotated = generate_dataset(ep, N, output_dir, prec=prec, seed=seed)
                        all_datasets[ep] = {k: str(v) for k, v in rotated.items()}
                        bench_info["dataset"] = {k: str(v) for k, v in rotated.items()}

                    run_result = subprocess.run(
                        [str(exe_path)],
                        capture_output=True, text=True,
                        cwd=str(output_dir.resolve()), timeout=300,
                    )
                    bench_info["run_exit_code"] = run_result.returncode
                    bench_info["run_ok"] = run_result.returncode == 0
                    bench_info["run_stdout"] = run_result.stdout
                    bench_info["run_stderr"] = run_result.stderr
                    with open(compile_log, "a") as fh:
                        fh.write(
                            f"\n=== RUN {run_idx + 1}/{baseline_runs} STDOUT ===\n{run_result.stdout}\n"
                            f"=== RUN {run_idx + 1}/{baseline_runs} STDERR ===\n{run_result.stderr}\n"
                            f"=== RUN {run_idx + 1}/{baseline_runs} EXIT CODE: {run_result.returncode} ===\n"
                        )
                    if run_result.returncode != 0:
                        raise BenchmarkRuntimeError(
                            ep,
                            f"Fortran benchmark executable failed (exit {run_result.returncode}). "
                            f"See {compile_log.name}",
                        )

                    match = re.search(r"FORTRAN_TIME_MS=\s*([\d.]+)", run_result.stdout)
                    if match:
                        run_times.append(float(match.group(1)))

                if run_times:
                    bench_info["time_ms"] = sum(run_times) / len(run_times)
                    log.info(
                        "  Fortran baseline time: %.4f ms (%d run(s), dataset reuse every %d)",
                        bench_info["time_ms"],
                        baseline_runs,
                        max(1, dataset_reuse_every),
                    )

                bin_file = output_dir / f"bench_{ep_lower}_output.bin"
                if bin_file.exists():
                    bench_info["output_file"] = str(bin_file)
                    bench_info["output_ok"] = True
                    expected_file = output_dir / f"dataset_{ep_lower}_expected.bin"
                    shutil.copy2(bin_file, expected_file)
                    bench_info["dataset"]["expected"] = str(expected_file)
                    all_datasets[ep]["expected"] = str(expected_file)
                else:
                    raise BenchmarkRuntimeError(
                        ep,
                        f"Fortran benchmark did not produce output file {bin_file.name}",
                    )
                if not bench_info["run_ok"]:
                    log.warning(f"  Fortran baseline run FAILED for {ep} (exit {run_result.returncode})")
                elif not bench_info["output_ok"]:
                    log.warning(f"  Fortran baseline for {ep} produced no output binary")
            else:
                log.warning(f"  gfortran FAILED for {ep} (exit {result.returncode})")
                log.warning(f"  stderr: {result.stderr[:500]}")
                raise CompilationError(
                    "Fortran",
                    f"gfortran compile failed for {ep} (exit {result.returncode}). See {compile_log.name}",
                )
        except subprocess.TimeoutExpired:
            bench_info["compile_stderr"] = "Timeout"
            log.warning(f"  gfortran timed out for {ep}")
            raise CompilationError("Fortran", f"gfortran timed out for {ep}")
        except Exception as e:
            if isinstance(e, (CompilationError, BenchmarkRuntimeError)):
                raise
            bench_info["compile_stderr"] = str(e)
            log.warning(f"  gfortran exception for {ep}: {e}")
            raise CompilationError("Fortran", f"gfortran exception for {ep}: {e}")

        benchmarks[ep] = bench_info

        # Compile and run precision Fortran driver for every entry point when present.
        # Produces bench_{ep_lower}_precision_output.bin used as baseline in Stage 4.
        prec_driver_file = output_dir / f"bench_{ep_lower}_precision.f"
        if not prec_driver_file.exists():
            prec_driver_file = output_dir / f"bench_{ep_lower}_precision.f90"
        if prec_driver_file.exists():
            prec_exe = (output_dir / f"bench_{ep_lower}_precision").resolve()
            prec_compile_cmd = (
                [
                    "gfortran", "-O2",
                    "-ffunction-sections", "-fdata-sections",
                    str(prec_driver_file.resolve()),
                ]
                + fortran_deps
                + ["-Wl,--gc-sections", "-Wl,--allow-multiple-definition", "-lm", "-o", str(prec_exe)]
            )
            prec_log = output_dir / f"gfortran_{ep_lower}_precision.log"
            try:
                if status_fn:
                    status_fn(f"Compiling Fortran precision benchmark for {ep}…")
                log.info(f"Compiling Fortran precision benchmark for {ep}")
                prec_result = subprocess.run(
                    prec_compile_cmd, capture_output=True, text=True, timeout=120,
                )
                prec_log.write_text(
                    f"=== COMMAND ===\n{' '.join(prec_compile_cmd)}\n\n"
                    f"=== STDOUT ===\n{prec_result.stdout}\n"
                    f"=== STDERR ===\n{prec_result.stderr}\n"
                    f"=== EXIT CODE: {prec_result.returncode} ===\n"
                )
                if prec_result.returncode == 0:
                    log.info(f"  gfortran precision OK for {ep}")
                    prec_run = subprocess.run(
                        [str(prec_exe)], capture_output=True, text=True,
                        cwd=str(output_dir.resolve()), timeout=300,
                    )
                    with open(prec_log, "a") as fh:
                        fh.write(
                            f"\n=== RUN STDOUT ===\n{prec_run.stdout}\n"
                            f"=== RUN STDERR ===\n{prec_run.stderr}\n"
                            f"=== RUN EXIT CODE: {prec_run.returncode} ===\n"
                        )
                    log.info(f"  Fortran precision run exit {prec_run.returncode}")
                    prec_bin_file = output_dir / f"bench_{ep_lower}_precision_output.bin"
                    if prec_run.returncode == 0 and prec_bin_file.exists():
                        prec_expected_file = output_dir / f"dataset_{ep_lower}_precision_expected.bin"
                        shutil.copy2(prec_bin_file, prec_expected_file)
                        all_datasets[ep + "_precision"]["expected"] = str(prec_expected_file)
                else:
                    log.warning(f"  gfortran precision FAILED for {ep}: {prec_result.stderr[:300]}")
            except Exception as e:
                prec_log.write_text(f"Exception: {e}\n")
                log.warning(f"  Fortran precision exception for {ep}: {e}")

    result_data = {
        "benchmarks": benchmarks,
        "bench_files": bench_files,
        "bench_c_files": bench_c_files,
        "bench_rs_files": bench_rs_files,
        "datasets": all_datasets,
    }
    (output_dir / "benchmarks.json").write_text(json.dumps(result_data, indent=2, default=str))

    # ── Phase 4: stage-2 report ───────────────────────────────────────────────
    if status_fn:
        status_fn("Generating Stage 2 benchmark report…")
    _generate_benchmark_report(
        output_dir=output_dir,
        benchmarks=benchmarks,
        ep_n=ep_n,
        all_datasets=all_datasets,
        entry_points=entry_points,
    )

    log.info("Stage complete")
    return result_data



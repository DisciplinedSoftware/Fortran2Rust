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

N_DEFAULT = 500  # default matrix size used for initial calibration
N_MIN = 64
N_MAX = 2000
TARGET_TIME_MS_MIN = 1.0
TARGET_TIME_MS_MAX = 100.0
MAX_CALIBRATION_STEPS = 6
VECTOR_N_MAX = 2_048_000
VECTOR_CALIBRATION_STEPS = 13
MIN_EFFECTIVE_TIMING_MS = 1e-3
TIMING_RUNS_UNLIMITED_GUARDRAIL = 50000
TIMING_RUNS_TARGET_WINDOW_MS = 5.0
DEFAULT_TIMING_DAMPING = 0.75


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

_PREFIX_TO_PREC: dict[str, _Precision] = {
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


def _prefix_precision(fn_name: str) -> _Precision:
    """Return a precision fallback based on a conventional leading type prefix."""
    return _PREFIX_TO_PREC.get(fn_name[0].upper(), _PREC_D)


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


def _format_fortran_invocation(prefix: str, args: list[str], max_len: int = 100) -> str:
    """Format a Fortran invocation line with free-form continuation when needed."""
    indent = "            "
    first = f"{prefix}("
    current = first
    lines: list[str] = []

    for idx, arg in enumerate(args):
        suffix = ", " if idx < len(args) - 1 else ")"
        token = f"{arg}{suffix}"
        if len(indent + current + token) <= max_len:
            current += token
            continue
        lines.append(indent + current + " &")
        current = "& " + token

    lines.append(indent + current)
    return "\n".join(lines)


def _indent_fortran_block(block: str, spaces: int) -> str:
    pad = " " * spaces
    return "\n".join(pad + line if line.strip() else line for line in block.splitlines())


def _estimate_timing_runs(elapsed_ms: float, timing_max_runs: int, timing_damping: float) -> int:
    """Estimate repeat count for timing with quantization safeguards and sublinear damping."""
    effective_ms = max(elapsed_ms, MIN_EFFECTIVE_TIMING_MS)
    target_window_ms = max(TARGET_TIME_MS_MIN, TIMING_RUNS_TARGET_WINDOW_MS)
    est_runs = int((target_window_ms + effective_ms - 1e-9) / effective_ms)

    quantized_timer = elapsed_ms <= (MIN_EFFECTIVE_TIMING_MS + 1e-9)
    if est_runs > 1 and timing_damping < 1.0 and not quantized_timer:
        est_runs = int(est_runs ** timing_damping)
    est_runs = max(1, est_runs)
    if timing_max_runs == 0:
        return max(1, min(TIMING_RUNS_UNLIMITED_GUARDRAIL, est_runs))
    return max(1, min(timing_max_runs, est_runs))


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


def _has_numeric_array_params(sig: dict | None) -> bool:
    if sig is None:
        return False
    for p in sig.get("params", []):
        if p.get("is_array") and p.get("type") in _FORTRAN_TYPE_TO_PREC:
            return True
    return False

def _is_vector_signature(sig: dict | None) -> bool:
    """Heuristically classify signatures as vector-shaped without using named allowlists."""
    if sig is None:
        return False

    params = sig.get("params", [])
    if not params:
        return False

    numeric_arrays = [
        p for p in params
        if p.get("is_array") and p.get("type") in _FORTRAN_TYPE_TO_PREC
    ]
    if not numeric_arrays:
        return False

    pnames = [str(p.get("name", "")).upper() for p in params]
    if any(name.startswith("LD") for name in pnames):
        return False
    if any(name in {"INCX", "INCY", "STRIDEX", "STRIDEY"} for name in pnames):
        return True

    return len(numeric_arrays) == 1 and len(params) <= 3


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


def _make_fortran_driver_source(
    fn_name: str,
    n: int,
    sig: dict | None,
    prec: _Precision,
    timing_runs: int = 1,
) -> str:
    """Return the standard Fortran benchmark driver source for *fn_name* at size *n*."""
    return _make_generic_driver(fn_name, n, sig, prec, timing_runs=timing_runs)


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
    vector_n_max: int,
    timing_max_runs: int,
    timing_damping: float,
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
    is_vector = _is_vector_signature(sig)
    is_size_sensitive = _has_numeric_array_params(sig)
    n_max = min(VECTOR_N_MAX, vector_n_max) if is_vector else min(N_MAX, matrix_n_max)
    max_steps = VECTOR_CALIBRATION_STEPS if is_vector else MAX_CALIBRATION_STEPS
    fortran_deps = _resolve_fortran_deps(source_dir, dep_files, call_graph, fn_upper)

    for step in range(max_steps):
        if status_fn:
            status_fn(f"Calibrating benchmark size for {fn_name} (N={n})…")
        log.info(f"Calibrating {fn_name}: attempt {step + 1}/{max_steps}, N={n}")

        calib_seed = 42 + (step // max(1, dataset_reuse_every))
        generate_dataset(fn_name, n, output_dir, prec=prec, seed=calib_seed, sig=sig, vector_mode=is_vector)
        driver_src = _make_fortran_driver_source(fn_name, n, sig, prec, timing_runs=1)
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
                if not is_size_sensitive:
                    timing_runs = _estimate_timing_runs(elapsed_ms, timing_max_runs, timing_damping)
                    log.info(
                        "  Calibration %s: scalar/non-size-sensitive function (%.4fms); using %s timing run(s)",
                        fn_name,
                        elapsed_ms,
                        timing_runs,
                    )
                    return n, timing_runs
                # Vector kernels (e.g., DASUM/DNRM2) are often sub-ms even at large N.
                # Prefer repeated timing runs over growing N to huge values, which can
                # dramatically increase compile/runtime memory pressure.
                if is_vector:
                    timing_runs = _estimate_timing_runs(elapsed_ms, timing_max_runs, timing_damping)
                    log.info(
                        "  Vector calibration %s: N=%s too fast (%.4fms); using %s timing run(s) instead of larger N",
                        fn_name,
                        n,
                        elapsed_ms,
                        timing_runs,
                    )
                    return n, timing_runs
                if n >= n_max:
                    timing_runs = _estimate_timing_runs(elapsed_ms, timing_max_runs, timing_damping)
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


def generate_precision_dataset(
    fn_name: str,
    N: int,
    output_dir: Path,
    prec: _Precision = _PREC_D,
    sig: dict | None = None,
    vector_mode: bool | None = None,
) -> dict[str, Path]:
    """Generate near-cancellation dataset for precision testing (fixed seed for reproducibility)."""
    rng = np.random.default_rng(43)  # distinct seed from main dataset
    fn_lo = fn_name.lower()
    dtype = np.dtype(prec.numpy_dtype)
    EPS = 1e-8
    if vector_mode is None:
        vector_mode = _is_vector_signature(sig)

    if vector_mode:
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


def generate_dataset(
    fn_name: str,
    N: int,
    output_dir: Path,
    prec: _Precision = _PREC_D,
    seed: int = 42,
    sig: dict | None = None,
    vector_mode: bool | None = None,
) -> dict[str, Path]:
    """Generate common input dataset files (typed binary) shared by Fortran/C/Rust."""
    rng = np.random.default_rng(seed)
    dtype = np.dtype(prec.numpy_dtype)

    if vector_mode is None:
        vector_mode = _is_vector_signature(sig)

    if vector_mode:
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

    A.flatten(order="F").tofile(str(a_path))
    B.flatten(order="F").tofile(str(b_path))
    params.tofile(str(p_path))
    return {"A": a_path, "B": b_path, "params": p_path}



def _make_generic_driver(
    fn_name: str,
    N: int,
    sig: dict | None,
    prec: _Precision,
    timing_runs: int = 1,
) -> str:
    """
    Generate a Fortran benchmark driver from a parsed signature.

    When *sig* is available (parsed from source), produces a type-correct driver
    with a proper typed call.  Falls back to a DGEMM-shaped placeholder using
    *prec* when the signature cannot be parsed.
    """
    fn_up = fn_name.upper()
    fn_lo = fn_name.lower()
    vector_mode = _is_vector_signature(sig)

    bench_runs = max(1, int(timing_runs))

    if sig is None:
        # Fallback: typed placeholder with the dominant precision
        ftype = prec.fortran_type
        zero  = prec.fortran_zero
        one   = prec.fortran_one
        call_stmt = _format_fortran_invocation(
            f"CALL {fn_up}",
            ["'N'", "'N'", "N", "N", "N", "ALPHA", "A", "N", "B", "N", "BETA", "C", "N"],
        )
        return f"""\
      PROGRAM BENCH_{fn_up}
      IMPLICIT NONE
      INTEGER, PARAMETER :: N = {N}
      {ftype} :: A(N,N), B(N,N), C(N,N)
      {ftype} :: ALPHA, BETA
        DOUBLE PRECISION :: ELAPSED, T1, T2
            INTEGER :: I, J, BENCH_ITER, BENCH_RUNS, BENCH_RUNS_MAX
            DOUBLE PRECISION :: BENCH_TARGET_MS

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

            BENCH_RUNS = {bench_runs}
            BENCH_RUNS_MAX = MAX(1000000, BENCH_RUNS * 1024)
            BENCH_TARGET_MS = {TARGET_TIME_MS_MIN:.1f}D0

                        DO
                            CALL CPU_TIME(T1)
                            DO BENCH_ITER = 1, BENCH_RUNS
                                ! TODO: replace with the correct {fn_up}(...) call
{_indent_fortran_block(call_stmt, 8)}
                            END DO
                            CALL CPU_TIME(T2)
                            ELAPSED = (T2-T1) * 1000.0D0
                            IF (ELAPSED .GE. BENCH_TARGET_MS) EXIT
                            IF (BENCH_RUNS .GE. BENCH_RUNS_MAX) EXIT
                            BENCH_RUNS = MIN(BENCH_RUNS_MAX, BENCH_RUNS * 2)
                        END DO

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
    array_vars: list[str] = []

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
                    decls.append(f"      {ptype} :: {bvar}_ORIG(N)")
                else:
                    decls.append(f"      {ptype} :: {bvar}(N,N)")
                    decls.append(f"      {ptype} :: {bvar}_ORIG(N,N)")
                ds = "A" if arr_idx == 0 else "B"
                file_opens.append(
                    f"      OPEN({file_n}, FILE='dataset_{fn_lo}_{ds}.bin', FORM='UNFORMATTED', ACCESS='STREAM', STATUS='OLD')\n"
                    f"      READ({file_n}) {bvar}\n"
                    f"      CLOSE({file_n})"
                )
                file_n  += 1
                arr_idx += 1
                array_vars.append(bvar)
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
        call_stmt = _format_fortran_invocation(f"BENCH_RESULT = {fn_up}", call_args)
        out_var     = "BENCH_RESULT"
    else:
        call_stmt = _format_fortran_invocation(f"CALL {fn_up}", call_args)
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
    save_block = "\n".join([f"      {name}_ORIG = {name}" for name in array_vars])
    restore_block = "\n".join([f"      {name} = {name}_ORIG" for name in array_vars])
    timed_call_block = _indent_fortran_block(call_stmt, 6)

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
        DOUBLE PRECISION :: ELAPSED, T1, T2, BENCH_TARGET_MS
    INTEGER :: BENCH_ITER, BENCH_RUNS, BENCH_RUNS_MAX

{file_open_block}
{init_block}
{save_block}

    BENCH_RUNS = {bench_runs}
    BENCH_RUNS_MAX = MAX(1000000, BENCH_RUNS * 1024)
    BENCH_TARGET_MS = {TARGET_TIME_MS_MIN:.1f}D0

        DO
          CALL CPU_TIME(T1)
          DO BENCH_ITER = 1, BENCH_RUNS
{_indent_fortran_block(restore_block, 8)}
{_indent_fortran_block(call_stmt, 8)}
          END DO
          CALL CPU_TIME(T2)
          ELAPSED = (T2-T1) * 1000.0D0
          IF (ELAPSED .GE. BENCH_TARGET_MS) EXIT
          IF (BENCH_RUNS .GE. BENCH_RUNS_MAX) EXIT
          BENCH_RUNS = MIN(BENCH_RUNS_MAX, BENCH_RUNS * 2)
        END DO

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
    src = _make_generic_driver(fn_name, N, sig, prec, timing_runs=1)
    src = src.replace(f"PROGRAM BENCH_{fn_up}", f"PROGRAM BENCH_{fn_up}_PREC", 1)
    src = src.replace(f"dataset_{fn_lo}_A.bin", f"dataset_{fn_lo}_precision_A.bin")
    src = src.replace(f"dataset_{fn_lo}_B.bin", f"dataset_{fn_lo}_precision_B.bin")
    src = src.replace(f"bench_{fn_lo}_output.bin", f"bench_{fn_lo}_precision_output.bin")
    return src


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
    vector_mode = _is_vector_signature(sig)
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
    max_parallel: int = 1,
    matrix_n_max: int = 512,
    vector_n_max: int = 262144,
    timing_max_runs: int = 12,
    timing_damping: float = DEFAULT_TIMING_DAMPING,
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
    ep_vector_mode: dict[str, bool] = {}
    ep_n:    dict[str, int] = {}
    ep_timing_runs: dict[str, int] = {}
    for ep in entry_points:
        if status_fn:
            status_fn(f"Parsing Fortran signature for {ep}…")
        sig = _parse_fn_signature(ep, source_dir)
        if sig is None:
            prec = _prefix_precision(ep)
            log.warning(f"  Could not parse signature for {ep}; using name-prefix precision {prec.fortran_type}")
        else:
            prec = _dominant_precision(sig)
            log.info(f"  Parsed signature for {ep}: {prec.fortran_type}")
        ep_prec[ep] = prec
        ep_sig[ep]  = sig
        ep_vector_mode[ep] = _is_vector_signature(sig)

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
            vector_n_max=vector_n_max,
            timing_max_runs=timing_max_runs,
            timing_damping=timing_damping,
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
        sig = ep_sig[ep]
        N = ep_n[ep]
        dataset = generate_dataset(ep, N, output_dir, prec=prec, sig=sig, vector_mode=ep_vector_mode[ep])
        all_datasets[ep] = {k: str(v) for k, v in dataset.items()}
        log.info(f"  dataset for {ep} ({prec.numpy_dtype}, N={N}): {list(dataset.keys())}")
        prec_dataset = generate_precision_dataset(ep, N, output_dir, prec=prec, sig=sig, vector_mode=ep_vector_mode[ep])
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

        driver_src      = _make_generic_driver(ep, N, sig, prec, timing_runs=ep_timing_runs[ep])
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

        # Rust benchmark sources are produced downstream from transpiled output.

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
                baseline_runs = 1
                run_result = None
                for run_idx in range(baseline_runs):
                    if run_idx == 0 or run_idx % max(1, dataset_reuse_every) == 0:
                        seed = 42 + (run_idx // max(1, dataset_reuse_every))
                        rotated = generate_dataset(
                            ep,
                            N,
                            output_dir,
                            prec=prec,
                            seed=seed,
                            sig=sig,
                            vector_mode=ep_vector_mode[ep],
                        )
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
                        "  Fortran baseline time: %.4f ms (driver timing_runs=%d)",
                        bench_info["time_ms"],
                        max(1, ep_timing_runs.get(ep, 1)),
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



from __future__ import annotations

import json
import re
import subprocess
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ._log import make_stage_logger

N_DEFAULT = 500  # matrix size giving ~1ms per DGEMM call


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

    for f in sorted(source_dir.glob("*.f")):
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

            params = [
                {
                    "name": pname,
                    "type": type_map.get(pname, {}).get("type", "DOUBLE PRECISION"),
                    "is_array": type_map.get(pname, {}).get("is_array", False),
                }
                for pname in param_names
            ]
            return {"is_function": is_function, "params": params}

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


def generate_precision_dataset(fn_name: str, N: int, output_dir: Path,
                               prec: _Precision = _PREC_D) -> dict[str, Path]:
    """Generate near-cancellation dataset for precision testing (fixed seed for reproducibility)."""
    rng = np.random.default_rng(43)  # distinct seed from main dataset
    fn_lo = fn_name.lower()
    dtype = np.dtype(prec.numpy_dtype)
    EPS = 1e-8
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
                     prec: _Precision = _PREC_D) -> dict[str, Path]:
    """Generate common input dataset files (typed binary) shared by Fortran/C/Rust."""
    rng = np.random.default_rng(42)
    fn_upper = fn_name.upper()
    dtype = np.dtype(prec.numpy_dtype)

    if fn_upper in KNOWN_BLAS:
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
      INTEGER :: ITER, COUNT_RATE, T1, T2
      DOUBLE PRECISION :: ELAPSED
      INTEGER :: I, J

      ! Load shared dataset (raw binary, column-major)
      OPEN(10, FILE='dataset_{fn_lo}_A.bin', FORM='UNFORMATTED',
     $     ACCESS='STREAM', STATUS='OLD')
      READ(10) A
      CLOSE(10)

      OPEN(11, FILE='dataset_{fn_lo}_B.bin', FORM='UNFORMATTED',
     $     ACCESS='STREAM', STATUS='OLD')
      READ(11) B
      CLOSE(11)

      OPEN(12, FILE='dataset_{fn_lo}_params.bin', FORM='UNFORMATTED',
     $     ACCESS='STREAM', STATUS='OLD')
      READ(12) ALPHA
      READ(12) BETA
      CLOSE(12)

      DO I = 1, N
        DO J = 1, N
          C(I,J) = {zero}
        END DO
      END DO

      ! Time ONLY the computation, not I/O
      CALL SYSTEM_CLOCK(COUNT_RATE=COUNT_RATE)
      CALL SYSTEM_CLOCK(T1)
      DO ITER = 1, 10
        CALL {fn_up}('N','N',N,N,N,ALPHA,A,N,B,N,BETA,C,N)
      END DO
      CALL SYSTEM_CLOCK(T2)
      ELAPSED = DBLE(T2-T1) / DBLE(COUNT_RATE) * 1000.0D0 / 10.0D0

      ! Write output (column-major raw binary for numpy comparison)
      OPEN(13, FILE='bench_{fn_lo}_output.bin', FORM='UNFORMATTED',
     $     ACCESS='STREAM', STATUS='REPLACE')
      WRITE(13) C
      CLOSE(13)

      WRITE(*,'(A,F12.4)') 'FORTRAN_TIME_MS=', ELAPSED
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
      OPEN(10, FILE='dataset_{fn_lo}_precision_A.bin', FORM='UNFORMATTED',
     $     ACCESS='STREAM', STATUS='OLD')
      READ(10) A
      CLOSE(10)
      OPEN(11, FILE='dataset_{fn_lo}_precision_B.bin', FORM='UNFORMATTED',
     $     ACCESS='STREAM', STATUS='OLD')
      READ(11) B
      CLOSE(11)

      DO I = 1, N
        DO J = 1, N
          C(I,J) = {zero}
        END DO
      END DO

      CALL {fn_up}('N','N',N,N,N,ALPHA,A,N,B,N,BETA,C,N)

      OPEN(13, FILE='bench_{fn_lo}_precision_output.bin',
     $     FORM='UNFORMATTED', ACCESS='STREAM',
     $     STATUS='REPLACE')
      WRITE(13) C
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
      INTEGER :: ITER, COUNT_RATE, T1, T2
      DOUBLE PRECISION :: ELAPSED
      INTEGER :: I, J

      OPEN(10, FILE='dataset_{fn_lo}_A.bin', FORM='UNFORMATTED',
     $     ACCESS='STREAM', STATUS='OLD')
      READ(10) A
      CLOSE(10)
      OPEN(11, FILE='dataset_{fn_lo}_B.bin', FORM='UNFORMATTED',
     $     ACCESS='STREAM', STATUS='OLD')
      READ(11) B
      CLOSE(11)

      ALPHA = {one}
      BETA = {zero}
      DO I = 1, N
        DO J = 1, N
          C(I,J) = {zero}
        END DO
      END DO

      CALL SYSTEM_CLOCK(COUNT_RATE=COUNT_RATE)
      CALL SYSTEM_CLOCK(T1)
      DO ITER = 1, 10
        ! TODO: replace with the correct {fn_up}(...) call
        CALL {fn_up}('N','N',N,N,N,ALPHA,A,N,B,N,BETA,C,N)
      END DO
      CALL SYSTEM_CLOCK(T2)
      ELAPSED = DBLE(T2-T1) / DBLE(COUNT_RATE) * 1000.0D0 / 10.0D0

      OPEN(13, FILE='bench_{fn_lo}_output.bin', FORM='UNFORMATTED',
     $     ACCESS='STREAM', STATUS='REPLACE')
      WRITE(13) C
      CLOSE(13)

      WRITE(*,'(A,F12.4)') 'FORTRAN_TIME_MS=', ELAPSED
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

        elif p_prec is not None:
            if p["is_array"]:
                decls.append(f"      {ptype} :: {bvar}(N,N)")
                ds = "A" if arr_idx == 0 else "B"
                file_opens.append(
                    f"      OPEN({file_n}, FILE='dataset_{fn_lo}_{ds}.bin',"
                    f" FORM='UNFORMATTED',\n"
                    f"     $     ACCESS='STREAM', STATUS='OLD')\n"
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
    call_stmt   = ""
    if is_fn:
        ret_prec  = prec
        result_decl = f"      {ret_prec.fortran_type} :: BENCH_RESULT\n"
        call_stmt   = f"BENCH_RESULT = {fn_up}({', '.join(call_args)})"
        out_var     = "BENCH_RESULT"
    else:
        call_stmt = f"CALL {fn_up}({', '.join(call_args)})"
        # Write the last float array as output, or first if none
        float_arrays = [p for p in params if p["is_array"] and p["type"] in _FORTRAN_TYPE_TO_PREC]
        out_var = f"BENCH_{float_arrays[-1]['name']}" if float_arrays else None

    decl_block      = "\n".join(decls)
    init_block      = "\n".join(inits)
    file_open_block = "\n".join(file_opens)

    output_block = ""
    if out_var:
        output_block = (
            f"      OPEN(13, FILE='bench_{fn_lo}_output.bin', FORM='UNFORMATTED',\n"
            f"     $     ACCESS='STREAM', STATUS='REPLACE')\n"
            f"      WRITE(13) {out_var}\n"
            f"      CLOSE(13)\n"
        )

    return f"""\
      PROGRAM BENCH_{fn_up}
      IMPLICIT NONE
      INTEGER, PARAMETER :: N = {N}
{result_decl}{decl_block}
      INTEGER :: ITER, COUNT_RATE, T1, T2
      DOUBLE PRECISION :: ELAPSED

{file_open_block}
{init_block}

      CALL SYSTEM_CLOCK(COUNT_RATE=COUNT_RATE)
      CALL SYSTEM_CLOCK(T1)
      DO ITER = 1, 10
        {call_stmt}
      END DO
      CALL SYSTEM_CLOCK(T2)
      ELAPSED = DBLE(T2-T1) / DBLE(COUNT_RATE) * 1000.0D0 / 10.0D0

{output_block}
      WRITE(*,'(A,F12.4)') 'FORTRAN_TIME_MS=', ELAPSED
      END PROGRAM
"""


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

    # Build typed variable declarations and the function call from the signature
    if sig is not None:
        params     = sig["params"]
        char_params: list[str] = []
        decls:      list[str] = []
        inits:      list[str] = []
        reads:      list[str] = []
        call_args:  list[str] = []
        arr_idx = 0

        for p in params:
            pname  = p["name"].lower()
            ptype  = p["type"]
            p_prec = _FORTRAN_TYPE_TO_PREC.get(ptype)
            cname  = f"bench_{pname}"

            if "CHARACTER" in ptype:
                decls.append(f"    char {cname} = 'N';")
                call_args.append(f"&{cname}")
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

            elif p_prec is not None:
                pctype = p_prec.c_type
                if p["is_array"]:
                    ds = "A" if arr_idx == 0 else "B"
                    decls.append(f"    static {pctype} {cname}[BENCH_N * BENCH_N];")
                    reads.append(
                        f'    read_bin("dataset_{fn_lo}_{ds}.bin",'
                        f" {cname}, (size_t)BENCH_N * BENCH_N);"
                    )
                    arr_idx += 1
                    call_args.append(cname)
                else:
                    pu = pname.upper()
                    val = p_prec.fortran_one.replace("D0", "").replace("0D0", "0").replace("E0", "")
                    decls.append(f"    {pctype} {cname};")
                    inits.append(f"    memset(&{cname}, 0, sizeof({cname}));")
                    if pu in ("ALPHA", "SCALE", "DA", "SA", "ZA", "CA", "A", "W"):
                        inits.append(f"    *((double *)&{cname}) = 1.0;  /* alpha = 1 */")
                    call_args.append(f"&{cname}")
            else:
                decls.append(f"    /* TODO: {ptype} {cname}; */")
                call_args.append(f"&{cname}")

        # Append ftnlen=1 for each CHARACTER parameter
        ftnlen_args = ["1"] * len(char_params)
        all_args = call_args + ftnlen_args
        call_line = f"    /* {fn_ext}({', '.join(all_args)}); */"

        # Determine output array (last float array in params)
        float_arrays = [
            p["name"].lower()
            for p in params
            if p["is_array"] and p["type"] in _FORTRAN_TYPE_TO_PREC
        ]
        out_arr   = f"bench_{float_arrays[-1]}" if float_arrays else "bench_a"
        out_ctype = ct

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
            f'    read_bin("dataset_{fn_lo}_A.bin", bench_A, (size_t)BENCH_N * BENCH_N);\n'
            f'    read_bin("dataset_{fn_lo}_B.bin", bench_B, (size_t)BENCH_N * BENCH_N);'
        )
        call_line   = f"    /* TODO: {fn_ext}(...); */"
        out_arr     = "bench_C"
        out_ctype   = ct

    return f"""\
/* C benchmark driver for {fn_name} — generated by Fortran2Rust */
/* TODO: verify function signature and call against the f2c output */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "f2c.h"

#define BENCH_N {N}

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
    fwrite({out_arr}, sizeof({out_ctype}), (size_t)BENCH_N * BENCH_N, out);
    fclose(out);

    printf("C_TIME_MS=%.4f\\n", elapsed_ms);
    return 0;
}}
"""


def generate_benchmarks(
    source_dir: Path,
    entry_points: list[str],
    dep_files: list[Path],
    output_dir: Path,
    status_fn=None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    N = N_DEFAULT
    log = make_stage_logger(output_dir)
    log.info(f"generate_benchmarks: entry_points={entry_points}, N={N}")

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

    all_datasets: dict[str, dict] = {}
    for ep in entry_points:
        prec = ep_prec[ep]
        dataset = generate_dataset(ep, N, output_dir, prec=prec)
        all_datasets[ep] = {k: str(v) for k, v in dataset.items()}
        log.info(f"  dataset for {ep} ({prec.numpy_dtype}): {list(dataset.keys())}")
        if ep.upper() in KNOWN_BLAS:
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
        prec = ep_prec[ep]
        sig  = ep_sig[ep]
        dataset_paths = {k: Path(v) for k, v in all_datasets[ep].items()}

        if ep_upper in KNOWN_BLAS:
            driver_src      = _make_dgemm_driver(ep, N, prec)
            precision_src   = _make_dgemm_precision_driver(ep, N, prec)
            c_driver_src    = _make_c_blas_driver(ep, N, prec)
            c_precision_src = _make_c_blas_precision_driver(ep, N, prec)
        else:
            driver_src      = _make_generic_driver(ep, N, sig, prec)
            precision_src   = None
            c_driver_src    = _make_c_generic_driver(ep, N, sig, prec)
            c_precision_src = None

        driver_file = output_dir / f"bench_{ep_lower}.f"
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
            prec_file = output_dir / f"bench_{ep_lower}_precision.f"
            prec_file.write_text(precision_src)
            bench_files.append(str(prec_file))

        # Write Rust bench drivers for real-precision BLAS functions.
        # These are used by stages 6-8 instead of the c2rust-generated bench files,
        # because c2rust bench files reference f2c I/O symbols (s_wsfe, do_fio, …)
        # that are undefined at binary link time.  The generated files include
        # no-op stubs for those symbols so the bench binary links cleanly.
        if ep_upper in KNOWN_BLAS and not prec.is_complex:
            rs_file = output_dir / f"bench_{ep_lower}.rs"
            rs_file.write_text(_make_rust_blas_driver(ep, N, prec))
            bench_rs_files.append(str(rs_file))
            log.info(f"  Wrote Rust bench driver: {rs_file.name}")

            rs_prec_file = output_dir / f"bench_{ep_lower}_precision.rs"
            rs_prec_file.write_text(_make_rust_blas_precision_driver(ep, N, prec))
            bench_rs_files.append(str(rs_prec_file))
            log.info(f"  Wrote Rust precision bench driver: {rs_prec_file.name}")

        # Compile and run Fortran benchmark to produce the reference output
        # Use absolute paths — relative paths fail when gfortran is not run from the repo root.
        fortran_deps = [str(Path(f).resolve()) for f in dep_files if Path(f).exists()]
        exe_path = (output_dir / f"bench_{ep_lower}").resolve()
        compile_cmd = (
            ["gfortran", "-O2", str(driver_file.resolve())]
            + fortran_deps
            + ["-lm", "-o", str(exe_path)]
        )

        bench_info: dict = {
            "driver_file": str(driver_file),
            "dataset": {k: str(v) for k, v in dataset_paths.items()},
            "compile_cmd": compile_cmd,
            "compile_ok": False,
            "compile_stdout": "",
            "compile_stderr": "",
            "run_stdout": "",
            "run_stderr": "",
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
                run_result = subprocess.run(
                    [str(exe_path)],
                    capture_output=True, text=True,
                    cwd=str(output_dir.resolve()), timeout=300,
                )
                bench_info["run_stdout"] = run_result.stdout
                bench_info["run_stderr"] = run_result.stderr
                with open(compile_log, "a") as fh:
                    fh.write(
                        f"\n=== RUN STDOUT ===\n{run_result.stdout}\n"
                        f"=== RUN STDERR ===\n{run_result.stderr}\n"
                        f"=== RUN EXIT CODE: {run_result.returncode} ===\n"
                    )
                match = re.search(r"FORTRAN_TIME_MS=\s*([\d.]+)", run_result.stdout)
                if match:
                    bench_info["time_ms"] = float(match.group(1))
                    log.info(f"  Fortran baseline time: {bench_info['time_ms']:.4f} ms")
                bin_file = output_dir / f"bench_{ep_lower}_output.bin"
                if bin_file.exists():
                    bench_info["output_file"] = str(bin_file)
            else:
                log.warning(f"  gfortran FAILED for {ep} (exit {result.returncode})")
                log.warning(f"  stderr: {result.stderr[:500]}")
        except subprocess.TimeoutExpired:
            bench_info["compile_stderr"] = "Timeout"
            log.warning(f"  gfortran timed out for {ep}")
        except Exception as e:
            bench_info["compile_stderr"] = str(e)
            log.warning(f"  gfortran exception for {ep}: {e}")

        benchmarks[ep] = bench_info

        # Compile and run precision Fortran driver (BLAS functions only).
        # Produces bench_{ep_lower}_precision_output.bin used as baseline in Stage 4.
        if ep_upper in KNOWN_BLAS:
            prec_driver_file = output_dir / f"bench_{ep_lower}_precision.f"
            if prec_driver_file.exists():
                prec_exe = (output_dir / f"bench_{ep_lower}_precision").resolve()
                prec_compile_cmd = (
                    ["gfortran", "-O2", str(prec_driver_file.resolve())]
                    + fortran_deps
                    + ["-lm", "-o", str(prec_exe)]
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
    log.info("Stage complete")
    return result_data



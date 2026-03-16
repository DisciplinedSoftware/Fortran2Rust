from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import numpy as np

from ._log import make_stage_logger

N_DEFAULT = 500  # matrix size giving ~1ms per DGEMM call

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


def generate_precision_dataset(fn_name: str, N: int, output_dir: Path) -> dict[str, Path]:
    """Generate near-cancellation dataset for precision testing (fixed seed for reproducibility)."""
    rng = np.random.default_rng(43)  # distinct seed from main dataset
    fn_lo = fn_name.lower()
    EPS = 1e-8
    A = (1.0 + EPS * (rng.random((N, N)).astype(np.float64) - 0.5))
    B = (1.0 + EPS * (rng.random((N, N)).astype(np.float64) - 0.5))
    a_path = output_dir / f"dataset_{fn_lo}_precision_A.bin"
    b_path = output_dir / f"dataset_{fn_lo}_precision_B.bin"
    A.flatten(order="F").tofile(str(a_path))
    B.flatten(order="F").tofile(str(b_path))
    return {"A": a_path, "B": b_path}


def generate_dataset(fn_name: str, N: int, output_dir: Path) -> dict[str, Path]:
    """Generate common input dataset files (raw float64 binary) shared by Fortran/C/Rust."""
    rng = np.random.default_rng(42)
    fn_upper = fn_name.upper()

    if fn_upper in KNOWN_BLAS:
        A = rng.random((N, N)).astype(np.float64)
        B = rng.random((N, N)).astype(np.float64)
        alpha = np.float64(1.0)
        beta = np.float64(0.0)

        a_path = output_dir / f"dataset_{fn_name.lower()}_A.bin"
        b_path = output_dir / f"dataset_{fn_name.lower()}_B.bin"
        p_path = output_dir / f"dataset_{fn_name.lower()}_params.bin"

        # Column-major (Fortran order) for A and B so Fortran reads them directly
        A.flatten(order="F").tofile(str(a_path))
        B.flatten(order="F").tofile(str(b_path))
        np.array([alpha, beta]).tofile(str(p_path))

        return {"A": a_path, "B": b_path, "params": p_path}
    else:
        # Generic: just write two NxN matrices
        A = rng.random((N, N)).astype(np.float64)
        B = rng.random((N, N)).astype(np.float64)
        a_path = output_dir / f"dataset_{fn_name.lower()}_A.bin"
        b_path = output_dir / f"dataset_{fn_name.lower()}_B.bin"
        A.flatten(order="F").tofile(str(a_path))
        B.flatten(order="F").tofile(str(b_path))
        return {"A": a_path, "B": b_path}


def _make_dgemm_driver(fn_name: str, N: int) -> str:
    fn_up = fn_name.upper()
    fn_lo = fn_name.lower()
    return f"""\
      PROGRAM BENCH_{fn_up}
      IMPLICIT NONE
      INTEGER, PARAMETER :: N = {N}
      DOUBLE PRECISION :: A(N,N), B(N,N), C(N,N)
      DOUBLE PRECISION :: ALPHA, BETA
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
          C(I,J) = 0.0D0
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


def _make_dgemm_precision_driver(fn_name: str, N: int) -> str:
    """Near-cancellation precision test driver — reads shared dataset for cross-language reproducibility."""
    fn_up = fn_name.upper()
    fn_lo = fn_name.lower()
    return f"""\
      PROGRAM BENCH_{fn_up}_PREC
      IMPLICIT NONE
      INTEGER, PARAMETER :: N = {N}
      DOUBLE PRECISION :: A(N,N), B(N,N), C(N,N)
      DOUBLE PRECISION :: ALPHA, BETA
      INTEGER :: I, J

      ALPHA = 1.0D0
      BETA = 0.0D0

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
          C(I,J) = 0.0D0
        END DO
      END DO

      CALL {fn_up}('N','N',N,N,N,ALPHA,A,N,B,N,BETA,C,N)

      OPEN(13, FILE='bench_{fn_lo}_precision_output.bin', FORM='UNFORMATTED',
     $     ACCESS='STREAM', STATUS='REPLACE')
      WRITE(13) C
      CLOSE(13)

      WRITE(*,*) 'PRECISION_TEST_DONE'
      END PROGRAM
"""


def _make_generic_driver(fn_name: str, N: int) -> str:
    fn_up = fn_name.upper()
    fn_lo = fn_name.lower()
    return f"""\
      PROGRAM BENCH_{fn_up}
      IMPLICIT NONE
      INTEGER, PARAMETER :: N = {N}
      DOUBLE PRECISION :: A(N,N), B(N,N), C(N,N)
      DOUBLE PRECISION :: ALPHA, BETA
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

      ALPHA = 1.0D0
      BETA = 0.0D0
      DO I = 1, N
        DO J = 1, N
          C(I,J) = 0.0D0
        END DO
      END DO

      CALL SYSTEM_CLOCK(COUNT_RATE=COUNT_RATE)
      CALL SYSTEM_CLOCK(T1)
      DO ITER = 1, 10
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


def _make_c_blas_driver(fn_name: str, N: int) -> str:
    """
    C benchmark driver for a BLAS dgemm-style function.
    Reads the numpy-generated binary dataset, calls the f2c-transpiled function,
    writes the output binary, and prints timing. Does NOT go through f2c.
    """
    fn_lo = fn_name.lower()
    fn_ext = fn_lo + "_"  # f2c appends underscore
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
                    doublereal *alpha, doublereal *a, integer *lda,
                    doublereal *b, integer *ldb, doublereal *beta,
                    doublereal *c__, integer *ldc,
                    ftnlen transa_len, ftnlen transb_len);

static doublereal bench_A[BENCH_N * BENCH_N];
static doublereal bench_B[BENCH_N * BENCH_N];
static doublereal bench_C[BENCH_N * BENCH_N];

static void read_bin(const char *path, doublereal *buf, size_t count) {{
    FILE *fp = fopen(path, "rb");
    if (!fp) {{ perror(path); exit(1); }}
    if (fread(buf, sizeof(doublereal), count, fp) != count) {{
        fprintf(stderr, "Short read: %s\\n", path); exit(1);
    }}
    fclose(fp);
}}

int main(void) {{
    doublereal params[2];
    read_bin("dataset_{fn_lo}_A.bin", bench_A, (size_t)BENCH_N * BENCH_N);
    read_bin("dataset_{fn_lo}_B.bin", bench_B, (size_t)BENCH_N * BENCH_N);
    read_bin("dataset_{fn_lo}_params.bin", params, 2);
    doublereal alpha = params[0], beta = params[1];

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
    fwrite(bench_C, sizeof(doublereal), (size_t)BENCH_N * BENCH_N, out);
    fclose(out);

    printf("C_TIME_MS=%.4f\\n", elapsed_ms);
    return 0;
}}
"""


def _make_c_blas_precision_driver(fn_name: str, N: int) -> str:
    """
    C near-cancellation precision benchmark driver for a BLAS dgemm-style function.
    Reads the shared precision dataset (generated with fixed seed 43), calls the
    f2c-transpiled function, and writes output binary. No timing — precision only.
    """
    fn_lo = fn_name.lower()
    fn_ext = fn_lo + "_"
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
                    doublereal *alpha, doublereal *a, integer *lda,
                    doublereal *b, integer *ldb, doublereal *beta,
                    doublereal *c__, integer *ldc,
                    ftnlen transa_len, ftnlen transb_len);

static doublereal prec_A[BENCH_N * BENCH_N];
static doublereal prec_B[BENCH_N * BENCH_N];
static doublereal prec_C[BENCH_N * BENCH_N];

static void read_bin(const char *path, doublereal *buf, size_t count) {{
    FILE *fp = fopen(path, "rb");
    if (!fp) {{ perror(path); exit(1); }}
    if (fread(buf, sizeof(doublereal), count, fp) != count) {{
        fprintf(stderr, "Short read: %s\\n", path); exit(1);
    }}
    fclose(fp);
}}

int main(void) {{
    read_bin("dataset_{fn_lo}_precision_A.bin", prec_A, (size_t)BENCH_N * BENCH_N);
    read_bin("dataset_{fn_lo}_precision_B.bin", prec_B, (size_t)BENCH_N * BENCH_N);

    doublereal alpha = 1.0, beta = 0.0;
    integer m = BENCH_N, n = BENCH_N, k = BENCH_N;
    integer lda = BENCH_N, ldb = BENCH_N, ldc = BENCH_N;
    char transa = 'N', transb = 'N';

    memset(prec_C, 0, sizeof(prec_C));
    {fn_ext}(&transa, &transb, &m, &n, &k, &alpha, prec_A, &lda, prec_B, &ldb, &beta, prec_C, &ldc, 1, 1);

    FILE *out = fopen("bench_{fn_lo}_precision_output.bin", "wb");
    if (!out) {{ perror("bench_{fn_lo}_precision_output.bin"); exit(1); }}
    fwrite(prec_C, sizeof(doublereal), (size_t)BENCH_N * BENCH_N, out);
    fclose(out);

    printf("C_PRECISION_DONE\\n");
    return 0;
}}
"""


def _make_c_generic_driver(fn_name: str, N: int) -> str:
    """Generic C benchmark stub — prints a placeholder; the LLM will fill in the real call."""
    fn_lo = fn_name.lower()
    fn_ext = fn_lo + "_"
    return f"""\
/* C benchmark driver for {fn_name} — generated by Fortran2Rust */
/* TODO: adjust function signature and call to match the f2c output */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "f2c.h"

#define BENCH_N {N}

static doublereal bench_A[BENCH_N * BENCH_N];
static doublereal bench_B[BENCH_N * BENCH_N];
static doublereal bench_C[BENCH_N * BENCH_N];

static void read_bin(const char *path, doublereal *buf, size_t count) {{
    FILE *fp = fopen(path, "rb");
    if (!fp) {{ perror(path); exit(1); }}
    if (fread(buf, sizeof(doublereal), count, fp) != count) {{
        fprintf(stderr, "Short read: %s\\n", path); exit(1);
    }}
    fclose(fp);
}}

int main(void) {{
    read_bin("dataset_{fn_lo}_A.bin", bench_A, (size_t)BENCH_N * BENCH_N);
    read_bin("dataset_{fn_lo}_B.bin", bench_B, (size_t)BENCH_N * BENCH_N);

    struct timespec t1, t2;
    clock_gettime(CLOCK_MONOTONIC, &t1);
    /* TODO: call {fn_ext}(...) here */
    clock_gettime(CLOCK_MONOTONIC, &t2);
    double elapsed_ms = ((t2.tv_sec - t1.tv_sec) * 1e9 + (t2.tv_nsec - t1.tv_nsec)) / 1e6;

    FILE *out = fopen("bench_{fn_lo}_output.bin", "wb");
    if (!out) {{ perror("bench_{fn_lo}_output.bin"); exit(1); }}
    fwrite(bench_C, sizeof(doublereal), (size_t)BENCH_N * BENCH_N, out);
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
    all_datasets: dict[str, dict] = {}
    for ep in entry_points:
        dataset = generate_dataset(ep, N, output_dir)
        all_datasets[ep] = {k: str(v) for k, v in dataset.items()}
        log.info(f"  dataset for {ep}: {list(dataset.keys())}")
        if ep.upper() in KNOWN_BLAS:
            prec_dataset = generate_precision_dataset(ep, N, output_dir)
            all_datasets[ep + "_precision"] = {k: str(v) for k, v in prec_dataset.items()}
    (output_dir / "datasets.json").write_text(json.dumps(all_datasets, indent=2))

    # ── Phase 2: benchmark drivers (Fortran + C) ──────────────────────────────
    benchmarks: dict[str, dict] = {}
    bench_files: list[str] = []      # Fortran .f drivers (gfortran, modern syntax)
    bench_c_files: list[str] = []    # C drivers (written directly, no f2c needed)

    for ep in entry_points:
        ep_upper = ep.upper()
        ep_lower = ep.lower()
        dataset_paths = {k: Path(v) for k, v in all_datasets[ep].items()}

        if ep_upper in KNOWN_BLAS:
            driver_src = _make_dgemm_driver(ep, N)
            precision_src = _make_dgemm_precision_driver(ep, N)
            c_driver_src = _make_c_blas_driver(ep, N)
            c_precision_src = _make_c_blas_precision_driver(ep, N)
        else:
            driver_src = _make_generic_driver(ep, N)
            precision_src = None
            c_driver_src = _make_c_generic_driver(ep, N)
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

    result_data = {
        "benchmarks": benchmarks,
        "bench_files": bench_files,
        "bench_c_files": bench_c_files,
        "datasets": all_datasets,
    }
    (output_dir / "benchmarks.json").write_text(json.dumps(result_data, indent=2, default=str))
    log.info("Stage complete")
    return result_data



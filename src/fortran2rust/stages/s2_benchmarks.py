from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import numpy as np

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
    """Near-cancellation precision test driver."""
    fn_up = fn_name.upper()
    fn_lo = fn_name.lower()
    return f"""\
      PROGRAM BENCH_{fn_up}_PREC
      IMPLICIT NONE
      INTEGER, PARAMETER :: N = {N}
      DOUBLE PRECISION :: A(N,N), B(N,N), C(N,N)
      DOUBLE PRECISION :: ALPHA, BETA
      INTEGER :: I, J
      DOUBLE PRECISION, PARAMETER :: EPS = 1.0D-8
      DOUBLE PRECISION :: RV

      ALPHA = 1.0D0
      BETA = 0.0D0
      DO I = 1, N
        DO J = 1, N
          ! Near-cancellation: values cluster near 1.0 with tiny perturbation
          CALL RANDOM_NUMBER(RV)
          A(I,J) = 1.0D0 + EPS * (RV - 0.5D0)
          CALL RANDOM_NUMBER(RV)
          B(I,J) = 1.0D0 + EPS * (RV - 0.5D0)
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


def generate_benchmarks(
    source_dir: Path,
    entry_points: list[str],
    dep_files: list[Path],
    output_dir: Path,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    benchmarks: dict[str, dict] = {}
    bench_files: list[str] = []
    N = N_DEFAULT

    for ep in entry_points:
        ep_upper = ep.upper()
        ep_lower = ep.lower()

        # Generate shared dataset files
        dataset = generate_dataset(ep, N, output_dir)

        if ep_upper in KNOWN_BLAS:
            driver_src = _make_dgemm_driver(ep, N)
            precision_src = _make_dgemm_precision_driver(ep, N)
        else:
            driver_src = _make_generic_driver(ep, N)
            precision_src = None

        driver_file = output_dir / f"bench_{ep_lower}.f"
        driver_file.write_text(driver_src)
        bench_files.append(str(driver_file))

        if precision_src:
            prec_file = output_dir / f"bench_{ep_lower}_precision.f"
            prec_file.write_text(precision_src)
            bench_files.append(str(prec_file))

        # Only compile the dep_files from the dependency graph
        fortran_deps = [str(f) for f in dep_files if f.exists()]
        compile_cmd = (
            ["gfortran", "-O2", str(driver_file)]
            + fortran_deps
            + ["-lm", "-o", str(output_dir / f"bench_{ep_lower}")]
        )

        bench_info: dict = {
            "driver_file": str(driver_file),
            "dataset": {k: str(v) for k, v in dataset.items()},
            "compile_cmd": compile_cmd,
            "compile_ok": False,
            "compile_error": "",
            "time_ms": None,
            "output_file": None,
        }

        try:
            result = subprocess.run(
                compile_cmd, capture_output=True, text=True,
                cwd=str(output_dir), timeout=120,
            )
            if result.returncode == 0:
                bench_info["compile_ok"] = True
                run_result = subprocess.run(
                    [str(output_dir / f"bench_{ep_lower}")],
                    capture_output=True, text=True,
                    cwd=str(output_dir), timeout=300,
                )
                stdout = run_result.stdout
                match = re.search(r"FORTRAN_TIME_MS=\s*([\d.]+)", stdout)
                if match:
                    bench_info["time_ms"] = float(match.group(1))

                bin_file = output_dir / f"bench_{ep_lower}_output.bin"
                if bin_file.exists():
                    bench_info["output_file"] = str(bin_file)
            else:
                bench_info["compile_error"] = result.stderr
        except subprocess.TimeoutExpired:
            bench_info["compile_error"] = "Timeout"
        except Exception as e:
            bench_info["compile_error"] = str(e)

        benchmarks[ep] = bench_info

    result_data = {"benchmarks": benchmarks, "bench_files": bench_files}
    (output_dir / "benchmarks.json").write_text(json.dumps(result_data, indent=2, default=str))
    return result_data


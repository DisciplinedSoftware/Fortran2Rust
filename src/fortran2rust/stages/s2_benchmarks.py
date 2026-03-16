from __future__ import annotations

import json
import subprocess
import re
from pathlib import Path

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

DGEMM_DRIVER = """\
      PROGRAM BENCH_DGEMM
      IMPLICIT NONE
      INTEGER, PARAMETER :: N = 500
      DOUBLE PRECISION :: A(N,N), B(N,N), C(N,N)
      DOUBLE PRECISION :: ALPHA, BETA
      INTEGER :: I, J, ITER, COUNT_RATE, T1, T2
      DOUBLE PRECISION :: ELAPSED

      ALPHA = 1.0D0
      BETA = 0.0D0
      DO I = 1, N
        DO J = 1, N
          A(I,J) = DBLE(I+J) / DBLE(N)
          B(I,J) = DBLE(I*J) / DBLE(N*N)
          C(I,J) = 0.0D0
        END DO
      END DO

      CALL SYSTEM_CLOCK(COUNT_RATE=COUNT_RATE)
      CALL SYSTEM_CLOCK(T1)
      DO ITER = 1, 10
        CALL DGEMM('N','N',N,N,N,ALPHA,A,N,B,N,BETA,C,N)
      END DO
      CALL SYSTEM_CLOCK(T2)
      ELAPSED = DBLE(T2-T1) / DBLE(COUNT_RATE) * 1000.0D0 / 10.0D0

      OPEN(10, FILE='bench_dgemm_baseline.bin', FORM='UNFORMATTED',
     $     ACCESS='SEQUENTIAL', STATUS='REPLACE')
      WRITE(10) C
      CLOSE(10)

      WRITE(*,'(A,F12.4,A)') 'FORTRAN_TIME_MS=', ELAPSED, ''
      END PROGRAM
"""

DGEMM_PRECISION_DRIVER = """\
      PROGRAM BENCH_DGEMM_PRECISION
      IMPLICIT NONE
      INTEGER, PARAMETER :: N = 500
      DOUBLE PRECISION :: A(N,N), B(N,N), C(N,N)
      DOUBLE PRECISION :: ALPHA, BETA
      DOUBLE PRECISION, PARAMETER :: EPS = 1.0D-8
      INTEGER :: I, J, ITER
      DOUBLE PRECISION :: RAND_VAL

      ALPHA = 1.0D0
      BETA = 0.0D0
      DO I = 1, N
        DO J = 1, N
          CALL RANDOM_NUMBER(RAND_VAL)
          A(I,J) = 1.0D0 + EPS * RAND_VAL
          CALL RANDOM_NUMBER(RAND_VAL)
          B(I,J) = 1.0D0 + EPS * RAND_VAL
          C(I,J) = 0.0D0
        END DO
      END DO

      DO ITER = 1, 10
        CALL DGEMM('N','N',N,N,N,ALPHA,A,N,B,N,BETA,C,N)
      END DO

      OPEN(10, FILE='bench_dgemm_precision_baseline.bin', FORM='UNFORMATTED',
     $     ACCESS='SEQUENTIAL', STATUS='REPLACE')
      WRITE(10) C
      CLOSE(10)

      WRITE(*,*) 'PRECISION_TEST_DONE'
      END PROGRAM
"""


def _make_generic_driver(fn_name: str, dep_files: list[Path]) -> str:
    """Generate a generic Fortran benchmark driver for a subroutine."""
    n = 500
    return f"""\
      PROGRAM BENCH_{fn_name.upper()}
      IMPLICIT NONE
      INTEGER, PARAMETER :: N = {n}
      DOUBLE PRECISION :: A(N,N), B(N,N), C(N,N)
      DOUBLE PRECISION :: ALPHA, BETA
      INTEGER :: I, J, ITER, COUNT_RATE, T1, T2
      DOUBLE PRECISION :: ELAPSED

      ALPHA = 1.0D0
      BETA = 0.0D0
      DO I = 1, N
        DO J = 1, N
          A(I,J) = DBLE(I+J) / DBLE(N)
          B(I,J) = DBLE(I*J) / DBLE(N*N)
          C(I,J) = 0.0D0
        END DO
      END DO

      CALL SYSTEM_CLOCK(COUNT_RATE=COUNT_RATE)
      CALL SYSTEM_CLOCK(T1)
      DO ITER = 1, 10
        CALL {fn_name.upper()}('N','N',N,N,N,ALPHA,A,N,B,N,BETA,C,N)
      END DO
      CALL SYSTEM_CLOCK(T2)
      ELAPSED = DBLE(T2-T1) / DBLE(COUNT_RATE) * 1000.0D0 / 10.0D0

      OPEN(10, FILE='bench_{fn_name.lower()}_baseline.bin', FORM='UNFORMATTED',
     $     ACCESS='SEQUENTIAL', STATUS='REPLACE')
      WRITE(10) C
      CLOSE(10)

      WRITE(*,'(A,F12.4,A)') 'FORTRAN_TIME_MS=', ELAPSED, ''
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

    for ep in entry_points:
        ep_upper = ep.upper()

        if ep_upper == "DGEMM":
            driver_src = DGEMM_DRIVER
            precision_src = DGEMM_PRECISION_DRIVER
        else:
            driver_src = _make_generic_driver(ep, dep_files)
            precision_src = None

        driver_file = output_dir / f"bench_{ep.lower()}.f"
        driver_file.write_text(driver_src)
        bench_files.append(str(driver_file))

        if precision_src:
            prec_file = output_dir / f"bench_{ep.lower()}_precision.f"
            prec_file.write_text(precision_src)
            bench_files.append(str(prec_file))

        # Build compile command
        fortran_deps = [str(f) for f in dep_files if f.exists()]
        compile_cmd = [
            "gfortran", "-O2",
            str(driver_file),
        ] + fortran_deps + [
            "-lm", "-o", str(output_dir / f"bench_{ep.lower()}")
        ]

        bench_info: dict = {
            "driver_file": str(driver_file),
            "compile_cmd": compile_cmd,
            "compile_ok": False,
            "compile_error": "",
            "time_ms": None,
            "output_file": None,
        }

        try:
            result = subprocess.run(
                compile_cmd,
                capture_output=True,
                text=True,
                cwd=str(output_dir),
                timeout=120,
            )
            if result.returncode == 0:
                bench_info["compile_ok"] = True
                # Run the benchmark
                run_result = subprocess.run(
                    [str(output_dir / f"bench_{ep.lower()}")],
                    capture_output=True,
                    text=True,
                    cwd=str(output_dir),
                    timeout=300,
                )
                stdout = run_result.stdout
                match = re.search(r"FORTRAN_TIME_MS=\s*([\d.]+)", stdout)
                if match:
                    bench_info["time_ms"] = float(match.group(1))

                bin_file = output_dir / f"bench_{ep.lower()}_baseline.bin"
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
    (output_dir / "benchmarks.json").write_text(json.dumps(result_data, indent=2))
    return result_data

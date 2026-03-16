from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from rich.console import Console

from ._log import make_stage_logger

_console = Console(stderr=True)

MINIMAL_F2C_H = """\
/* Minimal f2c.h for compilation */
#ifndef F2C_H
#define F2C_H

typedef int integer;
typedef unsigned int uinteger;
typedef char *address;
typedef short int shortint;
typedef float real;
typedef double doublereal;
typedef struct { real r, i; } complex;
typedef struct { doublereal r, i; } doublecomplex;
typedef long int logical;
typedef short int shortlogical;
typedef char logical1;
typedef char integer1;
typedef long long int longint;
typedef unsigned long long int ulongint;
#define TRUE_ (1)
#define FALSE_ (0)
typedef int flag;
typedef int ftnlen;
typedef int ftnint;

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#define abs(x) ((x) >= 0 ? (x) : -(x))
#define dabs(x) (doublereal)abs(x)
#define min(a,b) ((a) <= (b) ? (a) : (b))
#define max(a,b) ((a) >= (b) ? (a) : (b))
#define dmin(a,b) (doublereal)min(a,b)
#define dmax(a,b) (doublereal)max(a,b)

typedef int (*L_fp)(...);
typedef struct Namelist Namelist;

#endif /* F2C_H */
"""


def _find_or_write_f2c_h(output_dir: Path) -> Path:
    system_f2c = Path("/usr/include/f2c.h")
    dest = output_dir / "f2c.h"
    if system_f2c.exists():
        shutil.copy(system_f2c, dest)
    else:
        dest.write_text(MINIMAL_F2C_H)
    return dest


def run_f2c(source_dir: Path, fortran_files: list[Path], output_dir: Path, status_fn=None) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    _find_or_write_f2c_h(output_dir)
    log = make_stage_logger(output_dir)
    log.info(f"run_f2c: {len(fortran_files)} files to convert")

    c_files: list[str] = []
    errors: list[str] = []
    total = len(fortran_files)

    for i, f in enumerate(fortran_files):
        if not f.exists():
            errors.append(f"File not found: {f}")
            log.warning(f"File not found: {f}")
            continue

        if status_fn:
            status_fn(f"f2c: converting {f.name} ({i+1}/{total})…")

        # Copy .f file to output_dir so f2c writes .c there
        dest_f = output_dir / f.name
        if dest_f != f:
            shutil.copy(f, dest_f)

        result = subprocess.run(
            ["f2c", "-a", dest_f.name],
            capture_output=True,
            text=True,
            cwd=str(output_dir),
            timeout=60,
        )

        # Save full f2c output for this file
        f2c_log = output_dir / f"f2c_{dest_f.stem}.log"
        f2c_log.write_text(
            f"=== COMMAND ===\nf2c -a {dest_f.name}\n\n"
            f"=== STDOUT ===\n{result.stdout}\n"
            f"=== STDERR ===\n{result.stderr}\n"
            f"=== EXIT CODE: {result.returncode} ===\n"
        )

        c_name = dest_f.with_suffix(".c").name
        c_path = output_dir / c_name
        if c_path.exists():
            c_files.append(str(c_path))
            log.info(f"f2c OK: {f.name} -> {c_name}")
        else:
            last_err_line = result.stderr.strip().splitlines()[-1][:100] if result.stderr.strip() else "no output"
            _console.print(
                f"  [yellow]⚠ No C output for[/yellow] [bold]{dest_f.name}[/bold]"
                f" — will be converted by LLM in Stage 4: "
                f"[dim]{last_err_line}[/dim]"
            )
            log.warning(f"f2c produced no C output for {f.name}: {result.stderr.strip()}")
            errors.append(f"f2c did not produce {c_name}: {result.stderr}")

    # Build compile_commands.json
    compile_commands = [
        {
            "file": c,
            "directory": str(output_dir),
            "command": f"gcc -O2 -I{output_dir} -c {c}",
        }
        for c in c_files
    ]
    cc_path = output_dir / "compile_commands.json"
    cc_path.write_text(json.dumps(compile_commands, indent=2))

    # Verify overall compilation
    compile_ok = False
    compile_stdout = ""
    compile_stderr = ""
    if c_files:
        if status_fn:
            status_fn("Verifying C compilation…")
        gcc_cmd = ["gcc", "-O2", f"-I{output_dir}", "-o", "/dev/null"] + c_files + ["-lm", "-lf2c"]
        log.info(f"gcc verify: {' '.join(gcc_cmd)}")
        verify = subprocess.run(
            gcc_cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )
        compile_ok = verify.returncode == 0
        compile_stdout = verify.stdout
        compile_stderr = verify.stderr
        # Save full gcc verification output
        (output_dir / "gcc_verify.log").write_text(
            f"=== COMMAND ===\n{' '.join(gcc_cmd)}\n\n"
            f"=== STDOUT ===\n{compile_stdout}\n"
            f"=== STDERR ===\n{compile_stderr}\n"
            f"=== EXIT CODE: {verify.returncode} ===\n"
        )
        if compile_ok:
            log.info("gcc verify: OK")
        else:
            log.warning(f"gcc verify: FAILED\n{compile_stderr}")

    result_data = {
        "c_files": c_files,
        "compile_commands": compile_commands,
        "compile_ok": compile_ok,
        "compile_stdout": compile_stdout,
        "compile_stderr": compile_stderr,
        "errors": errors,
    }
    (output_dir / "f2c_result.json").write_text(json.dumps(result_data, indent=2))
    log.info("Stage complete")
    return result_data
    return result_data

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from rich.console import Console

if TYPE_CHECKING:
    from ..llm.base import LLMClient

from ..exceptions import CompilationError, MaxRetriesExceededError, NumericalPrecisionError, BenchmarkRuntimeError
from ._log import make_stage_logger

_console = Console(stderr=True)


def _first_error_line(error: str) -> str:
    """Extract the first meaningful error line for display."""
    for line in error.splitlines():
        if "error" in line.lower() and line.strip():
            return line.strip()[:120]
    return error.strip()[:120]


def _topological_sort(entry_points: list[str], call_graph: dict[str, list[str]]) -> list[str]:
    """Return entry_points sorted so that callees come before callers (leaves first).
    Uses Kahn's algorithm on the subgraph induced by entry_points."""
    ep_set = {ep.upper() for ep in entry_points}
    # Build in-degree based on which eps depend on which other eps
    deps: dict[str, set[str]] = {ep.upper(): set() for ep in entry_points}
    for ep in entry_points:
        ep_up = ep.upper()
        for callee in call_graph.get(ep_up, []):
            if callee.upper() in ep_set:
                deps[ep_up].add(callee.upper())

    result: list[str] = []
    ready = [ep for ep, d in deps.items() if not d]
    while ready:
        node = ready.pop(0)
        result.append(node)
        for other, other_deps in deps.items():
            if node in other_deps:
                other_deps.discard(node)
                if not other_deps:
                    ready.append(other)
    # Append any remaining (cycles)
    for ep in entry_points:
        if ep.upper() not in [r.upper() for r in result]:
            result.append(ep.upper())
    return result


def _get_failing_files(error: str, c_dir: Path) -> list[Path]:
    """Parse gcc error output to find the source files containing errors."""
    found: list[Path] = []
    seen: set[str] = set()
    for match in re.finditer(r"(\S+\.c):\d+:\d*:?\s*error:", error):
        fname = Path(match.group(1)).name
        if fname not in seen:
            seen.add(fname)
            candidate = c_dir / fname
            if candidate.exists():
                found.append(candidate)
    # Fallback: return all .c files if we couldn't identify specific ones
    return found if found else sorted(c_dir.glob("*.c"))


def _compile_c(c_dir: Path) -> tuple[bool, str]:
    # Exclude bench_*.c — each has its own main() and is compiled separately
    c_files = sorted(f for f in c_dir.glob("*.c") if not f.name.startswith("bench_"))
    if not c_files:
        return False, "No .c files found"
    result = subprocess.run(
        ["gcc", "-O2", f"-I{c_dir}", "-o", "/dev/null"]
        + [str(f) for f in c_files]
        + ["-lm", "-lf2c"],
        capture_output=True, text=True, timeout=120,
    )
    return result.returncode == 0, result.stdout + result.stderr


def _compile_and_run_bench(c_dir: Path, bench_c: Path, output_dir: Path, dataset_dir: Path) -> tuple[bool, str, Path | None, float | None]:
    """Compile and run a C benchmark driver. Returns (ok, error, output_bin_path, c_time_ms)."""
    c_lib_files = [f for f in sorted(c_dir.glob("*.c")) if "bench_" not in f.name]
    exe = output_dir / ("bench_c_" + bench_c.stem)
    cmd = (
        ["gcc", "-O2", f"-I{c_dir}", str(bench_c)]
        + [str(f) for f in c_lib_files]
        + ["-lm", "-lf2c", "-o", str(exe)]
    )
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    bench_log = output_dir / f"gcc_bench_{bench_c.stem}.log"
    bench_log.write_text(
        f"=== COMPILE COMMAND ===\n{' '.join(cmd)}\n\n"
        f"=== COMPILE STDOUT ===\n{r.stdout}\n"
        f"=== COMPILE STDERR ===\n{r.stderr}\n"
        f"=== COMPILE EXIT CODE: {r.returncode} ===\n"
    )
    if r.returncode != 0:
        return False, r.stdout + r.stderr, None, None

    run = subprocess.run(
        [str(exe)], capture_output=True, text=True,
        cwd=str(dataset_dir),   # run in dataset dir so it finds dataset_*.bin
        timeout=300,
    )
    with open(bench_log, "a") as fh:
        fh.write(
            f"\n=== RUN STDOUT ===\n{run.stdout}\n"
            f"=== RUN STDERR ===\n{run.stderr}\n"
            f"=== RUN EXIT CODE: {run.returncode} ===\n"
        )
    if run.returncode != 0:
        return False, run.stdout + run.stderr, None, None

    c_time_ms: float | None = None
    m = re.search(r"C_TIME_MS=([\d.]+)", run.stdout)
    if m:
        c_time_ms = float(m.group(1))

    # output bin is written to cwd (dataset_dir)
    fn_name = re.sub(r"^bench_", "", bench_c.stem)
    bin_out = dataset_dir / f"bench_{fn_name}_output_c.bin"
    # The C benchmark writes bench_<fn>_output.bin — copy so we don't overwrite Fortran's
    orig = dataset_dir / f"bench_{fn_name}_output.bin"
    if orig.exists():
        shutil.copy(orig, bin_out)
    return True, "", bin_out if bin_out.exists() else None, c_time_ms


def _repair_file(llm: "LLMClient", failing_file: Path, error: str) -> None:
    """Ask LLM to fix one specific file and write it back."""
    response = llm.repair(
        context=(
            "Fix this C file produced by the f2c Fortran-to-C transpiler. "
            "Return ONLY the complete corrected C file contents, no explanation."
        ),
        error=error,
        code=failing_file.read_text(),
    )
    # Strip markdown fences if present
    content = re.sub(r"^```[a-z]*\n?", "", response.strip(), flags=re.MULTILINE)
    content = re.sub(r"\n?```$", "", content.strip())
    failing_file.write_text(content.strip() + "\n")


def _generate_c_from_fortran(llm: "LLMClient", f_file: Path, output_dir: Path) -> Path:
    """Ask the LLM to convert a Fortran file that f2c could not handle into f2c-compatible C."""
    c_path = output_dir / f_file.with_suffix(".c").name
    response = llm.repair(
        context=(
            "Convert this Fortran source file to C using the f2c calling convention. "
            "Rules:\n"
            "- Include \"f2c.h\" at the top\n"
            "- All function names must have a trailing underscore (e.g. xerbla_)\n"
            "- Character arguments must be followed by an extra ftnlen length argument at the end\n"
            "- Use the types from f2c.h: integer, doublereal, real, ftnlen, etc.\n"
            "Return ONLY the complete C file, no explanation, no markdown fences."
        ),
        error="f2c could not convert this file (likely uses Fortran 90+ features like LEN_TRIM).",
        code=f_file.read_text(),
    )
    content = re.sub(r"^```[a-z]*\n?", "", response.strip(), flags=re.MULTILINE)
    content = re.sub(r"\n?```$", "", content.strip())
    c_path.write_text(content.strip() + "\n")
    return c_path


def fix_c_code(
    c_dir: Path,
    output_dir: Path,
    llm: "LLMClient",
    max_retries: int,
    baseline_dir: Path,
    call_graph: dict | None = None,
    entry_points: list[str] | None = None,
    status_fn=None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    log = make_stage_logger(output_dir)
    log.info(f"fix_c_code: c_dir={c_dir}, max_retries={max_retries}")

    for f in c_dir.glob("*.c"):
        shutil.copy(f, output_dir / f.name)
    for f in c_dir.glob("*.h"):
        shutil.copy(f, output_dir / f.name)

    llm_log: list[dict] = []
    llm_turns = 0
    total_retries = 0
    gcc_compile_log = output_dir / "gcc_compile.log"

    # ── Pre-step: LLM-convert any .f files that f2c could not handle ─────────
    for f_file in sorted(c_dir.glob("*.f")):
        c_equiv = output_dir / f_file.with_suffix(".c").name
        if not c_equiv.exists():
            if status_fn:
                status_fn(f"LLM: converting {f_file.name} to C (f2c could not handle it)…")
            log.info(f"LLM converting {f_file.name} to C (f2c could not handle it)")
            dest_f = output_dir / f_file.name
            shutil.copy(f_file, dest_f)
            _generate_c_from_fortran(llm, dest_f, output_dir)
            llm_turns += 1

    # ── Compile loop: fix one failing file at a time ──────────────────────────
    if status_fn:
        status_fn("Compiling C code…")
    compile_ok, compile_output = _compile_c(output_dir)
    compile_error = compile_output  # full output kept for result.json
    with open(gcc_compile_log, "a") as fh:
        fh.write(f"=== INITIAL COMPILE ===\n{compile_output}\n=== EXIT: {'OK' if compile_ok else 'FAIL'} ===\n\n")
    log.info(f"Initial C compile: {'OK' if compile_ok else 'FAILED'}")
    if not compile_ok:
        log.warning(compile_output)

    attempt = 0
    while not compile_ok and attempt < max_retries:
        failing_files = _get_failing_files(compile_output, output_dir)
        target = failing_files[0]
        _console.print(
            f"  [yellow]⚠ C compilation failed[/yellow] in [bold]{target.name}[/bold]: "
            f"[dim]{_first_error_line(compile_output)}[/dim]"
        )
        llm_log.append({
            "phase": "compile", "attempt": attempt,
            "target_file": target.name,
            "error": compile_output,
        })
        if status_fn:
            status_fn(f"LLM: fixing {target.name} (attempt {attempt+1}/{max_retries})…")
        log.info(f"LLM repair attempt {attempt+1}/{max_retries} for {target.name}")
        _repair_file(llm, target, compile_output)
        llm_turns += 1
        total_retries += 1
        compile_ok, compile_output = _compile_c(output_dir)
        compile_error = compile_output
        with open(gcc_compile_log, "a") as fh:
            fh.write(
                f"=== ATTEMPT {attempt+1} (after LLM fix of {target.name}) ===\n"
                f"{compile_output}\n=== EXIT: {'OK' if compile_ok else 'FAIL'} ===\n\n"
            )
        if compile_ok:
            log.info(f"C compile OK after attempt {attempt+1}")
        else:
            log.warning(f"C compile still failing after attempt {attempt+1}")
        attempt += 1

    if not compile_ok:
        exc = CompilationError("C", compile_error)
        raise MaxRetriesExceededError("Stage 4 (fix C)", exc)

    if status_fn:
        status_fn("C compilation successful")
    log.info("C compilation successful")

    # ── Benchmark loop ────────────────────────────────────────────────────────
    bench_ok = False
    bench_results: dict = {}
    if compile_ok:
        bench_c_files = sorted(output_dir.glob("bench_*.c"))
        if not bench_c_files:
            bench_ok = True  # nothing to benchmark
        else:
            all_passed = True
            for bench_c in bench_c_files:
                fn_name = re.sub(r"^bench_", "", bench_c.stem)
                fortran_bin = baseline_dir / f"bench_{fn_name}_output.bin"
                if not fortran_bin.exists():
                    continue  # no baseline to compare against

                if status_fn:
                    status_fn(f"Running C benchmark: {bench_c.stem}…")
                log.info(f"Running C benchmark: {bench_c.stem}")
                ok, err, c_bin, c_time_ms = _compile_and_run_bench(output_dir, bench_c, output_dir, baseline_dir)
                for b_attempt in range(max_retries):
                    if ok and c_bin and c_bin.exists():
                        c_data = np.fromfile(str(c_bin), dtype=np.float64)
                        f_data = np.fromfile(str(fortran_bin), dtype=np.float64)
                        if status_fn:
                            status_fn(f"Comparing {fn_name} output vs Fortran baseline…")
                        if c_data.shape == f_data.shape and np.allclose(c_data, f_data, atol=1e-10, rtol=1e-10):
                            bench_results[fn_name] = {"pass": True, "max_abs_diff": 0.0, "c_time_ms": c_time_ms}
                            log.info(f"  {fn_name}: numerical check PASSED, c_time_ms={c_time_ms}")
                            break
                        max_abs = float(np.max(np.abs(c_data - f_data))) if c_data.shape == f_data.shape else float("inf")
                        bench_results[fn_name] = {"pass": False, "max_abs_diff": max_abs, "c_time_ms": c_time_ms}
                        _console.print(
                            f"  [yellow]⚠ Numerical precision failed[/yellow] for [bold]{fn_name}[/bold]: "
                            f"max diff = [bold]{max_abs:.3e}[/bold]"
                        )
                        log.warning(f"  {fn_name}: numerical check FAILED, max_abs_diff={max_abs:.3e}")
                        err = f"Numerical mismatch: max_abs_diff={max_abs:.6e}"
                    else:
                        if not ok:
                            _console.print(
                                f"  [yellow]⚠ C benchmark failed to run[/yellow] for [bold]{fn_name}[/bold]: "
                                f"[dim]{_first_error_line(err)}[/dim]"
                            )
                            log.warning(f"  {fn_name}: benchmark run failed: {err}")
                    llm_log.append({
                        "phase": "bench", "fn": fn_name, "attempt": b_attempt, "error": err,
                    })
                    if status_fn:
                        status_fn(f"LLM: fixing numerical precision in {bench_c.name} (attempt {b_attempt+1}/{max_retries})…")
                    log.info(f"LLM bench repair attempt {b_attempt+1}/{max_retries} for {bench_c.name}")
                    _repair_file(llm, bench_c, err)
                    llm_turns += 1
                    total_retries += 1
                    ok, err, c_bin, c_time_ms = _compile_and_run_bench(output_dir, bench_c, output_dir, baseline_dir)
                else:
                    all_passed = False
                    # Report what went wrong on final failure
                    last = bench_results.get(fn_name, {})
                    if last.get("max_abs_diff", 0) > 0:
                        exc = NumericalPrecisionError(fn_name, last["max_abs_diff"])
                    elif not ok:
                        exc = BenchmarkRuntimeError(fn_name, err)
                    else:
                        exc = BenchmarkRuntimeError(fn_name, "unknown failure")
                    raise MaxRetriesExceededError("Stage 4 (benchmark)", exc)
            bench_ok = all_passed

    (output_dir / "llm_log.json").write_text(json.dumps(llm_log, indent=2))

    # Generate compile_commands.json with absolute paths for c2rust (Stage 5).
    # Include all .c files — bench drivers are transpiled too so Rust benchmarks can be built.
    lib_c_files = [
        f.resolve() for f in sorted(output_dir.glob("*.c"))
    ]
    compile_commands = [
        {
            "file": str(f),
            "directory": str(output_dir.resolve()),
            "command": f"gcc -O2 -I{output_dir.resolve()} -c {f}",
        }
        for f in lib_c_files
    ]
    cc_path = output_dir / "compile_commands.json"
    cc_path.write_text(json.dumps(compile_commands, indent=2))
    log.info(f"Wrote compile_commands.json with {len(lib_c_files)} entries")

    result = {
        "compile_ok": compile_ok,
        "bench_ok": bench_ok,
        "bench_results": bench_results,
        "llm_turns": llm_turns,
        "retries": total_retries,
        "compile_error": compile_error if not compile_ok else "",
        "compile_commands": str(cc_path),
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2))
    log.info("Stage complete")
    return result


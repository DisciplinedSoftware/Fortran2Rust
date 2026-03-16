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
    c_files = sorted(c_dir.glob("*.c"))
    if not c_files:
        return False, "No .c files found"
    result = subprocess.run(
        ["gcc", "-O2", f"-I{c_dir}", "-o", "/dev/null"]
        + [str(f) for f in c_files]
        + ["-lm", "-lf2c"],
        capture_output=True, text=True, timeout=120,
    )
    return result.returncode == 0, result.stderr


def _compile_and_run_bench(c_dir: Path, bench_c: Path, output_dir: Path, dataset_dir: Path) -> tuple[bool, str, Path | None]:
    """Compile and run a C benchmark driver. Returns (ok, error, output_bin_path)."""
    c_lib_files = [f for f in sorted(c_dir.glob("*.c")) if "bench_" not in f.name]
    exe = output_dir / ("bench_c_" + bench_c.stem)
    cmd = (
        ["gcc", "-O2", f"-I{c_dir}", str(bench_c)]
        + [str(f) for f in c_lib_files]
        + ["-lm", "-lf2c", "-o", str(exe)]
    )
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        return False, r.stderr, None

    run = subprocess.run(
        [str(exe)], capture_output=True, text=True,
        cwd=str(dataset_dir),   # run in dataset dir so it finds dataset_*.bin
        timeout=300,
    )
    if run.returncode != 0:
        return False, run.stderr, None

    # output bin is written to cwd (dataset_dir)
    fn_name = re.sub(r"^bench_", "", bench_c.stem)
    bin_out = dataset_dir / f"bench_{fn_name}_output_c.bin"
    # The C benchmark writes bench_<fn>_output.bin — rename/copy so we don't overwrite Fortran's
    orig = dataset_dir / f"bench_{fn_name}_output.bin"
    if orig.exists():
        shutil.copy(orig, bin_out)
    return True, "", bin_out if bin_out.exists() else None


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

    for f in c_dir.glob("*.c"):
        shutil.copy(f, output_dir / f.name)
    for f in c_dir.glob("*.h"):
        shutil.copy(f, output_dir / f.name)

    # ── Pre-step: LLM-convert any .f files that f2c could not handle ─────────
    for f_file in sorted(c_dir.glob("*.f")):
        c_equiv = output_dir / f_file.with_suffix(".c").name
        if not c_equiv.exists():
            _console.print(
                f"  [yellow]⚠ No C output for[/yellow] [bold]{f_file.name}[/bold] "
                f"— asking LLM to convert from Fortran…"
            )
            if status_fn:
                status_fn(f"LLM: converting {f_file.name} to C (f2c could not handle it)…")
            dest_f = output_dir / f_file.name
            shutil.copy(f_file, dest_f)
            _generate_c_from_fortran(llm, dest_f, output_dir)
            llm_turns += 1


    llm_turns = 0
    total_retries = 0

    # ── Compile loop: fix one failing file at a time ──────────────────────────
    if status_fn:
        status_fn("Compiling C code…")
    compile_ok, compile_error = _compile_c(output_dir)
    attempt = 0
    while not compile_ok and attempt < max_retries:
        failing_files = _get_failing_files(compile_error, output_dir)
        target = failing_files[0]
        _console.print(
            f"  [yellow]⚠ C compilation failed[/yellow] in [bold]{target.name}[/bold]: "
            f"[dim]{_first_error_line(compile_error)}[/dim]"
        )
        llm_log.append({
            "phase": "compile", "attempt": attempt,
            "target_file": target.name,
            "error_snippet": compile_error[:500],
        })
        if status_fn:
            status_fn(f"LLM: fixing {target.name} (attempt {attempt+1}/{max_retries})…")
        _repair_file(llm, target, compile_error)
        llm_turns += 1
        total_retries += 1
        compile_ok, compile_error = _compile_c(output_dir)
        attempt += 1

    if not compile_ok:
        exc = CompilationError("C", compile_error[:500])
        raise MaxRetriesExceededError("Stage 4 (fix C)", exc)

    if status_fn:
        status_fn("C compilation successful")

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
                ok, err, c_bin = _compile_and_run_bench(output_dir, bench_c, output_dir, baseline_dir)
                for b_attempt in range(max_retries):
                    if ok and c_bin and c_bin.exists():
                        c_data = np.fromfile(str(c_bin), dtype=np.float64)
                        f_data = np.fromfile(str(fortran_bin), dtype=np.float64)
                        if status_fn:
                            status_fn(f"Comparing {fn_name} output vs Fortran baseline…")
                        if c_data.shape == f_data.shape and np.allclose(c_data, f_data, atol=1e-10, rtol=1e-10):
                            bench_results[fn_name] = {"pass": True, "max_abs_diff": 0.0}
                            break
                        max_abs = float(np.max(np.abs(c_data - f_data))) if c_data.shape == f_data.shape else float("inf")
                        bench_results[fn_name] = {"pass": False, "max_abs_diff": max_abs}
                        _console.print(
                            f"  [yellow]⚠ Numerical precision failed[/yellow] for [bold]{fn_name}[/bold]: "
                            f"max diff = [bold]{max_abs:.3e}[/bold]"
                        )
                        err = f"Numerical mismatch: max_abs_diff={max_abs:.6e}"
                    else:
                        if not ok:
                            _console.print(
                                f"  [yellow]⚠ C benchmark failed to run[/yellow] for [bold]{fn_name}[/bold]: "
                                f"[dim]{_first_error_line(err)}[/dim]"
                            )
                    llm_log.append({
                        "phase": "bench", "fn": fn_name, "attempt": b_attempt, "error": err[:300],
                    })
                    if status_fn:
                        status_fn(f"LLM: fixing numerical precision in {bench_c.name} (attempt {b_attempt+1}/{max_retries})…")
                    _repair_file(llm, bench_c, err)
                    llm_turns += 1
                    total_retries += 1
                    ok, err, c_bin = _compile_and_run_bench(output_dir, bench_c, output_dir, baseline_dir)
                else:
                    all_passed = False
                    # Report what went wrong on final failure
                    last = bench_results.get(fn_name, {})
                    if last.get("max_abs_diff", 0) > 0:
                        exc = NumericalPrecisionError(fn_name, last["max_abs_diff"])
                    elif not ok:
                        exc = BenchmarkRuntimeError(fn_name, err[:200])
                    else:
                        exc = BenchmarkRuntimeError(fn_name, "unknown failure")
                    raise MaxRetriesExceededError("Stage 4 (benchmark)", exc)
            bench_ok = all_passed

    (output_dir / "llm_log.json").write_text(json.dumps(llm_log, indent=2))
    result = {
        "compile_ok": compile_ok,
        "bench_ok": bench_ok,
        "bench_results": bench_results,
        "llm_turns": llm_turns,
        "retries": total_retries,
        "compile_error": compile_error if not compile_ok else "",
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2))
    return result


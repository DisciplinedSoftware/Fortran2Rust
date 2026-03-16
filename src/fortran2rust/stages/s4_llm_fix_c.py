from __future__ import annotations

import json
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from jinja2 import BaseLoader, Environment
from rich.console import Console

if TYPE_CHECKING:
    from ..llm.base import LLMClient

from ..exceptions import CompilationError, MaxRetriesExceededError, NumericalPrecisionError, BenchmarkRuntimeError
from ._log import make_stage_logger

_console = Console(stderr=True)

_S4_REPORT_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Stage 4 — Fortran vs C Comparison</title>
<style>
:root {
  --primary: #007AC3;
  --navy: #1B3C6E;
  --bg: #F4F7FB;
  --amber: #F0AB00;
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
.pass { color: var(--success); font-weight: bold; }
.fail { color: var(--danger); font-weight: bold; }
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
  <h1>🔬 Stage 4 — Fortran vs C Comparison</h1>
  <p>{{ timestamp }}</p>
</header>

<div class="card">
  <h2>Summary</h2>
  <div class="summary-grid">
    <div class="metric"><div class="value">{{ summary.functions_benchmarked }}</div><div class="label">Functions Benchmarked</div></div>
    <div class="metric"><div class="value">{{ summary.functions_passed }}</div><div class="label">Numerical Checks Passed</div></div>
    <div class="metric"><div class="value">{{ summary.llm_repairs }}</div><div class="label">LLM Repairs</div></div>
    <div class="metric"><div class="value">{{ summary.fortran_loc }}</div><div class="label">Fortran LOC</div></div>
    <div class="metric"><div class="value">{{ summary.c_loc }}</div><div class="label">C LOC</div></div>
  </div>
</div>

{% if bench_rows %}
<div class="card">
  <h2>Numerical Accuracy &amp; Performance</h2>
  <table>
    <tr>
      <th>Function</th>
      <th>Fortran (ms)</th>
      <th>C (ms)</th>
      <th>C / Fortran</th>
      <th>Max |Δ|</th>
      <th>Status</th>
    </tr>
    {% for row in bench_rows %}
    <tr>
      <td><strong>{{ row.function }}</strong></td>
      <td>{{ "%.1f" | format(row.fortran_ms) if row.fortran_ms is not none else '<span class="na">N/A</span>' }}</td>
      <td>{{ "%.1f" | format(row.c_ms) if row.c_ms is not none else '<span class="na">N/A</span>' }}</td>
      <td>
        {% if row.ratio is not none %}
          {% if row.ratio <= 1.05 %}<span class="pass">{{ "%.2fx" | format(row.ratio) }}</span>
          {% elif row.ratio <= 2.0 %}<span style="color:var(--amber);font-weight:bold">{{ "%.2fx" | format(row.ratio) }}</span>
          {% else %}<span class="fail">{{ "%.2fx" | format(row.ratio) }}</span>{% endif %}
        {% else %}<span class="na">N/A</span>{% endif %}
      </td>
      <td>{{ "%.2e" | format(row.max_abs_diff) if row.max_abs_diff is not none else '<span class="na">N/A</span>' }}</td>
      <td>{% if row.pass %}<span class="status-pass">PASS</span>{% else %}<span class="status-fail">FAIL</span>{% endif %}</td>
    </tr>
    {% endfor %}
  </table>
</div>
{% endif %}

{% if loc_rows %}
<div class="card">
  <h2>Code Size: Fortran vs C</h2>
  <table>
    <tr>
      <th>Module</th>
      <th>Fortran LOC</th>
      <th>C LOC</th>
      <th>Expansion</th>
    </tr>
    {% for row in loc_rows %}
    <tr>
      <td>{{ row.name }}</td>
      <td>{{ row.fortran_loc if row.fortran_loc is not none else '<span class="na">—</span>' }}</td>
      <td>{{ row.c_loc if row.c_loc is not none else '<span class="na">—</span>' }}</td>
      <td>{% if row.ratio is not none %}{{ "%.1fx" | format(row.ratio) }}{% else %}<span class="na">—</span>{% endif %}</td>
    </tr>
    {% endfor %}
  </table>
</div>
{% endif %}

</body>
</html>"""

_S4_REPORT_MD = """# Stage 4 — Fortran vs C Comparison

**Generated:** {{ timestamp }}

## Summary

| Metric | Value |
|--------|-------|
| Functions Benchmarked | {{ summary.functions_benchmarked }} |
| Numerical Checks Passed | {{ summary.functions_passed }} |
| LLM Repairs | {{ summary.llm_repairs }} |
| Fortran LOC (total) | {{ summary.fortran_loc }} |
| C LOC (total) | {{ summary.c_loc }} |

{% if bench_rows %}
## Numerical Accuracy & Performance

| Function | Fortran (ms) | C (ms) | C / Fortran | Max \|Δ\| | Status |
|----------|-------------|--------|------------|--------|--------|
{% for row in bench_rows %}| {{ row.function }} | {{ "%.1f" | format(row.fortran_ms) if row.fortran_ms is not none else "N/A" }} | {{ "%.1f" | format(row.c_ms) if row.c_ms is not none else "N/A" }} | {{ "%.2fx" | format(row.ratio) if row.ratio is not none else "N/A" }} | {{ "%.2e" | format(row.max_abs_diff) if row.max_abs_diff is not none else "N/A" }} | {{ "PASS" if row.pass else "FAIL" }} |
{% endfor %}
{% endif %}
{% if loc_rows %}
## Code Size: Fortran vs C

| Module | Fortran LOC | C LOC | Expansion |
|--------|------------|-------|-----------|
{% for row in loc_rows %}| {{ row.name }} | {{ row.fortran_loc if row.fortran_loc is not none else "—" }} | {{ row.c_loc if row.c_loc is not none else "—" }} | {{ "%.1fx" | format(row.ratio) if row.ratio is not none else "—" }} |
{% endfor %}
{% endif %}
"""


def _count_loc(path: Path) -> int:
    """Count non-blank, non-comment source lines."""
    lines = 0
    try:
        for line in path.read_text(errors="replace").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            # Skip Fortran-style comments (! or c/C in column 1) and C-style (//)
            if stripped.startswith("!") or stripped.startswith("//"):
                continue
            if len(line) > 0 and line[0].lower() == "c" and (len(line) == 1 or not line[1].isalpha()):
                continue
            lines += 1
    except Exception:
        pass
    return lines


def _generate_fortran_c_report(
    output_dir: Path,
    fortran_dir: Path,
    baseline_dir: Path,
    bench_results: dict,
    llm_turns: int,
    total_retries: int,
) -> None:
    """Write fortran_c_comparison.{html,md} into output_dir."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Fortran baseline timing from s2 benchmarks.json
    fortran_times: dict[str, float | None] = {}
    bench_json = baseline_dir / "benchmarks.json"
    if bench_json.exists():
        try:
            data = json.loads(bench_json.read_text()).get("benchmarks", {})
            for fn, info in data.items():
                fortran_times[fn] = info.get("time_ms")
        except Exception:
            pass

    # Benchmark comparison rows
    bench_rows = []
    for fn, br in bench_results.items():
        fortran_ms = fortran_times.get(fn)
        c_ms = br.get("c_time_ms")
        max_abs_diff = br.get("max_abs_diff")
        passed = bool(br.get("pass"))
        ratio = (c_ms / fortran_ms) if (c_ms is not None and fortran_ms) else None
        bench_rows.append({
            "function": fn,
            "fortran_ms": fortran_ms,
            "c_ms": c_ms,
            "ratio": ratio,
            "max_abs_diff": max_abs_diff,
            "pass": passed,
        })

    # LOC comparison: match .f files in fortran_dir with .c files in output_dir (exclude bench_*)
    loc_rows = []
    f_files = {f.stem: f for f in sorted(fortran_dir.glob("*.f"))}
    c_files = {f.stem: f for f in sorted(output_dir.glob("*.c")) if not f.name.startswith("bench_")}
    all_stems = sorted(set(f_files) | set(c_files))
    for stem in all_stems:
        f_loc = _count_loc(f_files[stem]) if stem in f_files else None
        c_loc = _count_loc(c_files[stem]) if stem in c_files else None
        ratio = (c_loc / f_loc) if (c_loc is not None and f_loc) else None
        loc_rows.append({"name": stem, "fortran_loc": f_loc, "c_loc": c_loc, "ratio": ratio})

    total_f_loc = sum(r["fortran_loc"] for r in loc_rows if r["fortran_loc"] is not None)
    total_c_loc = sum(r["c_loc"] for r in loc_rows if r["c_loc"] is not None)

    summary = {
        "functions_benchmarked": len(bench_rows),
        "functions_passed": sum(1 for r in bench_rows if r["pass"]),
        "llm_repairs": total_retries,
        "fortran_loc": total_f_loc,
        "c_loc": total_c_loc,
    }

    ctx = {
        "timestamp": timestamp,
        "summary": summary,
        "bench_rows": bench_rows,
        "loc_rows": loc_rows,
    }

    env = Environment(loader=BaseLoader())
    (output_dir / "fortran_c_comparison.html").write_text(env.from_string(_S4_REPORT_HTML).render(**ctx))
    (output_dir / "fortran_c_comparison.md").write_text(env.from_string(_S4_REPORT_MD).render(**ctx))


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
    exe = output_dir / bench_c.stem  # e.g. bench_dgemm (not bench_c_bench_dgemm)
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

    # Copy dataset binaries into output_dir so the bench exe can run from there.
    for ds_file in sorted(dataset_dir.glob("dataset_*.bin")):
        dest = output_dir / ds_file.name
        if not dest.exists():
            shutil.copy(ds_file, dest)

    run = subprocess.run(
        [str(exe.resolve())], capture_output=True, text=True,
        cwd=str(output_dir),  # run from output_dir where datasets are now present
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

    fn_name = re.sub(r"^bench_", "", bench_c.stem)
    # The bench driver writes bench_{fn_name}_output.bin to cwd (output_dir)
    c_bin = output_dir / f"bench_{fn_name}_output.bin"
    return True, "", c_bin if c_bin.exists() else None, c_time_ms


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
    (output_dir / "llm_conversations.json").write_text(
        json.dumps(llm.pop_conversation_log(), indent=2)
    )

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

    if status_fn:
        status_fn("Generating Fortran vs C comparison report…")
    log.info("Generating Fortran vs C comparison report")
    _generate_fortran_c_report(output_dir, c_dir, baseline_dir, bench_results, llm_turns, total_retries)
    log.info("Stage complete")
    return result


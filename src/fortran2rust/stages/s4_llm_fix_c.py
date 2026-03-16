from __future__ import annotations

import json
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from jinja2 import BaseLoader, Environment
from rich.console import Console

if TYPE_CHECKING:
    from ..llm.base import LLMClient

from ..exceptions import CompilationError, MaxRetriesExceededError, NumericalPrecisionError, BenchmarkRuntimeError
from ._llm_cleanup import filter_errors_for_file
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

| Function | Fortran (ms) | C (ms) | C / Fortran | Max \\|Δ\\| | Status |
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
    """Compile C sources individually without linking.

    This isolates syntax/type issues in each file and avoids link-time noise from
    unresolved external symbols in unrelated translation units.
    """
    c_files = sorted(f for f in c_dir.glob("*.c") if not f.name.startswith("bench_"))
    if not c_files:
        return False, "No .c files found"

    all_ok = True
    chunks: list[str] = []
    for c_file in c_files:
        cmd = ["gcc", "-O2", f"-I{c_dir}", "-c", str(c_file), "-o", "/dev/null"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        chunks.append(
            f"=== COMPILE {c_file.name} ===\n"
            f"COMMAND: {' '.join(cmd)}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}\n"
            f"EXIT: {result.returncode}\n"
        )
        if result.returncode != 0:
            all_ok = False

    return all_ok, "\n".join(chunks)


def _find_fortran_source_for_function(src_dir: Path, fn_name: str) -> str | None:
    """Given a function name, find and extract its Fortran source from the library.

    Returns the Fortran source code if found, None otherwise.
    """
    # Try to find .f files in the source directory
    for f_file in src_dir.glob("*.f"):
        try:
            content = f_file.read_text(errors="ignore")
            # Look for a subroutine or function definition with this name
            # Case-insensitive search
            pattern = rf"(?:subroutine|function)\s+{re.escape(fn_name)}\b"
            if re.search(pattern, content, re.IGNORECASE):
                return content
        except Exception:
            pass

    return None


def _is_non_repairable_bench_error(error: str) -> bool:
    """Return True for benchmark failures that LLM source edits cannot fix."""
    text = (error or "").lower()
    markers = [
        "undefined reference to",
        "collect2: error",
        "ld returned 1 exit status",
        "cannot find -l",
        "cannot find /",
    ]
    return any(marker in text for marker in markers)


_C_DEF_RE = re.compile(
    r"^\s*(?:/\*[^*]*\*/\s*)?(?:void|int|char|double|float|long|short|signed|unsigned|logical|doublereal|real)\s+([a-z][a-z0-9_]*)\s*\(",
    flags=re.IGNORECASE,
)


def _defined_c_symbols(c_file: Path) -> set[str]:
    symbols: set[str] = set()
    for line in c_file.read_text(errors="replace").splitlines():
        m = _C_DEF_RE.match(line)
        if not m:
            continue
        sym = m.group(1).upper()
        if sym.endswith("_"):
            sym = sym[:-1]
        symbols.add(sym)
    return symbols


def _collect_bench_closure_symbols(fn_name: str, call_graph: dict | None) -> set[str]:
    root = fn_name.upper()
    graph = {str(k).upper(): [str(c).upper() for c in (v or [])] for k, v in (call_graph or {}).items()}
    required = {root}
    queue = [root]
    while queue:
        sym = queue.pop(0)
        for callee in graph.get(sym, []):
            if callee not in required:
                required.add(callee)
                queue.append(callee)
    return required


def _select_bench_lib_c_files(c_dir: Path, fn_name: str, call_graph: dict | None) -> list[Path]:
    c_lib_files = [f for f in sorted(c_dir.glob("*.c")) if "bench_" not in f.name]
    if not c_lib_files:
        return []

    symbol_to_files: dict[str, list[Path]] = {}
    for c_file in c_lib_files:
        for sym in _defined_c_symbols(c_file):
            symbol_to_files.setdefault(sym, []).append(c_file)

    required_symbols = _collect_bench_closure_symbols(fn_name, call_graph)
    selected: list[Path] = []
    seen: set[Path] = set()
    for sym in required_symbols:
        for c_file in symbol_to_files.get(sym, []):
            if c_file not in seen:
                seen.add(c_file)
                selected.append(c_file)

    if not selected:
        fallback = c_dir / f"{fn_name.lower()}.c"
        if fallback.exists():
            selected = [fallback]

    if not selected:
        selected = c_lib_files

    return selected


def _compile_and_run_bench(
    c_dir: Path,
    bench_c: Path,
    output_dir: Path,
    dataset_dir: Path,
    call_graph: dict | None = None,
) -> tuple[bool, str, Path | None, float | None]:
    """Compile and run a C benchmark driver. Returns (ok, error, output_bin_path, c_time_ms)."""
    fn_name = re.sub(r"^bench_", "", bench_c.stem)
    c_lib_files = _select_bench_lib_c_files(c_dir, fn_name, call_graph)
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

    # The bench driver writes bench_{fn_name}_output.bin to cwd (output_dir)
    c_bin = output_dir / f"bench_{fn_name}_output.bin"
    return True, "", c_bin if c_bin.exists() else None, c_time_ms


def _write_compile_commands(output_dir: Path) -> Path:
    """Write compile_commands.json for non-benchmark C sources and return its path."""
    lib_c_files = [
        f.resolve() for f in sorted(output_dir.glob("*.c")) if not f.name.startswith("bench_")
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
    return cc_path


_C_CODE_LINE_RE = re.compile(
    r"^(#\s*include\b|/\*|//|typedef\b|extern\b|static\b|struct\b|union\b|enum\b|"
    r"(void|int|char|double|float|long|short|signed|unsigned|integer|doublereal|real|"
    r"logical|ftnlen)\b)"
)

_FORTRAN_UNIT_RE = re.compile(
    r"^\s*(?:[a-z][a-z0-9_()*,\s]*\s+)?(function|subroutine|program)\s+([a-z][a-z0-9_]*)\b",
    flags=re.IGNORECASE,
)


def _looks_like_c_code(content: str) -> bool:
    return bool(
        re.search(
            r"#\s*include\b|;\s*$|\{\s*$|\breturn\b|\bstatic\b|\bextern\b|\btypedef\b",
            content,
            flags=re.MULTILINE,
        )
    )


def _has_balanced_c_braces(content: str) -> bool:
    depth = 0
    in_single = False
    in_double = False
    escape = False
    in_line_comment = False
    in_block_comment = False
    prev = ""

    for ch in content:
        if in_line_comment:
            if ch == "\n":
                in_line_comment = False
            prev = ch
            continue
        if in_block_comment:
            if prev == "*" and ch == "/":
                in_block_comment = False
            prev = ch
            continue
        if in_single:
            if not escape and ch == "'":
                in_single = False
            escape = (ch == "\\" and not escape)
            prev = ch
            continue
        if in_double:
            if not escape and ch == '"':
                in_double = False
            escape = (ch == "\\" and not escape)
            prev = ch
            continue

        if prev == "/" and ch == "/":
            in_line_comment = True
            prev = ch
            continue
        if prev == "/" and ch == "*":
            in_block_comment = True
            prev = ch
            continue
        if ch == "'":
            in_single = True
            escape = False
            prev = ch
            continue
        if ch == '"':
            in_double = True
            escape = False
            prev = ch
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth < 0:
                return False

        prev = ch

    return depth == 0 and not in_single and not in_double and not in_block_comment


def _is_plausibly_complete_c_rewrite(original: str, candidate: str) -> bool:
    """Guardrail for full-file rewrites to avoid accepting truncated LLM output."""
    if not candidate.strip():
        return False
    if not _has_balanced_c_braces(candidate):
        return False

    original_lines = [ln for ln in original.splitlines() if ln.strip()]
    candidate_lines = [ln for ln in candidate.splitlines() if ln.strip()]
    if not original_lines:
        return True

    # For medium/large files, require near-complete size retention.
    large_file = len(original) > 20_000 or len(original_lines) > 500
    if large_file:
        if len(candidate) < int(len(original) * 0.90):
            return False
        if len(candidate_lines) < int(len(original_lines) * 0.90):
            return False

    # Keep a light tail-anchor check so dropped suffixes are rejected.
    tail = "\n".join(original_lines[-8:]).strip()
    if tail and len(tail) > 20 and tail not in candidate:
        # Permit tail variation on small files where a full rewrite is plausible.
        if large_file:
            return False

    return True


def _extract_c_from_llm_response(response: str) -> str:
    text = (response or "").strip()
    if not text:
        return ""

    fenced_blocks = re.findall(r"```(?:c|C)?\s*\n(.*?)```", text, flags=re.DOTALL)
    if not fenced_blocks:
        fenced_blocks = re.findall(r"```[^\n`]*\n(.*?)```", text, flags=re.DOTALL)
    candidate = max(fenced_blocks, key=len).strip() if fenced_blocks else text

    lines = candidate.splitlines()
    start_idx = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if _C_CODE_LINE_RE.match(stripped):
            start_idx = i
            break
    if start_idx > 0:
        candidate = "\n".join(lines[start_idx:])

    candidate = re.sub(r"\n?```\s*$", "", candidate.strip())
    return candidate.strip()


def _normalize_f2c_include_order(content: str) -> str:
    """Ensure #include "f2c.h" appears after system includes.

    f2c.h defines macros like abs/min/max that can conflict with declarations in
    headers such as stdlib.h if f2c.h is included first.
    """
    lines = content.splitlines()
    first_include = next((i for i, line in enumerate(lines) if line.strip().startswith("#include")), None)
    if first_include is None:
        return content.strip()

    end = first_include
    while end < len(lines):
        stripped = lines[end].strip()
        if not stripped or stripped.startswith("#include"):
            end += 1
            continue
        break

    block = lines[first_include:end]
    f2c_includes = [line for line in block if "f2c.h" in line]
    if not f2c_includes:
        return content.strip()

    others = [line for line in block if "f2c.h" not in line]
    reordered = others + f2c_includes
    if reordered == block:
        return content.strip()

    lines[first_include:end] = reordered
    return "\n".join(lines).strip()


def _fortran_defined_units(path: Path) -> set[str]:
    """Extract defined Fortran unit names from a source file."""
    units: set[str] = set()
    try:
        for raw in path.read_text(errors="replace").splitlines():
            stripped = raw.strip()
            if not stripped:
                continue
            if stripped.startswith(("*", "!")):
                continue
            match = _FORTRAN_UNIT_RE.match(stripped)
            if not match:
                continue
            units.add(match.group(2).upper())
    except Exception:
        return set()
    return units


def _extract_required_fortran_units(path: Path, required_syms: set[str]) -> str:
    """Extract only required Fortran units from a source file.

    This keeps Stage 4 LLM conversion prompts/output small for monolithic files
    that contain large PROGRAM bodies plus many routines.
    """
    try:
        lines = path.read_text(errors="replace").splitlines()
    except Exception:
        return path.read_text(errors="replace")

    units: list[tuple[int, int, str, str]] = []
    starts: list[tuple[int, str, str]] = []  # (line_idx, kind, name)
    for i, raw in enumerate(lines):
        stripped = raw.strip()
        if not stripped or stripped.startswith(("*", "!")):
            continue
        match = _FORTRAN_UNIT_RE.match(stripped)
        if not match:
            continue
        kind = match.group(1).lower()
        name = match.group(2).upper()
        starts.append((i, kind, name))

    if not starts:
        return "\n".join(lines)

    for idx, (start, kind, name) in enumerate(starts):
        end = starts[idx + 1][0] if idx + 1 < len(starts) else len(lines)
        units.append((start, end, kind, name))

    # Keep any required symbol unit; if none match, keep non-program units as a fallback.
    keep: list[tuple[int, int, str, str]] = [u for u in units if u[3] in required_syms]
    if not keep:
        keep = [u for u in units if u[2] != "program"]
    if not keep:
        keep = units

    header_end = units[0][0]
    out_lines: list[str] = []
    if header_end > 0:
        out_lines.extend(lines[:header_end])
        out_lines.append("")

    for start, end, _, _ in keep:
        out_lines.extend(lines[start:end])
        out_lines.append("")

    return "\n".join(out_lines).strip() + "\n"


def _required_symbols(entry_points: list[str] | None, call_graph: dict | None) -> set[str]:
    """Compute a conservative required-symbol set from entry points and call graph."""
    required = {ep.upper() for ep in (entry_points or [])}
    graph = {str(k).upper(): [str(c).upper() for c in (v or [])] for k, v in (call_graph or {}).items()}
    queue = list(required)
    while queue:
        sym = queue.pop(0)
        for callee in graph.get(sym, []):
            if callee not in required:
                required.add(callee)
                queue.append(callee)
    return required


def _normalize_f2c_includes_in_dir(c_dir: Path) -> None:
    """Normalize include ordering in all C files in a directory."""
    for c_file in sorted(c_dir.glob("*.c")):
        original = c_file.read_text(errors="replace")
        normalized = _normalize_f2c_include_order(original)
        if normalized != original.strip():
            c_file.write_text(normalized + "\n")


def _compact_c_for_llm(code: str) -> tuple[str, str]:
    lines = code.splitlines()
    banner_lines: list[str] = []
    body_start = 0
    in_banner_comment = False

    for i, line in enumerate(lines):
        stripped = line.strip()
        if in_banner_comment:
            banner_lines.append(line)
            if "*/" in stripped:
                in_banner_comment = False
            body_start = i + 1
            continue
        if not stripped:
            banner_lines.append(line)
            body_start = i + 1
            continue
        if stripped.startswith("/*"):
            banner_lines.append(line)
            body_start = i + 1
            if "*/" not in stripped:
                in_banner_comment = True
            continue
        break

    compact_lines: list[str] = []
    in_block_comment = False
    for line in lines[body_start:]:
        stripped = line.strip()
        if in_block_comment:
            if "*/" in stripped:
                in_block_comment = False
            continue
        if stripped.startswith("/*"):
            if "*/" not in stripped:
                in_block_comment = True
            continue
        if stripped.startswith("//"):
            continue
        if not stripped:
            continue
        compact_lines.append(line.rstrip())

    compact_code = "\n".join(compact_lines).strip()
    preserved_banner = "\n".join(banner_lines).strip()
    return compact_code, preserved_banner


def _restore_c_after_llm(content: str, preserved_banner: str) -> str:
    restored = content.strip()
    if not preserved_banner:
        return restored
    if restored.startswith("/*"):
        return restored
    return f"{preserved_banner}\n\n{restored}".strip()


def _repair_file(
    llm: "LLMClient",
    failing_file: Path,
    error: str,
    attempt: int = 0,
    fortran_source: str | None = None,
) -> bool:
    """Ask LLM to fix one specific file and write it back.

    Returns True when file content changed, False otherwise.
    """
    original = failing_file.read_text()
    compact_code, preserved_banner = _compact_c_for_llm(original)

    # Build context with optional Fortran reference
    context = (
        "Fix this C file produced by the f2c Fortran-to-C transpiler. "
        "Leading documentation comments were removed before sending to reduce token usage. "
        "Return ONLY the complete corrected C file contents, no explanation."
    )
    if fortran_source:
        context += (
            "\n\nFor reference, here is the original Fortran code that this C file implements:\n"
            f"{fortran_source}"
        )

    response = llm.repair(
        context=context,
        error=error,
        code=compact_code,
        attempt=attempt,
    )
    content = _extract_c_from_llm_response(response)
    if not _looks_like_c_code(content):
        content = original
    else:
        content = _restore_c_after_llm(content, preserved_banner)
        content = _normalize_f2c_include_order(content)
        if not _is_plausibly_complete_c_rewrite(original, content):
            content = original
    final_content = content.strip() + "\n"
    changed = final_content != original
    failing_file.write_text(final_content)
    return changed


def _generate_c_from_fortran(
    llm: "LLMClient",
    f_file: Path,
    output_dir: Path,
    fortran_source: str | None = None,
) -> Path:
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
        code=fortran_source if fortran_source is not None else f_file.read_text(),
    )
    content = _extract_c_from_llm_response(response)
    if not _looks_like_c_code(content):
        raise CompilationError("C", "LLM conversion did not return recognizable C code")
    if not _has_balanced_c_braces(content):
        raise CompilationError("C", "LLM conversion appears truncated (unbalanced braces)")
    content = _normalize_f2c_include_order(content)
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

    _normalize_f2c_includes_in_dir(output_dir)

    llm_log: list[dict] = []
    llm_turns = 0
    total_retries = 0
    gcc_compile_log = output_dir / "gcc_compile.log"

    # ── Pre-step: LLM-convert any .f files that f2c could not handle ─────────
    required_syms = _required_symbols(entry_points, call_graph)
    f_candidates = [
        f_file for f_file in sorted(c_dir.glob("*.f"))
        if not (output_dir / f_file.with_suffix(".c").name).exists()
    ]
    skipped_irrelevant_units: list[Path] = []
    f_files_to_convert: list[Path] = []
    for f_file in f_candidates:
        defined = _fortran_defined_units(f_file)
        if required_syms and defined and required_syms.isdisjoint(defined):
            skipped_irrelevant_units.append(f_file)
            continue
        f_files_to_convert.append(f_file)

    if skipped_irrelevant_units:
        log.info(
            "Skipping LLM conversion for .f file(s) that define no required symbols: "
            + ", ".join(p.name for p in skipped_irrelevant_units)
        )
    if f_files_to_convert:
        if status_fn:
            status_fn(f"LLM: converting {len(f_files_to_convert)} Fortran file(s) to C (parallel)…")

        def _convert_one(f_file: Path) -> None:
            log.info(f"LLM converting {f_file.name} to C (f2c could not handle it)")
            dest_f = output_dir / f_file.name
            shutil.copy(f_file, dest_f)
            selected_source = _extract_required_fortran_units(dest_f, required_syms)
            _generate_c_from_fortran(llm, dest_f, output_dir, fortran_source=selected_source)

        with ThreadPoolExecutor(max_workers=len(f_files_to_convert)) as executor:
            list(executor.map(_convert_one, f_files_to_convert))

        llm_turns += len(f_files_to_convert)

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
    no_progress_rounds = 0
    while not compile_ok and attempt < max_retries:
        failing_files = _get_failing_files(compile_output, output_dir)
        _console.print(
            f"  [yellow]⚠ C compilation failed[/yellow] in "
            f"[bold]{', '.join(f.name for f in failing_files)}[/bold]: "
            f"[dim]{_first_error_line(compile_output)}[/dim]"
        )

        # Fix all currently-failing files in parallel, then recompile once.
        def _fix_one(target: Path) -> bool:
            llm_log.append({
                "phase": "compile", "attempt": attempt,
                "target_file": target.name,
                "error": compile_output,
            })
            if status_fn:
                status_fn(f"LLM: fixing {target.name} (attempt {attempt+1}/{max_retries})…")
            log.info(f"LLM repair attempt {attempt+1}/{max_retries} for {target.name}")

            # Try to find the corresponding Fortran source file
            # C files from f2c often have similar names (e.g., dgemm.c from dgemm.f)
            fortran_src = None
            stem = target.stem
            for ext in [".f", ".F"]:
                f_file = c_dir / (stem + ext)
                if f_file.exists():
                    fortran_src = f_file.read_text(errors="ignore")
                    break

            return _repair_file(
                llm,
                target,
                filter_errors_for_file(compile_output, target.name),
                attempt=attempt,
                fortran_source=fortran_src,
            )

        with ThreadPoolExecutor(max_workers=len(failing_files)) as executor:
            changed_flags = list(executor.map(_fix_one, failing_files))

        if not any(changed_flags):
            no_progress_rounds += 1
            log.warning(
                "No effective source changes after LLM repair round %d for file(s): %s",
                attempt + 1,
                ", ".join(f.name for f in failing_files),
            )
        else:
            no_progress_rounds = 0

        llm_turns += len(failing_files)
        total_retries += len(failing_files)
        compile_ok, compile_output = _compile_c(output_dir)
        compile_error = compile_output
        with open(gcc_compile_log, "a") as fh:
            fh.write(
                f"=== ATTEMPT {attempt+1} (after LLM fix of "
                f"{[f.name for f in failing_files]}) ===\n"
                f"{compile_output}\n=== EXIT: {'OK' if compile_ok else 'FAIL'} ===\n\n"
            )
        if compile_ok:
            log.info(f"C compile OK after attempt {attempt+1}")
        else:
            log.warning(f"C compile still failing after attempt {attempt+1}")

        # Cost guard: stop early when repeated LLM rounds produce no effective edits.
        if not compile_ok and no_progress_rounds >= 2:
            compile_error = (
                compile_output
                + "\n\nStage 4 aborted early: LLM repairs made no effective source changes "
                + "for two consecutive rounds."
            )
            break
        attempt += 1

    if not compile_ok:
        exc = CompilationError("C", compile_error)
        raise MaxRetriesExceededError("Stage 4 (fix C)", exc)

    if status_fn:
        status_fn("C compilation successful")
    log.info("C compilation successful")

    # Write compile_commands.json as soon as C compiles, so Stage 5 can proceed
    # even if benchmark validation later fails.
    cc_path = _write_compile_commands(output_dir)
    log.info("Wrote compile_commands.json")

    # ── Benchmark loop ────────────────────────────────────────────────────────
    bench_ok = False
    bench_results: dict = {}
    if compile_ok:
        bench_c_files = sorted(f for f in output_dir.glob("bench_*.c") if not f.stem.endswith("_precision"))
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
                ok, err, c_bin, c_time_ms = _compile_and_run_bench(
                    output_dir,
                    bench_c,
                    output_dir,
                    baseline_dir,
                    call_graph=call_graph,
                )
                skip_fn_llm_repairs = False
                bench_succeeded = False
                for b_attempt in range(max_retries):
                    if ok and c_bin and c_bin.exists():
                        c_data = np.fromfile(str(c_bin), dtype=np.float64)
                        f_data = np.fromfile(str(fortran_bin), dtype=np.float64)
                        if status_fn:
                            status_fn(f"Comparing {fn_name} output vs Fortran baseline…")
                        if c_data.shape == f_data.shape and np.allclose(c_data, f_data, atol=1e-10, rtol=1e-10):
                            bench_results[fn_name] = {"pass": True, "max_abs_diff": 0.0, "c_time_ms": c_time_ms}
                            log.info(f"  {fn_name}: numerical check PASSED, c_time_ms={c_time_ms}")
                            bench_succeeded = True
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
                            if _is_non_repairable_bench_error(err):
                                bench_results[fn_name] = {
                                    "pass": False,
                                    "max_abs_diff": None,
                                    "c_time_ms": None,
                                    "reason": "non-repairable link/dependency error",
                                }
                                all_passed = False
                                skip_fn_llm_repairs = True
                                log.warning(
                                    "  %s: skipping LLM retries due to non-repairable benchmark error",
                                    fn_name,
                                )
                                break
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

                    # Try to find the original Fortran source for context
                    fortran_src = _find_fortran_source_for_function(c_dir, fn_name)
                    _repair_file(
                        llm,
                        bench_c,
                        filter_errors_for_file(err, bench_c.name),
                        attempt=b_attempt,
                        fortran_source=fortran_src,
                    )
                    llm_turns += 1
                    total_retries += 1
                    ok, err, c_bin, c_time_ms = _compile_and_run_bench(
                        output_dir,
                        bench_c,
                        output_dir,
                        baseline_dir,
                        call_graph=call_graph,
                    )
                if skip_fn_llm_repairs:
                    bench_results[fn_name] = {
                        "pass": False,
                        "max_abs_diff": None,
                        "c_time_ms": None,
                        "reason": "non-repairable link/dependency error",
                    }
                    continue
                if not bench_succeeded:
                    all_passed = False
                    # Report what went wrong on final failure
                    last = bench_results.get(fn_name, {})
                    if last.get("max_abs_diff", 0) > 0:
                        exc = NumericalPrecisionError(fn_name, last["max_abs_diff"])
                    elif not ok:
                        exc = BenchmarkRuntimeError(fn_name, err)
                    else:
                        exc = BenchmarkRuntimeError(fn_name, "unknown failure")
                    bench_results[fn_name] = {
                        "pass": False,
                        "max_abs_diff": last.get("max_abs_diff"),
                        "c_time_ms": last.get("c_time_ms"),
                        "reason": str(exc),
                    }
                    log.warning(f"  {fn_name}: benchmark validation failed after retries: {exc}")
            bench_ok = all_passed

    (output_dir / "llm_log.json").write_text(json.dumps(llm_log, indent=2))
    (output_dir / "llm_conversations.json").write_text(
        json.dumps(llm.pop_conversation_log(), indent=2)
    )

    # Refresh compile_commands after potential benchmark-file LLM edits.
    cc_path = _write_compile_commands(output_dir)
    log.info("Wrote compile_commands.json")

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


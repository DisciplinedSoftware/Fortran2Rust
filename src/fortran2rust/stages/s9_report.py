from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from jinja2 import BaseLoader, Environment

from ._log import make_stage_logger

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Fortran2Rust Report — {{ run_id }}</title>
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
.perf-ratio { color: var(--primary); font-weight: bold; }
.warn { color: var(--amber); font-weight: bold; }
.summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1rem; }
.metric { text-align: center; padding: 1rem; background: var(--bg); border-radius: 6px; }
.metric .value { font-size: 2rem; font-weight: bold; color: var(--primary); }
.metric .label { font-size: 0.8rem; color: #666; margin-top: 0.25rem; }
.status-pass { background: #e8f7ef; color: var(--success); padding: 0.2rem 0.6rem; border-radius: 4px; font-weight: bold; }
.status-fail { background: #fde8eb; color: var(--danger); padding: 0.2rem 0.6rem; border-radius: 4px; font-weight: bold; }
</style>
</head>
<body>
<header>
  <h1>🦀 Fortran2Rust Pipeline Report</h1>
  <p>Run ID: <strong>{{ run_id }}</strong> &nbsp;|&nbsp; {{ timestamp }}</p>
</header>

<div class="card">
  <h2>Summary</h2>
  <div class="summary-grid">
    <div class="metric"><div class="value">{{ summary.total_functions }}</div><div class="label">Functions Converted</div></div>
    <div class="metric"><div class="value">{{ summary.stages_completed }}/{{ summary.stages_total }}</div><div class="label">Stages Completed</div></div>
    <div class="metric"><div class="value">{{ summary.llm_turns_total }}</div><div class="label">LLM Turns Used</div></div>
    <div class="metric"><div class="value"><span class="{{ 'pass' if summary.overall_ok else 'fail' }}">{{ 'PASS' if summary.overall_ok else 'FAIL' }}</span></div><div class="label">Overall Status</div></div>
  </div>
</div>

{% if perf_table %}
<div class="card">
  <h2>Performance Comparison</h2>
  <table>
    <tr><th>Function</th><th>Fortran (ms)</th><th>Rust (ms)</th><th>Speedup</th></tr>
    {% for row in perf_table %}
    <tr>
      <td>{{ row.function }}</td>
      <td>{{ "%.4f" | format(row.fortran_ms) if row.fortran_ms is not none else "N/A" }}</td>
      <td>{{ "%.4f" | format(row.rust_ms) if row.rust_ms is not none else "N/A" }}</td>
      <td>{% if row.speedup is not none %}<span class="perf-ratio">{{ "%.2fx" | format(row.speedup) }}</span>{% else %}N/A{% endif %}</td>
    </tr>
    {% endfor %}
  </table>
</div>
{% endif %}

{% if precision_table %}
<div class="card">
  <h2>Numerical Precision</h2>
  <table>
    <tr><th>Function</th><th>Max Abs Diff</th><th>Max Rel Diff</th><th>Status</th></tr>
    {% for row in precision_table %}
    <tr>
      <td>{{ row.function }}</td>
      <td>{{ "%.2e" | format(row.max_abs_diff) if row.max_abs_diff is not none else "N/A" }}</td>
      <td>{{ "%.2e" | format(row.max_rel_diff) if row.max_rel_diff is not none else "N/A" }}</td>
      <td><span class="{{ 'status-pass' if row.ok else 'status-fail' }}">{{ 'PASS' if row.ok else 'FAIL' }}</span></td>
    </tr>
    {% endfor %}
  </table>
</div>
{% endif %}

<div class="card">
  <h2>Stage Log</h2>
  <table>
    <tr><th>Stage</th><th>Name</th><th>Status</th><th>LLM Turns</th><th>Notes</th></tr>
    {% for stage in stage_log %}
    <tr>
      <td>{{ stage.num }}</td>
      <td>{{ stage.name }}</td>
      <td><span class="{{ 'status-pass' if stage.ok else 'status-fail' }}">{{ 'PASS' if stage.ok else 'FAIL' }}</span></td>
      <td>{{ stage.llm_turns }}</td>
      <td>{{ stage.notes }}</td>
    </tr>
    {% endfor %}
  </table>
</div>

</body>
</html>"""

MD_TEMPLATE = """# Fortran2Rust Pipeline Report

**Run ID:** {{ run_id }}
**Timestamp:** {{ timestamp }}

## Summary

| Metric | Value |
|--------|-------|
| Functions Converted | {{ summary.total_functions }} |
| Stages Completed | {{ summary.stages_completed }}/{{ summary.stages_total }} |
| LLM Turns Used | {{ summary.llm_turns_total }} |
| Overall Status | {{ 'PASS' if summary.overall_ok else 'FAIL' }} |

{% if perf_table %}
## Performance Comparison

| Function | Fortran (ms) | Rust (ms) | Speedup |
|----------|-------------|-----------|---------|
{% for row in perf_table %}| {{ row.function }} | {{ "%.4f" | format(row.fortran_ms) if row.fortran_ms is not none else "N/A" }} | {{ "%.4f" | format(row.rust_ms) if row.rust_ms is not none else "N/A" }} | {{ "%.2fx" | format(row.speedup) if row.speedup is not none else "N/A" }} |
{% endfor %}
{% endif %}

{% if precision_table %}
## Numerical Precision

| Function | Max Abs Diff | Max Rel Diff | Status |
|----------|-------------|-------------|--------|
{% for row in precision_table %}| {{ row.function }} | {{ "%.2e" | format(row.max_abs_diff) if row.max_abs_diff is not none else "N/A" }} | {{ "%.2e" | format(row.max_rel_diff) if row.max_rel_diff is not none else "N/A" }} | {{ 'PASS' if row.ok else 'FAIL' }} |
{% endfor %}
{% endif %}

## Stage Log

| Stage | Name | Status | LLM Turns | Notes |
|-------|------|--------|-----------|-------|
{% for stage in stage_log %}| {{ stage.num }} | {{ stage.name }} | {{ 'PASS' if stage.ok else 'FAIL' }} | {{ stage.llm_turns }} | {{ stage.notes }} |
{% endfor %}
"""

STAGE_NAMES = {
    1: "Dependency Analysis",
    2: "Benchmark Generation",
    3: "Fortran → C (f2c)",
    4: "LLM Fix C",
    5: "C → Rust (c2rust)",
    6: "LLM Fix Rust",
    7: "LLM: Make Safe",
    8: "LLM: Make Idiomatic",
    9: "Report Generation",
}


def _collect_stage_log(run_dir: Path, stage_results: dict) -> list[dict]:
    log = []
    for num, name in STAGE_NAMES.items():
        stage_result = stage_results.get(num, {})
        ok = "error" not in stage_result
        llm_turns = stage_result.get("llm_turns", 0)
        notes = stage_result.get("error", "") if not ok else ""
        log.append({"num": num, "name": name, "ok": ok, "llm_turns": llm_turns, "notes": str(notes)[:100]})
    return log


def generate_report(run_dir: Path, config: dict, status_fn=None) -> Path:
    run_id = config.get("run_id", run_dir.name)
    entry_points = config.get("entry_points", [])
    stage_results = config.get("stage_results", {})
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Use the s9 stage directory (created by pipeline.py before calling this function)
    s9_dirs = [d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("s9_")]
    log_dir = s9_dirs[0] if s9_dirs else run_dir
    log = make_stage_logger(log_dir)
    log.info(f"generate_report: run_id={run_id}, entry_points={entry_points}")

    if status_fn:
        status_fn("Collecting benchmark results…")
    log.info("Collecting benchmark results")

    # Collect benchmark data
    bench_data = {}
    s2_dirs = [d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("s2_")]
    if s2_dirs:
        bench_json = s2_dirs[0] / "benchmarks.json"
        if bench_json.exists():
            try:
                bench_data = json.loads(bench_json.read_text()).get("benchmarks", {})
            except Exception:
                pass

    perf_table = []
    precision_table = []
    for ep in entry_points:
        ep_data = bench_data.get(ep, {})
        fortran_ms = ep_data.get("time_ms")
        perf_table.append({
            "function": ep,
            "fortran_ms": fortran_ms,
            "rust_ms": None,
            "speedup": None,
        })
        precision_table.append({
            "function": ep,
            "max_abs_diff": None,
            "max_rel_diff": None,
            "ok": True,
        })

    stage_log = _collect_stage_log(run_dir, stage_results)
    stages_completed = sum(1 for s in stage_log if s["ok"])
    llm_turns_total = sum(s["llm_turns"] for s in stage_log)
    overall_ok = all(s["ok"] for s in stage_log)
    log.info(f"Stages completed: {stages_completed}/{len(STAGE_NAMES)}, LLM turns: {llm_turns_total}, overall: {'PASS' if overall_ok else 'FAIL'}")

    summary = {
        "total_functions": len(entry_points),
        "stages_completed": stages_completed,
        "stages_total": len(STAGE_NAMES),
        "llm_turns_total": llm_turns_total,
        "overall_ok": overall_ok,
    }

    log.info("Rendering reports")

    ctx = {
        "run_id": run_id,
        "timestamp": timestamp,
        "summary": summary,
        "perf_table": perf_table,
        "precision_table": precision_table,
        "stage_log": stage_log,
    }

    env = Environment(loader=BaseLoader())
    html_tmpl = env.from_string(HTML_TEMPLATE)
    md_tmpl = env.from_string(MD_TEMPLATE)

    html_path = run_dir / "report.html"
    md_path = run_dir / "report.md"

    if status_fn:
        status_fn("Rendering HTML report…")
    log.info("Rendering HTML report")
    html_path.write_text(html_tmpl.render(**ctx))
    if status_fn:
        status_fn("Rendering Markdown report…")
    log.info("Rendering Markdown report")
    md_path.write_text(md_tmpl.render(**ctx))

    log.info("Stage complete")
    return html_path

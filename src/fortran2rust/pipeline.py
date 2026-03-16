from __future__ import annotations

from pathlib import Path
from typing import Callable

from rich.console import Console

from .config import Config, make_run_id


def run_pipeline(config: Config, library_path: Path, entry_points: list[str]) -> Path:
    """Run the full conversion pipeline. Returns the run directory."""
    console = Console()
    run_id = make_run_id()
    run_dir = config.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"\n[bold blue]Fortran2Rust Pipeline[/bold blue] — run [cyan]{run_id}[/cyan]")
    console.print(f"Library: [green]{library_path}[/green]")
    console.print(f"Entry points: [yellow]{', '.join(entry_points)}[/yellow]\n")

    from .llm import get_llm_client
    llm = get_llm_client(
        provider=config.llm_provider,
        model=config.llm_model,
        openai_api_key=config.openai_api_key,
        anthropic_api_key=config.anthropic_api_key,
        google_api_key=config.google_api_key,
        openrouter_api_key=config.openrouter_api_key,
        ollama_base_url=config.ollama_base_url,
    )

    results: dict = {}

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

    with console.status("") as _pipeline_status:
        for stage_num in config.stages:
            if stage_num not in STAGE_NAMES:
                continue
            stage_slug = (
                STAGE_NAMES[stage_num]
                .lower()
                .replace(" ", "_")
                .replace("→", "to")
                .replace("(", "")
                .replace(")", "")
                .replace(":", "")
                .strip()
            )
            stage_dir = run_dir / f"s{stage_num}_{stage_slug}"
            stage_dir.mkdir(parents=True, exist_ok=True)

            console.rule(f"[bold]Stage {stage_num}: {STAGE_NAMES[stage_num]}[/bold]")

            # Build a status_fn that prefixes messages with the stage number
            def _make_status_fn(sn: int) -> Callable[[str], None]:
                def _fn(msg: str) -> None:
                    _pipeline_status.update(f"[bold cyan]Stage {sn}:[/bold cyan] {msg}")
                return _fn

            status_fn = _make_status_fn(stage_num)
            status_fn("Starting…")

            try:
                if stage_num == 1:
                    from .stages.s1_analyze import analyze_dependencies
                    results[1] = analyze_dependencies(library_path, entry_points, stage_dir, status_fn=status_fn)

                elif stage_num == 2:
                    from .stages.s2_benchmarks import generate_benchmarks
                    dep_files = [Path(f) for f in results.get(1, {}).get("files", [])]
                    results[2] = generate_benchmarks(library_path, entry_points, dep_files, stage_dir, status_fn=status_fn)

                elif stage_num == 3:
                    from .stages.s3_f2c import run_f2c
                    import shutil as _shutil
                    dep_files = [Path(f) for f in results.get(1, {}).get("files", [])]
                    # Benchmark .f files use modern Fortran (Fortran 90/2003) which f2c cannot handle.
                    # C benchmark drivers were generated directly in stage 2 — copy them into stage 3.
                    bench_c_files = [Path(f) for f in results.get(2, {}).get("bench_c_files", [])]
                    for bc in bench_c_files:
                        if bc.exists():
                            _shutil.copy(bc, stage_dir / bc.name)
                    results[3] = run_f2c(library_path, dep_files, stage_dir, status_fn=status_fn)

                elif stage_num == 4:
                    from .stages.s4_llm_fix_c import fix_c_code
                    s3_dirs = [d for d in run_dir.iterdir() if d.name.startswith("s3_")]
                    s3_dir = run_dir / s3_dirs[0].name
                    s2_dirs = [d for d in run_dir.iterdir() if d.name.startswith("s2_")]
                    s2_dir = run_dir / s2_dirs[0].name
                    cg = results.get(1, {}).get("call_graph", {})
                    results[4] = fix_c_code(s3_dir, stage_dir, llm, config.max_retries, s2_dir,
                                            call_graph=cg, entry_points=entry_points, status_fn=status_fn)

                elif stage_num == 5:
                    from .stages.s5_c2rust import ensure_c2rust, transpile_to_rust
                    ensure_c2rust(status_fn=status_fn)
                    s4_dirs = [d for d in run_dir.iterdir() if d.name.startswith("s4_")]
                    s4_dir = run_dir / s4_dirs[0].name
                    results[5] = transpile_to_rust(s4_dir, s4_dir / "compile_commands.json", stage_dir, status_fn=status_fn)

                elif stage_num == 6:
                    from .stages.s6_llm_fix_rust import fix_rust_code
                    s5_dirs = [d for d in run_dir.iterdir() if d.name.startswith("s5_")]
                    s5_dir = run_dir / s5_dirs[0].name
                    s2_dirs = [d for d in run_dir.iterdir() if d.name.startswith("s2_")]
                    s2_dir = run_dir / s2_dirs[0].name
                    results[6] = fix_rust_code(s5_dir, stage_dir, llm, config.max_retries, s2_dir, status_fn=status_fn)

                elif stage_num == 7:
                    from .stages.s7_llm_safe import make_safe
                    s6_dirs = [d for d in run_dir.iterdir() if d.name.startswith("s6_")]
                    s6_dir = run_dir / s6_dirs[0].name
                    s2_dirs = [d for d in run_dir.iterdir() if d.name.startswith("s2_")]
                    s2_dir = run_dir / s2_dirs[0].name
                    results[7] = make_safe(s6_dir, stage_dir, llm, config.max_retries, s2_dir, status_fn=status_fn)

                elif stage_num == 8:
                    from .stages.s8_llm_idiomatic import make_idiomatic
                    s7_dirs = [d for d in run_dir.iterdir() if d.name.startswith("s7_")]
                    s7_dir = run_dir / s7_dirs[0].name
                    s2_dirs = [d for d in run_dir.iterdir() if d.name.startswith("s2_")]
                    s2_dir = run_dir / s2_dirs[0].name
                    results[8] = make_idiomatic(s7_dir, stage_dir, llm, config.max_retries, s2_dir, status_fn=status_fn)

                elif stage_num == 9:
                    from .stages.s9_report import generate_report
                    results[9] = generate_report(run_dir, {
                        "run_id": run_id,
                        "entry_points": entry_points,
                        "config": config.__dict__,
                        "stage_results": results,
                    }, status_fn=status_fn)

                console.print(f"  [green]✓ Stage {stage_num} complete[/green]")

                if stage_num == 4:
                    bench_results = results[4].get("bench_results", {})
                    if bench_results:
                        fortran_times = {
                            fn: results.get(2, {}).get("benchmarks", {}).get(fn, {}).get("time_ms")
                            for fn in bench_results
                        }
                        parts = []
                        for fn, br in bench_results.items():
                            diff = br.get("max_abs_diff", 0.0)
                            diff_str = f"Δ={diff:.2e}"
                            c_ms = br.get("c_time_ms")
                            f_ms = fortran_times.get(fn)
                            if c_ms is not None and f_ms:
                                ratio = c_ms / f_ms
                                timing_str = f"C:{c_ms:.1f}ms, {ratio:.2f}x Fortran"
                            elif c_ms is not None:
                                timing_str = f"C:{c_ms:.1f}ms"
                            else:
                                timing_str = None
                            detail = f"{diff_str}, {timing_str}" if timing_str else diff_str
                            if br.get("pass"):
                                parts.append(f"[green]{fn}[/green] ✓ ({detail})")
                            else:
                                parts.append(f"[red]{fn}[/red] ✗ ({detail})")
                        console.print(f"  Benchmarks: {' | '.join(parts)}")

            except Exception as e:
                console.print(f"  [red]✗ Stage {stage_num} failed: {e}[/red]")
                results[stage_num] = {"error": str(e)}

    report_path = run_dir / "report.html"
    # Only open preview if stage 9 actually ran and produced the report
    if 9 in results and "error" not in results[9] and report_path.exists():
        import subprocess as _sp
        try:
            _sp.run(["code", "--reuse-window", str(report_path)], check=False, timeout=5)
            _sp.run(["code", "--reuse-window", str(run_dir / "report.md")], check=False, timeout=5)
        except Exception:
            pass
    console.print(f"\n[bold green]Pipeline complete![/bold green] Report: [cyan]{report_path}[/cyan]\n")
    return run_dir

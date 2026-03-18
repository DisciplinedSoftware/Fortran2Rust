from __future__ import annotations

from pathlib import Path
from typing import Callable
import shutil

from rich.console import Console

from .config import Config, make_run_id


def _export_run_folder(run_dir: Path) -> None:
    """Collect compiled executables and datasets into run_dir/run/ for easy access."""
    import stat as _stat

    run_folder = run_dir / "run"
    run_folder.mkdir(exist_ok=True)

    def _is_compiled_executable(p: Path) -> bool:
        if not p.is_file():
            return False
        if p.suffix in {".c", ".f", ".rs", ".json", ".log", ".toml", ".lock",
                        ".d", ".dot", ".rlib", ".a", ".so", ".txt", ".md", ".html"}:
            return False
        return bool(p.stat().st_mode & _stat.S_IXUSR)

    # Stage 2: Fortran benchmarks + datasets
    s2_dirs = sorted(run_dir.glob("s2_*"))
    if s2_dirs:
        s2_dir = s2_dirs[0]
        for exe in sorted(s2_dir.iterdir()):
            if _is_compiled_executable(exe):
                shutil.copy2(exe, run_folder / f"fortran_{exe.name}")
        for ds in sorted(s2_dir.glob("dataset_*.bin")):
            shutil.copy2(ds, run_folder / ds.name)
        for jf in ("benchmarks.json", "datasets.json"):
            src = s2_dir / jf
            if src.exists():
                shutil.copy2(src, run_folder / jf)

    # Stage 4: C benchmarks
    s4_dirs = sorted(run_dir.glob("s4_*"))
    if s4_dirs:
        for exe in sorted(s4_dirs[0].iterdir()):
            if _is_compiled_executable(exe):
                shutil.copy2(exe, run_folder / f"c_{exe.name}")

    # Rust benchmarks: prefer latest completed stage (s8 → s7 → s6)
    for prefix in ("s8_", "s7_", "s6_"):
        rust_dirs = sorted(run_dir.glob(f"{prefix}*"))
        if not rust_dirs:
            continue
        target_release = rust_dirs[0] / "target" / "release"
        if not target_release.is_dir():
            continue
        found = False
        for exe in sorted(target_release.iterdir()):
            if _is_compiled_executable(exe) and exe.name.startswith("bench_"):
                shutil.copy2(exe, run_folder / f"rust_{exe.name}")
                found = True
        if found:
            break

    (run_folder / "README.md").write_text(
        "# Benchmark Executables\n\n"
        "Compiled benchmarks and datasets from the Fortran2Rust pipeline.\n\n"
        "## Executables\n\n"
        "| Prefix | Language | Stage |\n"
        "|--------|----------|-------|\n"
        "| `fortran_*` | Fortran (gfortran) | s2 baseline |\n"
        "| `c_*` | C (gcc) | s4 LLM-fixed C |\n"
        "| `rust_*` | Rust (cargo release) | s6–s8 (best available) |\n\n"
        "## How to run\n\n"
        "Run benchmarks **from this directory** so they find the dataset files:\n\n"
        "```bash\n"
        "cd run/\n"
        "./rust_bench_<function>     # Rust benchmark\n"
        "./fortran_bench_<function>  # Fortran baseline\n"
        "./c_bench_<function>        # C benchmark\n"
        "```\n\n"
        "Each binary prints `RUST_TIME_MS=...` (or `C_TIME_MS=...` / `FORTRAN_TIME_MS=...`) "
        "to stdout and writes a `bench_<fn>_output.bin` result file.\n\n"
        "## Datasets\n\n"
        "`dataset_*.bin` files are raw binary arrays (little-endian float64, column-major) "
        "shared across all three languages for reproducible numerical comparison.\n"
    )


def _export_final_rust_to_run_root(run_dir: Path, rust_stage_dir: Path) -> None:
    cargo_src = rust_stage_dir / "Cargo.toml"
    src_src = rust_stage_dir / "src"
    if not cargo_src.exists() or not src_src.exists():
        return

    cargo_dst = run_dir / "Cargo.toml"
    src_dst = run_dir / "src"

    shutil.copy2(cargo_src, cargo_dst)
    if src_dst.exists():
        shutil.rmtree(src_dst)
    shutil.copytree(src_src, src_dst)


def _evaluate_stage_result(stage_num: int, stage_result: dict) -> tuple[bool, str]:
    if not stage_result:
        if stage_num == 9:
            return True, ""
        return False, "no stage result"

    if not isinstance(stage_result, dict):
        if stage_num == 9 and isinstance(stage_result, Path):
            return True, ""
        return True, ""

    if "error" in stage_result:
        return False, str(stage_result.get("error", ""))

    notes: list[str] = []

    if stage_num == 2:
        benchmarks = stage_result.get("benchmarks", {})
        if benchmarks:
            compile_flags = [
                bool(info.get("compile_ok"))
                for info in benchmarks.values()
                if isinstance(info, dict)
            ]
            if compile_flags and not any(compile_flags):
                notes.append("Fortran baselines did not compile")
            run_failures = [
                fn for fn, info in benchmarks.items()
                if isinstance(info, dict) and info.get("compile_ok") and not bool(info.get("run_ok", False))
            ]
            if run_failures:
                notes.append("Fortran baseline run failures")
            missing_outputs = [
                fn for fn, info in benchmarks.items()
                if isinstance(info, dict) and info.get("compile_ok") and info.get("run_ok") and not bool(info.get("output_ok", False))
            ]
            if missing_outputs:
                notes.append("Fortran baseline outputs missing")

    if stage_num == 4:
        if stage_result.get("compile_ok") is False:
            notes.append("compile failed")
        if stage_result.get("bench_ok") is False:
            notes.append("benchmark checks failed")
        bench_results = stage_result.get("bench_results")
        if isinstance(bench_results, dict) and not bench_results:
            notes.append("no C benchmark comparisons produced")

    if stage_num == 5 and stage_result.get("ok") is False:
        notes.append("c2rust transpile incomplete")

    if stage_num in (6, 7, 8):
        bench_results = stage_result.get("bench_results")
        if isinstance(bench_results, dict):
            if not bench_results:
                notes.append("no Rust benchmark results")
            elif any(not bool(br.get("run_ok")) for br in bench_results.values()):
                notes.append("Rust benchmark run failures")
            elif any(not bool(br.get("pass", False)) for br in bench_results.values()):
                notes.append("Rust numerical checks failed")

    return (len(notes) == 0), "; ".join(notes)


def _blocking_stage_reason(stage_num: int, results: dict) -> str | None:
    if stage_num == 5:
        stage4 = results.get(4, {})
        if stage4.get("error"):
            return "Skipped because Stage 4 failed"
        if stage4.get("compile_ok") is False:
            return "Skipped because Stage 4 did not produce compilable C output"

    if stage_num == 6 and results.get(5, {}).get("error"):
        return "Skipped because Stage 5 failed"

    if stage_num == 7 and results.get(6, {}).get("error"):
        return "Skipped because Stage 6 failed"

    if stage_num == 8 and results.get(7, {}).get("error"):
        return "Skipped because Stage 7 failed"

    return None


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
        llm_max_tokens=config.llm_max_tokens,
        openai_api_key=config.openai_api_key,
        anthropic_api_key=config.anthropic_api_key,
        google_api_key=config.google_api_key,
        openrouter_api_key=config.openrouter_api_key,
        github_token=config.github_token,
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

    stage_issues: list[tuple[int, str]] = []
    aborted_reason: str | None = None

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

            blocking_reason = _blocking_stage_reason(stage_num, results)
            if blocking_reason:
                results[stage_num] = {"error": blocking_reason}
                stage_issues.append((stage_num, blocking_reason))
                console.print(f"  [yellow]↷ Stage {stage_num} skipped: {blocking_reason}[/yellow]")
                continue

            try:
                if stage_num == 1:
                    from .stages.s1_analyze import analyze_dependencies
                    results[1] = analyze_dependencies(library_path, entry_points, stage_dir, status_fn=status_fn)

                elif stage_num == 2:
                    from .stages.s2_benchmarks import generate_benchmarks
                    dep_files = [Path(f) for f in results.get(1, {}).get("files", [])]
                    call_graph = results.get(1, {}).get("call_graph", {})
                    results[2] = generate_benchmarks(
                        library_path,
                        entry_points,
                        dep_files,
                        stage_dir,
                        call_graph=call_graph,
                        max_parallel=config.s2_max_parallel,
                        matrix_n_max=config.s2_matrix_n_max,
                        timing_max_runs=config.s2_timing_max_runs,
                        dataset_reuse_every=config.s2_dataset_reuse_every,
                        status_fn=status_fn,
                    )

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
                    results[7] = make_safe(
                        s6_dir,
                        stage_dir,
                        llm,
                        config.max_retries,
                        s2_dir,
                        llm_max_parallel=config.llm_max_parallel,
                        status_fn=status_fn,
                    )

                elif stage_num == 8:
                    from .stages.s8_llm_idiomatic import make_idiomatic
                    s7_dirs = [d for d in run_dir.iterdir() if d.name.startswith("s7_")]
                    s7_dir = run_dir / s7_dirs[0].name
                    s2_dirs = [d for d in run_dir.iterdir() if d.name.startswith("s2_")]
                    s2_dir = run_dir / s2_dirs[0].name
                    results[8] = make_idiomatic(
                        s7_dir,
                        stage_dir,
                        llm,
                        config.max_retries,
                        s2_dir,
                        llm_max_parallel=config.llm_max_parallel,
                        status_fn=status_fn,
                    )
                    # Keep final Rust crate easy to consume from the run root.
                    _export_final_rust_to_run_root(run_dir, stage_dir)

                elif stage_num == 9:
                    from .stages.s9_report import generate_report
                    results[9] = generate_report(run_dir, {
                        "run_id": run_id,
                        "entry_points": entry_points,
                        "config": config.__dict__,
                        "stage_results": results,
                    }, status_fn=status_fn)

                stage_ok, stage_notes = _evaluate_stage_result(stage_num, results.get(stage_num, {}))
                if stage_ok:
                    console.print(f"  [green]✓ Stage {stage_num} complete[/green]")
                else:
                    stage_issues.append((stage_num, stage_notes))
                    details = f": {stage_notes}" if stage_notes else ""
                    console.print(
                        f"  [yellow]⚠ Stage {stage_num} completed with issues{details}[/yellow]"
                    )

                if stage_num == 4:
                    bench_results = results[4].get("bench_results", {})
                    if bench_results:
                        fortran_times = {
                            fn: results.get(2, {}).get("benchmarks", {}).get(fn, {}).get("time_ms")
                            for fn in bench_results
                        }
                        parts = []
                        for fn, br in bench_results.items():
                            diff = br.get("max_abs_diff")
                            diff_str = f"Δ={diff:.2e}" if diff is not None else "Δ=?"
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

                elif stage_num in (6, 7, 8):
                    bench_results = results[stage_num].get("bench_results", {})
                    if bench_results:
                        fortran_times = {
                            fn: results.get(2, {}).get("benchmarks", {}).get(fn, {}).get("time_ms")
                            for fn in bench_results
                        }
                        parts = []
                        for fn, br in bench_results.items():
                            diff = br.get("max_abs_diff")
                            diff_str = f"Δ={diff:.2e}" if diff is not None else "Δ=?"
                            r_ms = br.get("time_ms")
                            f_ms = fortran_times.get(fn)
                            if r_ms is not None and f_ms:
                                timing_str = f"Rust:{r_ms:.1f}ms, {r_ms/f_ms:.2f}x Fortran"
                            elif r_ms is not None:
                                timing_str = f"Rust:{r_ms:.1f}ms"
                            else:
                                timing_str = None
                            detail = f"{diff_str}, {timing_str}" if timing_str else diff_str
                            if br.get("pass"):
                                parts.append(f"[green]{fn}[/green] ✓ ({detail})")
                            elif br.get("run_ok"):
                                parts.append(f"[yellow]{fn}[/yellow] ⚠ (numerical mismatch: {detail})")
                            else:
                                parts.append(f"[red]{fn}[/red] ✗ (run failed)")
                        console.print(f"  Benchmarks: {' | '.join(parts)}")

            except Exception as e:
                console.print(f"  [red]✗ Stage {stage_num} failed: {e}[/red]")
                results[stage_num] = {"error": str(e)}
                if stage_num == 2:
                    console.print(
                        "  [bold red]Aborting pipeline:[/bold red] "
                        "Stage 2 Fortran benchmark executable did not run correctly."
                    )
                    aborted_reason = "Stage 2 Fortran benchmark executable did not run correctly"
                    break

    _export_run_folder(run_dir)

    report_path = run_dir / "report.html"
    if report_path.exists():
        import subprocess as _sp
        try:
            _sp.run(["xdg-open", str(report_path)], check=False, timeout=5)
        except Exception:
            pass
    run_folder = run_dir / "run"
    if aborted_reason:
        console.print(
            f"\n[bold red]Pipeline aborted[/bold red]. Report: [cyan]{report_path}[/cyan]\n"
            f"Executables & datasets: [cyan]{run_folder}[/cyan]\n"
        )
        raise RuntimeError(aborted_reason)
    if stage_issues:
        console.print(
            f"\n[bold yellow]Pipeline finished with issues[/bold yellow] "
            f"([yellow]{len(stage_issues)} stage(s) affected[/yellow]). "
            f"Report: [cyan]{report_path}[/cyan]\n"
            f"Executables & datasets: [cyan]{run_folder}[/cyan]\n"
        )
    else:
        console.print(
            f"\n[bold green]Pipeline complete![/bold green] Report: [cyan]{report_path}[/cyan]\n"
            f"Executables & datasets: [cyan]{run_folder}[/cyan]\n"
        )
    return run_dir

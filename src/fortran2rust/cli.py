from __future__ import annotations

import argparse
from pathlib import Path

from .config import PROVIDERS, load_config


def main():
    parser = argparse.ArgumentParser(
        prog="fortran2rust",
        description="Convert Fortran code to safe, idiomatic Rust",
    )
    parser.add_argument(
        "-i", "--non-interactive",
        action="store_true",
        help="Skip menus and run from CLI args",
    )
    parser.add_argument(
        "--library",
        metavar="PATH|blas",
        help="Path to Fortran library directory, or 'blas' to auto-download BLAS demo source",
    )
    parser.add_argument(
        "--entry-points",
        metavar="NAMES|all",
        help="Comma-separated function names or 'all'",
    )
    parser.add_argument(
        "--stages",
        type=int,
        default=None,
        metavar="N",
        choices=range(1, 10),
        help="Run all stages from 1 through N; in non-interactive mode defaults to 9",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Skip LLM stages 7 (make safe) and 8 (make idiomatic) to reduce token usage",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=None,
        help="LLM repair retries for stages 4,6,7,8",
    )
    parser.add_argument(
        "--llm-provider",
        choices=PROVIDERS,
        metavar="PROVIDER",
        help="LLM provider: " + "|".join(PROVIDERS),
    )
    parser.add_argument("--model", metavar="NAME", help="Override LLM model name")
    parser.add_argument(
        "--output-dir",
        metavar="PATH",
        default="./artifacts",
        help="Artifact output directory (default: ./artifacts)",
    )

    args = parser.parse_args()

    if args.non_interactive:
        _run_non_interactive(args)
    else:
        _run_interactive(args)


def _run_non_interactive(args):
    from rich.console import Console

    from .blas import get_blas_source
    from .pipeline import run_pipeline
    from .stages.s1_analyze import list_entry_points

    console = Console()
    console.print("[bold blue]Fortran2Rust[/bold blue] — non-interactive mode")

    if args.library == "blas":
        library_path = get_blas_source(console)
    elif args.library:
        library_path = Path(args.library).expanduser().resolve()
    else:
        console.print("[dim]No --library provided; defaulting to BLAS demo source[/dim]")
        library_path = get_blas_source(console)

    if args.entry_points:
        if args.entry_points == "all":
            entry_points = list_entry_points(library_path)
        else:
            entry_points = [e.strip() for e in args.entry_points.split(",") if e.strip()]
    else:
        entry_points = ["dgemm"]

    stage_limit = args.stages or 9
    overrides = {
        "stages": [s for s in range(1, stage_limit + 1) if not args.quick or s not in (7, 8)],
        "output_dir": Path(args.output_dir),
    }
    if args.max_retries is not None:
        overrides["max_retries"] = args.max_retries
    if args.llm_provider:
        overrides["llm_provider"] = args.llm_provider
    if args.model:
        overrides["llm_model"] = args.model

    config = load_config(**overrides)
    run_pipeline(config, library_path, entry_points)


def _run_interactive(args):
    from .menu import run_interactive_menu
    from .pipeline import run_pipeline

    library_path, entry_points, config = run_interactive_menu()

    if args.library:
        if args.library == "blas":
            from .blas import get_blas_source
            library_path = get_blas_source()
        else:
            library_path = Path(args.library).expanduser().resolve()
    if args.entry_points:
        if args.entry_points == "all":
            from .stages.s1_analyze import list_entry_points
            entry_points = list_entry_points(library_path)
        else:
            entry_points = [e.strip() for e in args.entry_points.split(",")]

    if args.stages is not None:
        config.stages = [s for s in range(1, args.stages + 1) if not args.quick or s not in (7, 8)]
    elif args.quick:
        config.stages = [s for s in config.stages if s not in (7, 8)]
    if args.max_retries is not None:
        config.max_retries = args.max_retries
    config.output_dir = Path(args.output_dir)
    if args.llm_provider:
        config.llm_provider = args.llm_provider
    if args.model:
        config.llm_model = args.model

    run_pipeline(config, library_path, entry_points)


if __name__ == "__main__":
    main()

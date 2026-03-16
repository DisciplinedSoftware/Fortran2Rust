from __future__ import annotations

import argparse
from pathlib import Path

from .config import PROVIDERS, load_config


def parse_stages(s: str) -> list[int]:
    stages = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            stages.extend(range(int(a), int(b) + 1))
        else:
            stages.append(int(part))
    return sorted(set(stages))


def main():
    parser = argparse.ArgumentParser(
        prog="fortran2rust",
        description="Convert Fortran code to safe, idiomatic Rust",
    )
    parser.add_argument(
        "-i", "--non-interactive",
        action="store_true",
        help="Skip all menus; auto-convert dgemm from BLAS using .env config",
    )
    parser.add_argument(
        "--library",
        metavar="PATH|blas",
        help="Path to Fortran library directory, or 'blas' to auto-download BLAS",
    )
    parser.add_argument(
        "--entry-points",
        metavar="NAMES|all",
        help="Comma-separated function names or 'all'",
    )
    parser.add_argument(
        "--stages",
        default="1-9",
        help="Stages to run, e.g. '1-9' or '1,3,5' (default: 1-9)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Skip LLM stages 7 (make safe) and 8 (make idiomatic) to reduce token usage",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="LLM repair retries for stages 4,6,7,8 (default: 5)",
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

    console = Console()
    console.print("[bold blue]Fortran2Rust[/bold blue] — non-interactive mode (dgemm demo)")

    library_path = get_blas_source(console)
    entry_points = ["dgemm"]

    overrides = {
        "max_retries": args.max_retries,
        "stages": [s for s in parse_stages(args.stages) if not args.quick or s not in (7, 8)],
        "output_dir": Path(args.output_dir),
    }
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

    run_pipeline(config, library_path, entry_points)


if __name__ == "__main__":
    main()

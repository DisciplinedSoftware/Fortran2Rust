from __future__ import annotations

from pathlib import Path

import questionary
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from .config import PROVIDERS, Config, load_config

DEFAULT_MODELS = {
    "openai": "gpt-4o",
    "anthropic": "claude-opus-4-5",
    "google": "gemini-1.5-pro",
    "openrouter": "anthropic/claude-opus-4",
    "ollama": "llama3.1",
}


def run_interactive_menu() -> tuple[Path, list[str], Config]:
    """Run the interactive menu and return (library_path, entry_points, config)."""
    console = Console()
    console.print(Panel(
        Text("Fortran2Rust Pipeline", style="bold white", justify="center"),
        style="bold blue",
        padding=(1, 4),
    ))

    lib_choice = questionary.select(
        "Select library source:",
        choices=[
            "BLAS (auto-download from netlib)",
            "Enter a custom path",
        ],
    ).ask()

    if lib_choice == "BLAS (auto-download from netlib)":
        from .blas import get_blas_source
        library_path = get_blas_source(console)
    else:
        path_str = questionary.path("Enter path to Fortran library directory:").ask()
        library_path = Path(path_str).expanduser().resolve()

    ep_choice = questionary.select(
        "Select entry points:",
        choices=[
            "List all and select interactively",
            "Convert entire library (all entry points)",
            "Enter function names manually",
        ],
    ).ask()

    if ep_choice == "List all and select interactively":
        from .stages.s1_analyze import list_entry_points
        with console.status("[bold blue]Scanning Fortran files…") as spin:
            all_eps = list_entry_points(
                library_path,
                status_fn=lambda msg: spin.update(f"[bold blue]{msg}"),
            )
        if not all_eps:
            console.print("[yellow]No entry points found.[/yellow]")
            entry_points: list[str] = []
        else:
            entry_points = questionary.checkbox(
                "Select entry points to convert:",
                choices=all_eps,
            ).ask() or []
    elif ep_choice == "Convert entire library (all entry points)":
        from .stages.s1_analyze import list_entry_points
        with console.status("[bold blue]Scanning Fortran files…") as spin:
            entry_points = list_entry_points(
                library_path,
                status_fn=lambda msg: spin.update(f"[bold blue]{msg}"),
            )
    else:
        names = questionary.text("Enter function names (comma-separated):").ask()
        entry_points = [n.strip() for n in names.split(",") if n.strip()]

    existing = load_config()
    provider = questionary.select(
        "Select LLM provider:",
        choices=PROVIDERS,
        default=existing.llm_provider if existing.llm_provider in PROVIDERS else PROVIDERS[0],
    ).ask()

    default_model = existing.llm_model or DEFAULT_MODELS.get(provider, "")
    model = questionary.text(f"Model name [{default_model}]:", default=default_model).ask()

    max_retries = int(questionary.text("Max LLM retries per stage [5]:", default="5").ask() or "5")

    all_stages = list(range(1, 10))
    _STAGE_NAMES = {
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
    last_stage = questionary.select(
        "Run up to stage:",
        choices=[
            questionary.Choice(f"Stage {i}: {_STAGE_NAMES[i]}", value=i)
            for i in all_stages
        ],
        default=9,
    ).ask() or 9
    stages = list(range(1, last_stage + 1))

    config = load_config(
        llm_provider=provider,
        llm_model=model,
        max_retries=max_retries,
        stages=stages,
    )

    console.print(f"\n[bold]Configuration:[/bold] {provider} / {model} / retries={max_retries} / stages 1–{last_stage}\n")
    confirm = questionary.confirm("Start pipeline?", default=True).ask()
    if not confirm:
        console.print("[yellow]Cancelled.[/yellow]")
        raise SystemExit(0)

    return library_path, entry_points, config

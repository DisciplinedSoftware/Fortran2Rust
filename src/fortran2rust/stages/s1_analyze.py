from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path

logger = logging.getLogger(__name__)


def _parse_file(path: Path):
    from fparser.two.parser import ParserFactory
    from fparser.common.readfortran import FortranFileReader

    reader = FortranFileReader(str(path), ignore_comments=True)
    parser = ParserFactory().create(std="f2003")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return parser(reader)
    except Exception as e:
        logger.warning(f"Failed to parse {path}: {e}")
        return None


def list_entry_points(source_dir: Path, status_fn=None) -> list[str]:
    from fparser.two import Fortran2003
    from fparser.two.utils import walk

    names = []
    for f in sorted(f for f in source_dir.glob("*.f") if not f.name.startswith(".")):
        if status_fn:
            status_fn(f"Parsing {f.name}…")
        tree = _parse_file(f)
        if tree is None:
            continue
        for node in walk(tree, Fortran2003.Subroutine_Stmt):
            names.append(str(node.items[1]))
        for node in walk(tree, Fortran2003.Function_Stmt):
            names.append(str(node.items[1]))
    return sorted(set(names))


def analyze_dependencies(source_dir: Path, entry_points: list[str], output_dir: Path, status_fn=None) -> dict:
    from fparser.two import Fortran2003
    from fparser.two.utils import walk

    # Map: function_name (upper) -> source file path
    name_to_file: dict[str, Path] = {}
    # Map: function_name (upper) -> set of called function names (upper)
    call_graph: dict[str, set[str]] = {}

    all_files = sorted(f for f in source_dir.glob("*.f") if not f.name.startswith("."))

    if status_fn:
        status_fn(f"Scanning {len(all_files)} Fortran files…")

    for i, f in enumerate(all_files):
        if status_fn:
            status_fn(f"Parsing {f.name} ({i+1}/{len(all_files)})…")
        tree = _parse_file(f)
        if tree is None:
            continue

        # Process each subprogram separately for accurate per-function call attribution
        for subprog in walk(tree, (Fortran2003.Subroutine_Subprogram,
                                   Fortran2003.Function_Subprogram)):
            stmts = walk(subprog, Fortran2003.Subroutine_Stmt) or walk(subprog, Fortran2003.Function_Stmt)
            if not stmts:
                continue
            name = str(stmts[0].items[1]).upper()
            name_to_file[name] = f

            calls: set[str] = set()
            for call_node in walk(subprog, Fortran2003.Call_Stmt):
                calls.add(str(call_node.items[0]).upper())

            if name not in call_graph:
                call_graph[name] = set()
            call_graph[name].update(calls)

    # BFS from entry points to find all reachable functions/files
    ep_upper = [ep.upper() for ep in entry_points]
    visited: set[str] = set()
    queue = list(ep_upper)
    while queue:
        fn = queue.pop()
        if fn in visited:
            continue
        visited.add(fn)
        for callee in call_graph.get(fn, set()):
            if callee not in visited and callee in name_to_file:
                queue.append(callee)

    reachable_files = sorted(set(
        str(name_to_file[fn]) for fn in visited if fn in name_to_file
    ))
    reachable_functions = sorted(visited)

    if status_fn:
        status_fn(f"Found {len(reachable_files)} source files, {len(reachable_functions)} functions")

    # Print summary of functions that will be converted
    from rich.columns import Columns
    from rich.console import Console
    from rich.text import Text
    _con = Console()
    _con.print("\n[bold]Functions to be converted:[/bold]")
    _con.print(Columns([Text(fn, style="yellow") for fn in reachable_functions], equal=True, expand=False))
    _con.print()

    # Build serializable call graph (sets -> lists)
    serializable_cg = {k: sorted(v) for k, v in call_graph.items() if k in visited}

    output_dir.mkdir(parents=True, exist_ok=True)

    if status_fn:
        status_fn("Writing dependency graph…")

    result = {
        "files": reachable_files,
        "functions": reachable_functions,
        "call_graph": serializable_cg,
    }

    (output_dir / "dep_graph.json").write_text(json.dumps(result, indent=2))

    # Write DOT file
    dot_lines = ["digraph call_graph {", "  rankdir=LR;"]
    for fn, callees in serializable_cg.items():
        for callee in callees:
            dot_lines.append(f'  "{fn}" -> "{callee}";')
    dot_lines.append("}")
    (output_dir / "dep_graph.dot").write_text("\n".join(dot_lines))

    return result

from __future__ import annotations

import json
import logging
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from ._log import make_stage_logger

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Worker function — runs in a child process.
# Must be a module-level function so ProcessPoolExecutor can pickle it.
# Returns only plain data (strings/tuples) so no fparser objects cross the
# process boundary.
# ---------------------------------------------------------------------------

def _parse_and_extract(path_str: str) -> tuple[str, list[tuple[str, str]], list[tuple[str, str]]]:
    """Parse one Fortran file and return its definitions and call edges.

    Returns:
        (path_str,
         name_entries: list of (func_name_upper, path_str),
         call_edges:   list of (caller_upper, callee_upper))

    An empty pair of lists is returned when parsing fails.
    """
    import warnings
    from fparser.common.readfortran import FortranFileReader
    from fparser.two import Fortran2003
    from fparser.two.parser import ParserFactory
    from fparser.two.utils import walk

    reader = FortranFileReader(path_str, ignore_comments=True)
    parser = ParserFactory().create(std="f2003")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tree = parser(reader)
    except Exception as exc:
        logger.warning(f"Failed to parse {path_str}: {exc}")
        return path_str, [], []

    name_entries: list[tuple[str, str]] = []
    call_edges: list[tuple[str, str]] = []

    for subprog in walk(tree, (Fortran2003.Subroutine_Subprogram,
                               Fortran2003.Function_Subprogram)):
        stmts = (walk(subprog, Fortran2003.Subroutine_Stmt)
                 or walk(subprog, Fortran2003.Function_Stmt))
        if not stmts:
            continue
        name = str(stmts[0].items[1]).upper()
        name_entries.append((name, path_str))

        calls: set[str] = set()
        for node in walk(subprog, Fortran2003.Call_Stmt):
            calls.add(str(node.items[0]).upper())
        for ref in walk(subprog, Fortran2003.Part_Ref):
            calls.add(str(ref.items[0]).upper())
        for ref in walk(subprog, Fortran2003.Function_Reference):
            calls.add(str(ref.items[0]).upper())
        for ref in walk(subprog, Fortran2003.Structure_Constructor):
            calls.add(str(ref.items[0]).upper())
        for callee in calls:
            call_edges.append((name, callee))

    return path_str, name_entries, call_edges


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def list_entry_points(source_dir: Path, status_fn=None) -> list[str]:
    all_files = sorted(f for f in source_dir.glob("*.f") if not f.name.startswith("."))
    if not all_files:
        return []

    if status_fn:
        status_fn(f"Parsing {len(all_files)} Fortran file(s) (parallel)…")

    names: list[str] = []
    with ProcessPoolExecutor() as executor:
        for _path, name_entries, _edges in executor.map(
            _parse_and_extract, [str(f) for f in all_files]
        ):
            names.extend(n for n, _ in name_entries)

    return sorted(set(names))


def analyze_dependencies(source_dir: Path, entry_points: list[str], output_dir: Path, status_fn=None) -> dict:
    log = make_stage_logger(output_dir)
    log.info(f"analyze_dependencies: source_dir={source_dir}, entry_points={entry_points}")

    all_files = sorted(f for f in source_dir.glob("*.f") if not f.name.startswith("."))

    if status_fn:
        status_fn(f"Parsing {len(all_files)} Fortran file(s) (parallel)…")
    log.info(f"Parsing {len(all_files)} Fortran source files in parallel")

    # Map: function_name (upper) -> source file path
    name_to_file: dict[str, Path] = {}
    # Map: function_name (upper) -> set of called function names (upper)
    call_graph: dict[str, set[str]] = {}

    with ProcessPoolExecutor() as executor:
        for path_str, name_entries, call_edges in executor.map(
            _parse_and_extract, [str(f) for f in all_files]
        ):
            f = Path(path_str)
            for name, _ in name_entries:
                existing = name_to_file.get(name)
                # Prefer the file whose stem matches the function name.
                if existing is None or f.stem.upper() == name:
                    name_to_file[name] = f
            for caller, callee in call_edges:
                if caller not in call_graph:
                    call_graph[caller] = set()
                call_graph[caller].add(callee)

    log.info(f"Parsed {len(all_files)} files; found {len(name_to_file)} definitions")

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
    log.info(f"Reachable files: {len(reachable_files)}, reachable functions: {len(reachable_functions)}")
    for fn in reachable_functions:
        log.info(f"  function: {fn} <- {name_to_file.get(fn, '(unknown)')}")

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
    log.info("Writing dep_graph.json and dep_graph.dot")

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

    log.info("Stage complete")
    return result

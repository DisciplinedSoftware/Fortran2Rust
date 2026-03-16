from __future__ import annotations

import re
from pathlib import Path


def strip_markdown_fences(content: str) -> str:
    content = re.sub(r"^```[a-zA-Z0-9_+-]*\s*\n?", "", content.strip())
    content = re.sub(r"\n?```\s*$", "", content)
    return content.strip()


def compact_rust_for_llm(code: str) -> tuple[str, str]:
    lines = code.splitlines()
    preserved_prefix: list[str] = []
    body_start = 0
    in_block_comment = False

    for i, line in enumerate(lines):
        stripped = line.strip()
        if in_block_comment:
            preserved_prefix.append(line)
            if "*/" in stripped:
                in_block_comment = False
            body_start = i + 1
            continue
        if not stripped:
            preserved_prefix.append(line)
            body_start = i + 1
            continue
        if stripped.startswith(("//", "///", "//!")):
            preserved_prefix.append(line)
            body_start = i + 1
            continue
        if stripped.startswith("/*"):
            preserved_prefix.append(line)
            body_start = i + 1
            if "*/" not in stripped:
                in_block_comment = True
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
        if stripped.startswith(("//", "///", "//!")):
            continue
        if not stripped:
            continue
        compact_lines.append(line.rstrip())

    return "\n".join(compact_lines).strip(), "\n".join(preserved_prefix).strip()


def restore_rust_after_llm(content: str, preserved_prefix: str) -> str:
    restored = content.strip()
    if not preserved_prefix:
        return restored
    if restored.startswith(("//", "///", "//!", "/*")):
        return restored
    return f"{preserved_prefix}\n\n{restored}".strip()


def split_llm_file_response(response: str, allowed_suffixes: tuple[str, ...]) -> list[tuple[str, str]]:
    parts = re.split(
        r"^---\s+(\S+)\s+---$",
        response,
        flags=re.MULTILINE,
    )
    files: list[tuple[str, str]] = []
    if len(parts) >= 3:
        it = iter(parts[1:])
        for filename, content in zip(it, it):
            if filename.endswith(allowed_suffixes):
                files.append((filename, strip_markdown_fences(content)))
    return files


def restore_rust_files_after_llm(
    written_files: list[Path],
    preserved_prefixes: dict[Path, str],
) -> None:
    for target in written_files:
        if target.suffix != ".rs":
            continue
        preserved_prefix = preserved_prefixes.get(target.resolve(), "")
        if not preserved_prefix:
            continue
        content = target.read_text()
        target.write_text(restore_rust_after_llm(content, preserved_prefix) + "\n")


def filter_errors_for_file(build_output: str, filename: str) -> str:
    """Return only the cargo/gcc error blocks that reference *filename*.

    When multiple files fail simultaneously every repair call currently
    receives the full build log.  Filtering to per-file errors removes
    the noise from unrelated files and shrinks the ERROR section of the
    prompt proportionally to the number of co-failing files.

    Falls back to the full *build_output* string if no matching blocks
    are found (e.g. unfamiliar output format).
    """
    basename = Path(filename).name

    # ── Cargo (Rust) format ────────────────────────────────────────────
    # Blocks start with a line like:
    #   error[E0425]: ...
    #   error: ...
    #   warning[...]: ...
    # and end at the next such line or end of output.
    # We keep blocks whose text references the target filename.
    cargo_block_re = re.compile(r"^(error|warning)[\[:]", re.MULTILINE)
    if cargo_block_re.search(build_output):
        # Split at every line that starts a new error/warning block, preserving
        # the full line by splitting on the *newline before* the marker.
        block_split_re = re.compile(r"\n(?=(?:error|warning)[\[:])", re.MULTILINE)
        raw_blocks = block_split_re.split(build_output)

        matching = [b for b in raw_blocks if basename in b]
        if matching:
            omitted = len(raw_blocks) - len(matching)
            result = "\n".join(matching)
            if omitted:
                result += f"\n[{omitted} error block(s) from other files omitted]\n"
            return result
        return build_output

    # ── GCC / Clang (C) format ─────────────────────────────────────────
    # Errors are line-prefixed:  filename.c:line:col: error: ...
    # Continuation / context lines start with a space or contain " | ".
    lines = build_output.splitlines(keepends=True)
    relevant: list[str] = []
    in_block = False
    for line in lines:
        if line.startswith(f"{basename}:") or f"/{basename}:" in line:
            relevant.append(line)
            in_block = True
        elif in_block and (line.startswith(" ") or " | " in line or line.strip() == "^"):
            relevant.append(line)
        else:
            in_block = False

    return "".join(relevant) if relevant else build_output


_BATCH_SIZE = 4


def batch_repair_files(
    llm,
    failing: list[Path],
    build_output: str,
    context: str,
    attempt: int,
) -> list[tuple[Path, str, str]]:
    """Repair *failing* Rust files, batching up to _BATCH_SIZE files per LLM call.

    Returns a list of (file, response_text, preserved_prefix) tuples.

    Files are grouped into batches.  Each batch is sent as a single
    ``--- filename ---`` delimited prompt so the number of LLM calls is
    ceil(N / _BATCH_SIZE) instead of N.  The batch error section contains
    only the error blocks that reference the files in that batch.

    If the LLM response cannot be split back into per-file sections the
    function falls back to individual full-file responses (writing the
    entire response for each file in the batch — callers discard invalid
    content via their normal validation logic).
    """
    results: list[tuple[Path, str, str]] = []

    for i in range(0, len(failing), _BATCH_SIZE):
        batch = failing[i : i + _BATCH_SIZE]

        if len(batch) == 1:
            rs_file = batch[0]
            compact_code, preserved_prefix = compact_rust_for_llm(rs_file.read_text())
            response = llm.repair(
                context=context,
                error=filter_errors_for_file(build_output, rs_file.name),
                code=compact_code,
                attempt=attempt,
            )
            results.append((rs_file, response, preserved_prefix))
            continue

        # Build a combined prompt with one section per file.
        compacted: list[tuple[Path, str, str]] = [
            (f, *compact_rust_for_llm(f.read_text())) for f in batch
        ]
        code_section = "\n\n".join(
            f"--- {f.name} ---\n{compact_code}" for f, compact_code, _ in compacted
        )
        batch_error = "\n\n".join(
            filter_errors_for_file(build_output, f.name) for f in batch
        )
        batch_context = (
            context
            + " Each file is delimited by '--- filename.rs ---'. "
            "Return each fixed file in the same '--- filename.rs ---' format."
        )
        response = llm.repair(
            context=batch_context,
            error=batch_error,
            code=code_section,
            attempt=attempt,
        )

        # Try to split the multi-file response.
        file_map = {f.name: (f, prefix) for f, _, prefix in compacted}
        parsed = split_llm_file_response(response, (".rs",))
        if parsed:
            returned = {name for name, _ in parsed}
            for name, content in parsed:
                if name in file_map:
                    f, prefix = file_map[name]
                    results.append((f, content, prefix))
            # Any file not returned by the LLM is left with its original content.
            for f, compact_code, prefix in compacted:
                if f.name not in returned:
                    results.append((f, f.read_text(), prefix))
        else:
            # Response is not split — fall back to individual calls for this batch.
            for f, compact_code, prefix in compacted:
                fb_response = llm.repair(
                    context=context,
                    error=filter_errors_for_file(build_output, f.name),
                    code=compact_code,
                    attempt=attempt,
                )
                results.append((f, fb_response, prefix))

    return results
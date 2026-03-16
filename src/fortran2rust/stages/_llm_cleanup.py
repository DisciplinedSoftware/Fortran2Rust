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
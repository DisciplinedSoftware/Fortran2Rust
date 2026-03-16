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
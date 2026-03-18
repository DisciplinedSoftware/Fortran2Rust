from __future__ import annotations

import pytest

from fortran2rust.stages.s1_analyze import analyze_dependencies


def test_analyze_dependencies_fails_when_no_fortran_sources(tmp_path) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(RuntimeError, match="No Fortran source files"):
        analyze_dependencies(tmp_path, ["DGEMM"], out_dir)


def test_analyze_dependencies_fails_on_unknown_entry_points(tmp_path) -> None:
    src = tmp_path / "foo.f"
    src.write_text(
        """
      SUBROUTINE FOO()
      RETURN
      END
      """.strip()
        + "\n"
    )
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(RuntimeError, match=r"Entry point\(s\) not found"):
        analyze_dependencies(tmp_path, ["DGEMM"], out_dir)

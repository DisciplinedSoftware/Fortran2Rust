from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from fortran2rust import cli


def test_non_interactive_defaults_to_dgemm_when_entry_points_omitted(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _fake_run_pipeline(_config, _library_path, entry_points):
        captured["entry_points"] = entry_points

    monkeypatch.setattr("fortran2rust.pipeline.run_pipeline", _fake_run_pipeline)
    monkeypatch.setattr("fortran2rust.cli.load_config", lambda **kwargs: SimpleNamespace(**kwargs))

    args = SimpleNamespace(
        library="blas",
        entry_points=None,
        stages=2,
        quick=False,
        max_retries=None,
        llm_provider=None,
        model=None,
        output_dir=str(tmp_path / "artifacts"),
    )

    cli._run_non_interactive(args)

    assert captured["entry_points"] == ["dgemm"]


def test_non_interactive_uses_given_entry_points(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _fake_run_pipeline(_config, _library_path, entry_points):
        captured["entry_points"] = entry_points

    monkeypatch.setattr("fortran2rust.pipeline.run_pipeline", _fake_run_pipeline)
    monkeypatch.setattr("fortran2rust.cli.load_config", lambda **kwargs: SimpleNamespace(**kwargs))

    args = SimpleNamespace(
        library=str(tmp_path),
        entry_points="DGEMM,DASUM",
        stages=2,
        quick=False,
        max_retries=None,
        llm_provider=None,
        model=None,
        output_dir=str(tmp_path / "artifacts"),
    )

    cli._run_non_interactive(args)

    assert captured["entry_points"] == ["DGEMM", "DASUM"]


def test_non_interactive_defaults_to_blas_library_when_library_omitted(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}
    expected_library = tmp_path / "blas-src"

    def _fake_run_pipeline(_config, library_path, _entry_points):
        captured["library_path"] = library_path

    monkeypatch.setattr("fortran2rust.pipeline.run_pipeline", _fake_run_pipeline)
    monkeypatch.setattr("fortran2rust.cli.load_config", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr("fortran2rust.blas.get_blas_source", lambda _console=None: expected_library)

    args = SimpleNamespace(
        library=None,
        entry_points=None,
        stages=2,
        quick=False,
        max_retries=None,
        llm_provider=None,
        model=None,
        output_dir=str(tmp_path / "artifacts"),
    )

    cli._run_non_interactive(args)

    assert captured["library_path"] == expected_library

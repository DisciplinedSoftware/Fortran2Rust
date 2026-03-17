from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from fortran2rust import cli
from fortran2rust.config import Config


def test_interactive_keeps_menu_stage_and_retry_selection(monkeypatch, tmp_path: Path) -> None:
    config = Config(stages=[1, 2], max_retries=2)
    expected_library = tmp_path / "lib"
    expected_entry_points = ["dasum"]

    def _fake_menu():
        return expected_library, expected_entry_points, config

    captured: dict[str, object] = {}

    def _fake_run_pipeline(cfg, library_path, entry_points):
        captured["config"] = cfg
        captured["library_path"] = library_path
        captured["entry_points"] = entry_points

    monkeypatch.setattr("fortran2rust.menu.run_interactive_menu", _fake_menu)
    monkeypatch.setattr("fortran2rust.pipeline.run_pipeline", _fake_run_pipeline)

    args = SimpleNamespace(
        library=None,
        entry_points=None,
        stages=None,
        quick=False,
        max_retries=None,
        llm_provider=None,
        model=None,
        output_dir=str(tmp_path / "artifacts"),
    )

    cli._run_interactive(args)

    assert captured["library_path"] == expected_library
    assert captured["entry_points"] == expected_entry_points
    assert captured["config"].stages == [1, 2]
    assert captured["config"].max_retries == 2


def test_interactive_explicit_stage_override_still_applies(monkeypatch, tmp_path: Path) -> None:
    config = Config(stages=[1, 2, 3, 4], max_retries=2)

    def _fake_menu():
        return tmp_path / "lib", ["dasum"], config

    captured: dict[str, object] = {}

    def _fake_run_pipeline(cfg, _library_path, _entry_points):
        captured["config"] = cfg

    monkeypatch.setattr("fortran2rust.menu.run_interactive_menu", _fake_menu)
    monkeypatch.setattr("fortran2rust.pipeline.run_pipeline", _fake_run_pipeline)

    args = SimpleNamespace(
        library=None,
        entry_points=None,
        stages=6,
        quick=True,
        max_retries=7,
        llm_provider=None,
        model=None,
        output_dir=str(tmp_path / "artifacts"),
    )

    cli._run_interactive(args)

    assert captured["config"].stages == [1, 2, 3, 4, 5, 6]
    assert captured["config"].max_retries == 7
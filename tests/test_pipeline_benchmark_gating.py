from __future__ import annotations

from fortran2rust.pipeline import _evaluate_stage_result


def test_stage6_marks_numerical_failures_as_issue() -> None:
    ok, notes = _evaluate_stage_result(
        6,
        {
            "bench_results": {
                "dgemm": {
                    "run_ok": True,
                    "pass": False,
                    "max_abs_diff": 1.0,
                }
            }
        },
    )

    assert ok is False
    assert "Rust numerical checks failed" in notes


def test_stage6_accepts_passing_benchmarks() -> None:
    ok, notes = _evaluate_stage_result(
        6,
        {
            "bench_results": {
                "dgemm": {
                    "run_ok": True,
                    "pass": True,
                    "max_abs_diff": 0.0,
                }
            }
        },
    )

    assert ok is True
    assert notes == ""

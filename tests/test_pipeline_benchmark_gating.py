from __future__ import annotations

from fortran2rust.pipeline import _blocking_stage_reason, _evaluate_stage_result


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


def test_stage5_is_blocked_when_stage4_failed() -> None:
    reason = _blocking_stage_reason(
        5,
        {
            4: {
                "error": "C compilation failed: LLM conversion did not return recognizable C code",
                "compile_ok": False,
            }
        },
    )

    assert reason == "Skipped because Stage 4 failed"


def test_stage5_is_blocked_when_stage4_has_no_compilable_c_output() -> None:
    reason = _blocking_stage_reason(
        5,
        {
            4: {
                "compile_ok": False,
                "bench_ok": False,
            }
        },
    )

    assert reason == "Skipped because Stage 4 did not produce compilable C output"

from __future__ import annotations

import json

from fortran2rust import repair_cache
from fortran2rust.stages import s4_llm_fix_c


def test_stage4_compile_commands_include_benchmark_c_files(tmp_path) -> None:
    (tmp_path / "dgemm.c").write_text("int dgemm_(void) { return 0; }\n")
    (tmp_path / "bench_dgemm.c").write_text("int main(void) { return 0; }\n")

    cc_path = s4_llm_fix_c._write_compile_commands(tmp_path)
    entries = json.loads(cc_path.read_text())
    stems = sorted({entry["file"].split("/")[-1].removesuffix(".c") for entry in entries})

    assert stems == ["bench_dgemm", "dgemm"]


def test_repair_cache_scope_isolated(tmp_path) -> None:
    code = "int f(void){return 0;}"
    error = "compile error"
    context = "fix it"

    repair_cache.store(code, error, context, "resp-a", cache_dir=tmp_path, cache_scope="run-a")

    assert repair_cache.lookup(code, error, context, cache_dir=tmp_path, cache_scope="run-a") == "resp-a"
    assert repair_cache.lookup(code, error, context, cache_dir=tmp_path, cache_scope="run-b") is None


class _BadLLM:
    def repair(self, **_kwargs) -> str:
        return "not valid c"

    def pop_conversation_log(self) -> list[dict]:
        return []


def test_stage4_persists_failure_artifacts_when_preconversion_fails(tmp_path) -> None:
    c_dir = tmp_path / "s3"
    out_dir = tmp_path / "s4"
    baseline_dir = tmp_path / "s2"
    c_dir.mkdir()
    baseline_dir.mkdir()

    (c_dir / "f2c.h").write_text("typedef int integer; typedef int ftnlen;\n")
    (c_dir / "dgemm.c").write_text("int dgemm_(void) { return 0; }\n")
    (c_dir / "xerbla.f").write_text(
        "      SUBROUTINE XERBLA(SRNAME,INFO)\n"
        "      CHARACTER*(*) SRNAME\n"
        "      INTEGER INFO\n"
        "      RETURN\n"
        "      END\n"
    )

    result = s4_llm_fix_c.fix_c_code(
        c_dir,
        out_dir,
        _BadLLM(),
        max_retries=2,
        baseline_dir=baseline_dir,
        call_graph={"DGEMM": ["XERBLA"]},
        entry_points=["DGEMM"],
    )

    saved = json.loads((out_dir / "result.json").read_text())

    assert result["compile_ok"] is False
    assert result["error"] == saved["error"]
    assert "recognizable C code" in result["error"]
    assert result["llm_turns"] == 1
    assert result["compile_commands"] == ""
    assert (out_dir / "llm_log.json").exists()
    assert (out_dir / "llm_conversations.json").exists()


class _NoopLLM:
    def pop_conversation_log(self) -> list[dict]:
        return []


def test_stage4_compile_fix_status_reports_batch_files_once(tmp_path, monkeypatch) -> None:
    c_dir = tmp_path / "s3"
    out_dir = tmp_path / "s4"
    baseline_dir = tmp_path / "s2"
    c_dir.mkdir()
    baseline_dir.mkdir()

    (c_dir / "f2c.h").write_text("typedef int integer; typedef int ftnlen;\n")
    (c_dir / "a.c").write_text("int a_(void) { return 0; }\n")
    (c_dir / "b.c").write_text("int b_(void) { return 0; }\n")

    compile_outputs = [
        (False, "a.c:1:1: error: bad\nb.c:2:1: error: bad"),
        (True, ""),
    ]

    def _fake_compile_c(_c_dir):
        return compile_outputs.pop(0)

    def _fake_repair_file(*_args, **_kwargs):
        return True

    monkeypatch.setattr(s4_llm_fix_c, "_compile_c", _fake_compile_c)
    monkeypatch.setattr(s4_llm_fix_c, "_repair_file", _fake_repair_file)

    statuses: list[str] = []

    result = s4_llm_fix_c.fix_c_code(
        c_dir,
        out_dir,
        _NoopLLM(),
        max_retries=3,
        baseline_dir=baseline_dir,
        status_fn=statuses.append,
    )

    assert result["compile_ok"] is True
    assert any(
        s == "LLM: fixing 2 file(s) (a.c, b.c) (attempt 1/3)…"
        for s in statuses
    )
    assert not any("LLM: fixing a.c (attempt 1/3)" in s for s in statuses)
    assert not any("LLM: fixing b.c (attempt 1/3)" in s for s in statuses)

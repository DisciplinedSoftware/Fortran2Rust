from __future__ import annotations

import json
from pathlib import Path

from fortran2rust.stages import s5_c2rust, s6_llm_fix_rust


class _FakeCompletedProcess:
    def __init__(self, returncode: int, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class _FakeLLM:
    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []

    def repair(self, context: str, error: str, code: str, attempt: int = 0) -> str:
        self.calls.append({
            "context": context,
            "error": error,
            "code": code,
            "attempt": str(attempt),
        })
        return (
            "pub unsafe extern \"C\" fn xerbla_(srname: *mut i8, info: *mut i32, srname_len: i32) {\n"
            "    let _ = (srname, info, srname_len);\n"
            "}\n"
        )

    def pop_conversation_log(self) -> list[dict]:
        return []


def test_transpile_to_rust_reports_missing_modules(tmp_path, monkeypatch) -> None:
    c_dir = tmp_path / "c"
    c_dir.mkdir()
    (c_dir / "dgemm.c").write_text("int dgemm_(void) { return 0; }\n")
    (c_dir / "xerbla.c").write_text("int xerbla_(void) { return 0; }\n")

    compile_commands = tmp_path / "compile_commands.json"
    compile_commands.write_text(json.dumps([
        {"file": str(c_dir / "dgemm.c"), "directory": str(c_dir), "command": "gcc -c dgemm.c"},
        {"file": str(c_dir / "xerbla.c"), "directory": str(c_dir), "command": "gcc -c xerbla.c"},
    ]))

    output_dir = tmp_path / "s5"

    def fake_run(cmd, capture_output, text, timeout):
        src_dir = output_dir / "src"
        src_dir.mkdir(parents=True, exist_ok=True)
        (src_dir / "dgemm.rs").write_text("pub unsafe extern \"C\" fn dgemm_() {}\n")
        return _FakeCompletedProcess(
            1,
            stderr=f"Error while processing {c_dir / 'xerbla.c'}.\n",
        )

    monkeypatch.setattr(s5_c2rust.subprocess, "run", fake_run)

    result = s5_c2rust.transpile_to_rust(c_dir, compile_commands, output_dir)

    assert result["expected_modules"] == ["dgemm", "xerbla"]
    assert result["generated_modules"] == ["dgemm"]
    assert result["missing_modules"] == ["xerbla"]
    saved = json.loads((output_dir / "c2rust_result.json").read_text())
    assert saved["missing_modules"] == ["xerbla"]


def test_fix_rust_code_preconverts_missing_modules_before_build(tmp_path, monkeypatch) -> None:
    run_dir = tmp_path / "run"
    s4_dir = run_dir / "s4_llm_fix_c"
    s5_dir = run_dir / "s5_c_to_rust_c2rust"
    s2_dir = run_dir / "s2_benchmark_generation"
    s6_dir = run_dir / "s6_llm_fix_rust"
    s4_dir.mkdir(parents=True)
    (s5_dir / "src").mkdir(parents=True)
    s2_dir.mkdir(parents=True)

    xerbla_c = s4_dir / "xerbla.c"
    xerbla_c.write_text(
        "#include <stdio.h>\n"
        "#include <stdlib.h>\n"
        "int xerbla_(char *srname, int *info, int srname_len) {\n"
        "    printf(\"bad\\n\");\n"
        "    exit(1);\n"
        "}\n"
    )
    dgemm_c = s4_dir / "dgemm.c"
    dgemm_c.write_text("int dgemm_(void) { return 0; }\n")
    (s4_dir / "compile_commands.json").write_text(json.dumps([
        {"file": str(dgemm_c), "directory": str(s4_dir), "command": "gcc -c dgemm.c"},
        {"file": str(xerbla_c), "directory": str(s4_dir), "command": "gcc -c xerbla.c"},
    ]))

    (s5_dir / "Cargo.toml").write_text("[package]\nname='demo'\nversion='0.1.0'\nedition='2021'\n")
    (s5_dir / "src" / "dgemm.rs").write_text("pub unsafe extern \"C\" fn dgemm_() {}\n")
    (s5_dir / "src" / "lib.rs").write_text("pub mod dgemm;\n")
    (s5_dir / "c2rust_result.json").write_text(json.dumps({
        "stderr": f"{xerbla_c}:4:5: error: call to undeclared library function 'printf'\n",
    }))

    llm = _FakeLLM()
    build_calls: list[Path] = []

    def fake_cargo_build(cargo_toml: Path):
        build_calls.append(cargo_toml)
        src_dir = cargo_toml.parent / "src"
        assert (src_dir / "xerbla.rs").exists()
        lib_rs = (src_dir / "lib.rs").read_text()
        assert "pub mod xerbla;" in lib_rs
        return True, ""

    monkeypatch.setattr(s6_llm_fix_rust, "_cargo_build", fake_cargo_build)
    monkeypatch.setattr(s6_llm_fix_rust, "run_rust_benchmarks", lambda *args, **kwargs: {})
    monkeypatch.setattr(s6_llm_fix_rust, "print_bench_summary", lambda *args, **kwargs: None)

    result = s6_llm_fix_rust.fix_rust_code(s5_dir, s6_dir, llm, max_retries=2, baseline_dir=s2_dir)

    assert build_calls == [s6_dir / "Cargo.toml"]
    assert result["preconverted_modules"] == ["xerbla"]
    assert result["llm_turns"] == 1
    assert result["retries"] == 0
    assert len(llm.calls) == 1
    assert "printf" in llm.calls[0]["error"]
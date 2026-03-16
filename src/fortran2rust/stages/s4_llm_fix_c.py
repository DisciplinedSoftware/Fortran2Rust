from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..llm.base import LLMClient


def _read_all_c_files(directory: Path) -> str:
    parts = []
    for f in sorted(directory.glob("*.c")):
        parts.append(f"--- {f.name} ---\n{f.read_text()}")
    return "\n\n".join(parts)


def _apply_llm_response(response: str, output_dir: Path) -> None:
    """Parse LLM response and write files back. Handles multi-file responses."""
    # Try to split on --- filename.ext --- markers
    parts = re.split(r"^---\s+(\S+\.(?:c|h|rs|toml))\s+---$", response, flags=re.MULTILINE)
    if len(parts) >= 3:
        # parts: [pre, filename1, content1, filename2, content2, ...]
        it = iter(parts[1:])
        for filename, content in zip(it, it):
            (output_dir / filename).write_text(content.strip() + "\n")
    else:
        # Single file — try to find something useful
        # Strip markdown fences if present
        content = re.sub(r"```[a-z]*\n?", "", response).strip()
        # Write to the first .c file we find (or generic fixed.c)
        c_files = sorted(output_dir.glob("*.c"))
        target = c_files[0] if c_files else output_dir / "fixed.c"
        target.write_text(content + "\n")


def _compile_c(c_dir: Path) -> tuple[bool, str]:
    c_files = sorted(c_dir.glob("*.c"))
    if not c_files:
        return False, "No .c files found"
    result = subprocess.run(
        ["gcc", "-O2", f"-I{c_dir}", "-o", "/dev/null"] + [str(f) for f in c_files] + ["-lm", "-lf2c"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    return result.returncode == 0, result.stderr


def _run_c_benchmark(c_dir: Path, bench_driver: Path, output_dir: Path) -> tuple[bool, str, Path | None]:
    """Compile and run C benchmark, return (ok, error, bin_path)."""
    c_files = sorted(c_dir.glob("*.c"))
    exe_path = output_dir / "bench_c"
    result = subprocess.run(
        ["gcc", "-O2", f"-I{c_dir}", str(bench_driver)] + [str(f) for f in c_files] + ["-lm", "-lf2c", "-o", str(exe_path)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        return False, result.stderr, None

    run_result = subprocess.run(
        [str(exe_path)],
        capture_output=True,
        text=True,
        cwd=str(output_dir),
        timeout=300,
    )
    if run_result.returncode != 0:
        return False, run_result.stderr, None

    bin_files = list(output_dir.glob("*.bin"))
    return True, "", bin_files[0] if bin_files else None


def fix_c_code(
    c_dir: Path,
    output_dir: Path,
    llm: "LLMClient",
    max_retries: int,
    baseline_dir: Path,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy all .c and .h files
    for f in c_dir.glob("*.c"):
        shutil.copy(f, output_dir / f.name)
    for f in c_dir.glob("*.h"):
        shutil.copy(f, output_dir / f.name)

    llm_log: list[dict] = []
    llm_turns = 0
    retries = 0

    # Compile loop
    compile_ok, compile_error = _compile_c(output_dir)
    for attempt in range(max_retries):
        if compile_ok:
            break
        code = _read_all_c_files(output_dir)
        response = llm.repair(
            context="Fix this C code produced by f2c Fortran transpiler. Ensure all declarations are correct.",
            error=compile_error,
            code=code,
        )
        llm_log.append({"phase": "compile", "attempt": attempt, "error": compile_error, "response_len": len(response)})
        llm_turns += 1
        retries += 1
        _apply_llm_response(response, output_dir)
        compile_ok, compile_error = _compile_c(output_dir)

    # Benchmark loop (only if compile succeeded)
    bench_ok = False
    if compile_ok:
        # Find benchmark driver in baseline_dir
        bench_drivers = list(baseline_dir.glob("bench_*.c"))
        if not bench_drivers:
            bench_ok = True  # no benchmark available, skip
        else:
            import numpy as np

            for bench_driver in bench_drivers:
                ok, err, c_bin = _run_c_benchmark(output_dir, bench_driver, output_dir)
                if not ok:
                    continue

                # Find corresponding fortran baseline
                fn_name = re.sub(r"^bench_", "", bench_driver.stem)
                fortran_bin = baseline_dir / f"bench_{fn_name}_baseline.bin"
                if not fortran_bin.exists():
                    fortran_bin = next(baseline_dir.glob(f"*{fn_name}*baseline*.bin"), None)

                if fortran_bin and c_bin:
                    try:
                        fortran_data = np.fromfile(str(fortran_bin), dtype=np.float64)
                        c_data = np.fromfile(str(c_bin), dtype=np.float64)
                        if fortran_data.shape == c_data.shape and np.allclose(c_data, fortran_data, atol=1e-12, rtol=1e-12):
                            bench_ok = True
                        else:
                            max_abs = float(np.max(np.abs(c_data - fortran_data))) if fortran_data.shape == c_data.shape else float("inf")
                            for attempt in range(max_retries):
                                code = _read_all_c_files(output_dir)
                                response = llm.repair(
                                    context=f"Fix numerical precision: max abs diff = {max_abs:.6e}. The C output must match the Fortran baseline to within atol=rtol=1e-12.",
                                    error=f"Numerical mismatch: max_abs_diff={max_abs:.6e}",
                                    code=code,
                                )
                                llm_log.append({"phase": "bench", "attempt": attempt, "max_abs": max_abs})
                                llm_turns += 1
                                retries += 1
                                _apply_llm_response(response, output_dir)
                                ok2, err2, c_bin2 = _run_c_benchmark(output_dir, bench_driver, output_dir)
                                if ok2 and c_bin2:
                                    c_data2 = np.fromfile(str(c_bin2), dtype=np.float64)
                                    if fortran_data.shape == c_data2.shape and np.allclose(c_data2, fortran_data, atol=1e-12, rtol=1e-12):
                                        bench_ok = True
                                        break
                    except Exception:
                        pass

    (output_dir / "llm_log.json").write_text(json.dumps(llm_log, indent=2))
    result = {
        "compile_ok": compile_ok,
        "bench_ok": bench_ok,
        "llm_turns": llm_turns,
        "retries": retries,
        "compile_error": compile_error if not compile_ok else "",
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2))
    return result

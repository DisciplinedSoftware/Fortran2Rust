# Fortran2Rust — Copilot Instructions

## Project overview

Fortran2Rust is a 9-stage automated pipeline that converts legacy Fortran source code into safe, idiomatic Rust. It chains classical transpilation tools (`f2c`, `c2rust`) with LLM-driven repair and refactoring passes, validating numerical precision against the original Fortran output at each stage.

## Setup & commands

**Install (dev mode):**
```bash
pip install -e '.[dev]'
```

**Run the pipeline:**
```bash
fortran2rust                    # interactive menu
fortran2rust --non-interactive --library /path/to/fortran --entry-points all
fortran2rust --non-interactive --library blas --entry-points dgemm
fortran2rust --stages 4,6 --library /path/to/fortran --entry-points dgemm
```

**Lint:**
```bash
ruff check src/
```
Line length is 100 chars, target Python 3.10 (`[tool.ruff]` in `pyproject.toml`).

**Tests:**
```bash
pytest
```

**Environment:** Copy `.env.example` to `.env` and fill in the LLM API key for the chosen provider before running.

## Architecture

### 9-stage pipeline (`src/fortran2rust/pipeline.py`)

Stages run sequentially; each writes artifacts into `artifacts/{run_id}/s{N}_{slug}/`. Each stage function accepts a `status_fn: Callable[[str], None]` for Rich terminal status updates, and returns a `dict` that is passed forward through `results[N]`.

| Stage | Module | What it does |
|-------|--------|--------------|
| 1 | `s1_analyze.py` | BFS dependency analysis via fparser2; produces `dep_graph.json` and `.dot` |
| 2 | `s2_benchmarks.py` | Generates Fortran + C benchmark drivers and shared binary datasets; runs Fortran to establish the numerical baseline |
| 3 | `s3_f2c.py` | Runs `f2c -a` on each reachable Fortran file; produces `compile_commands.json` |
| 4 | `s4_llm_fix_c.py` | LLM repair loop: fixes `gcc` compilation errors and validates C numerical output against Fortran baseline (atol/rtol 1e-10) |
| 5 | `s5_c2rust.py` | Runs `c2rust transpile`; scaffolds `Cargo.toml` and `src/lib.rs` |
| 6 | `s6_llm_fix_rust.py` | LLM repair loop: fixes `cargo build` errors; captures timing and numerical diff vs Fortran |
| 7 | `s7_llm_safe.py` | LLM pass to eliminate `unsafe` blocks, file-by-file; reverts file if build breaks |
| 8 | `s8_llm_idiomatic.py` | LLM pass for idiomatic Rust (iterators, snake_case, Result/Option); reverts file if build breaks |
| 9 | `s9_report.py` | Generates `report.html` and `report.md` in the run directory |

### LLM abstraction (`src/fortran2rust/llm/`)

`LLMClient` (ABC in `base.py`) defines two methods:
- `complete(system, user) -> str` — raw chat completion
- `repair(context, error, code) -> str` — standard fix-broken-code prompt; returns ONLY corrected code, no markdown

New providers must extend `LLMClient` and register in `factory.py`'s `get_llm_client()`. Supported providers: `openai`, `anthropic`, `google`, `openrouter`, `ollama`.

### Configuration (`src/fortran2rust/config.py`)

`Config` is a plain `@dataclass` loaded from environment variables by `load_config(**overrides)`. The `stages` field (list of ints) controls which pipeline stages run, allowing partial re-runs.

### Exception hierarchy (`src/fortran2rust/exceptions.py`)

```
PipelineError
├── CompilationError(language, snippet)
├── NumericalPrecisionError(fn_name, max_diff)
├── BenchmarkRuntimeError(fn_name, snippet)
├── ConversionError(tool, message)
└── MaxRetriesExceededError(stage, wrapped)   # wraps any PipelineError
```

All LLM repair loops raise `MaxRetriesExceededError` when `max_retries` is exhausted. The pipeline catches generic `Exception` per-stage and stores `{"error": str(e)}` rather than aborting.

## Key conventions

### f2c calling convention
C functions transpiled from Fortran get a trailing underscore (e.g., `dgemm_`). Character (`char*`) arguments are followed by an extra `ftnlen` length argument at the end of the signature. Types come from `f2c.h` (`integer`, `doublereal`, `real`, `ftnlen`, etc.).

### LLM response format
Multi-file responses use `--- filename.rs ---` delimiters (parsed by regex in `_apply_llm_response`). Single-file responses strip markdown fences. LLM prompts explicitly say "Return ONLY the corrected complete file(s), no explanations, no markdown fences."

### Binary dataset format
Input/output datasets are raw `float64` binary files in Fortran column-major order (numpy `order="F"`). All three languages (Fortran, C, Rust) read the same shared dataset files. Fortran benchmark output is the ground truth; later stages compare against it using `numpy.allclose` or `numpy.max(abs(diff))`.

### Stage artifact naming
Output directories follow the pattern `s{N}_{slug}/` where the slug is derived from the stage name (e.g., `s4_llm_fix_c`). Every stage writes:
- `stage.log` — timestamped human-readable log of all events in that stage
- `result.json` — structured outcome (exit status, counts, errors — untruncated)
- `llm_log.json` — per-turn LLM call log (stages 4, 6, 7, 8)

Compilation output is kept as dedicated log files (full stdout+stderr, untruncated):
- `s2_*/gfortran_{fn}.log` — gfortran compile + run output per entry point
- `s3_*/f2c_{stem}.log` — f2c output per Fortran file; `gcc_verify.log` — verification compile
- `s4_*/gcc_compile.log` — all gcc attempts appended in order
- `s5_*/c2rust.log` — full c2rust transpile output
- `s6_*/cargo_build.log`, `s7_*/cargo_build.log`, `s8_*/cargo_build.log` — all cargo build attempts appended in order

### Adding a new stage
1. Create `src/fortran2rust/stages/s{N}_{slug}.py` with a top-level function matching the pattern of existing stages.
2. Add the stage name to `STAGE_NAMES` in `pipeline.py`.
3. Add the dispatch branch in the `if/elif` chain in `run_pipeline()`.
4. Update `Config.stages` default range if the new stage should run by default.

### Stage-2 benchmark generation
`s2_benchmarks.py` is signature-driven. It parses callable signatures from source and generates typed benchmark drivers and shared datasets without name-based kernel allowlists. Precision datasets use fixed seed 43 in addition to the main benchmark seed 42.

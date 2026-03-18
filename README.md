# Fortran2Rust

Fortran2Rust is a multi-stage automated pipeline that converts legacy Fortran source code into safe, idiomatic Rust. It combines classical transpilation tools (`f2c`, `c2rust`) with LLM-driven repair and refactoring passes to produce Rust code that compiles cleanly, passes numerical precision validation against the original Fortran output, eliminates `unsafe` blocks, and follows modern Rust idioms — all with minimal human intervention.

> **Full documentation coming soon.**

## Quickstart (Dev Container)

The easiest way to get started is with the included dev container, which pre-installs all dependencies (gfortran, f2c, LLVM 17, Rust, c2rust, Python 3).

1. Open the repository in VS Code.
2. When prompted, click **"Reopen in Container"** (or run **Dev Containers: Reopen in Container** from the command palette).
3. Copy `.env.example` to `.env` and configure your LLM provider.
   - Set `LLM_PROVIDER` (default: `openai`; options: `openai`, `anthropic`, `google`, `openrouter`, `github`, `ollama`).
   - For most providers, set a provider-specific API key (e.g., `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`).
   - For `github` provider, set `GITHUB_TOKEN` (reads from `gh auth` if available, or set explicitly).
   - You can also set a generic `LLM_API_KEY` as a fallback for all providers.
   - For large files, increase `LLM_MAX_TOKENS` (for example `16384`) to reduce truncated LLM outputs.
   - Tune `LLM_MAX_PARALLEL` (default `2`) to cap concurrent LLM requests in stages 7/8 and avoid RAM spikes.
   - Stage 2 memory/timing tuning: `S2_MAX_PARALLEL` (default `2`), `S2_MATRIX_N_MAX` (default `768`), `S2_TIMING_MAX_RUNS` (default `12`), and `S2_DATASET_REUSE_EVERY` (default `3`).
4. Run the pipeline:
   ```bash
   fortran2rust
   ```
   The interactive menu will guide you through selecting a library, entry points, and LLM provider.

To run non-interactively (converts `dgemm` from BLAS as a demo):
```bash
fortran2rust --non-interactive
```

Each pipeline run writes artifacts under `artifacts/<run_id>/`. The final Rust output is available directly at the run root as `Cargo.toml` and `src/` (exported from Stage 8), in addition to the per-stage folders and report files.

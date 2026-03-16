# Fortran2Rust

Fortran2Rust is a multi-stage automated pipeline that converts legacy Fortran source code into safe, idiomatic Rust. It combines classical transpilation tools (`f2c`, `c2rust`) with LLM-driven repair and refactoring passes to produce Rust code that compiles cleanly, passes numerical precision validation against the original Fortran output, eliminates `unsafe` blocks, and follows modern Rust idioms — all with minimal human intervention.

> **Full documentation coming soon.**

## Quickstart (Dev Container)

The easiest way to get started is with the included dev container, which pre-installs all dependencies (gfortran, f2c, LLVM 17, Rust, c2rust, Python 3).

1. Open the repository in VS Code.
2. When prompted, click **"Reopen in Container"** (or run **Dev Containers: Reopen in Container** from the command palette).
3. Copy `.env.example` to `.env` and fill in your LLM API key.
   - For large files, increase `LLM_MAX_TOKENS` (for example `16384`) to reduce truncated LLM outputs.
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

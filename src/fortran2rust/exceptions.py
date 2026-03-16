from __future__ import annotations


class PipelineError(Exception):
    """Base class for all pipeline-stage errors."""


class CompilationError(PipelineError):
    def __init__(self, language: str, snippet: str = "") -> None:
        self.language = language
        self.snippet = snippet
        super().__init__(f"{language} compilation failed")


class NumericalPrecisionError(PipelineError):
    def __init__(self, fn_name: str, max_diff: float) -> None:
        self.fn_name = fn_name
        self.max_diff = max_diff
        super().__init__(
            f"Numerical precision check failed for '{fn_name}': max diff = {max_diff:.3e}"
        )


class BenchmarkRuntimeError(PipelineError):
    def __init__(self, fn_name: str, snippet: str = "") -> None:
        self.fn_name = fn_name
        self.snippet = snippet
        super().__init__(f"Benchmark failed to execute for '{fn_name}'")


class ConversionError(PipelineError):
    def __init__(self, tool: str, message: str = "") -> None:
        self.tool = tool
        super().__init__(f"{tool} conversion failed: {message}" if message else f"{tool} conversion failed")


class MaxRetriesExceededError(PipelineError):
    def __init__(self, stage: str, wrapped: PipelineError) -> None:
        self.stage = stage
        self.wrapped = wrapped
        super().__init__(f"{wrapped} (max retries exceeded)")

from __future__ import annotations

import urllib.request
import tarfile
from pathlib import Path

from rich.progress import Progress, SpinnerColumn, TextColumn

BLAS_URL = "https://netlib.org/blas/blas.tgz"
CACHE_DIR = Path.home() / ".fortran2rust" / "blas"


def get_blas_source(console=None) -> Path:
    """Download and extract BLAS source if not cached. Returns path to .f files directory."""
    if CACHE_DIR.exists() and any(CACHE_DIR.glob("*.f")):
        if console:
            console.print(f"[dim]Using cached BLAS at {CACHE_DIR}[/dim]")
        return CACHE_DIR

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tgz_path = CACHE_DIR / "blas.tgz"

    with Progress(SpinnerColumn(), TextColumn("[bold blue]Downloading BLAS from netlib..."), transient=True) as p:
        p.add_task("download", total=None)
        urllib.request.urlretrieve(BLAS_URL, tgz_path)

    with tarfile.open(tgz_path) as tar:
        for member in tar.getmembers():
            if member.name.endswith(".f") or member.name.endswith(".f90"):
                member.name = Path(member.name).name  # flatten
                tar.extract(member, CACHE_DIR)

    tgz_path.unlink()
    return CACHE_DIR

from __future__ import annotations

import logging
from pathlib import Path


def make_stage_logger(stage_dir: Path) -> logging.Logger:
    """Return a Logger that writes timestamped lines to stage_dir/stage.log.

    Creates an unregistered Logger instance so multiple pipeline runs in the
    same process never share handlers or pollute the root logger.
    """
    log = logging.Logger(str(stage_dir))
    log.setLevel(logging.DEBUG)
    fh = logging.FileHandler(stage_dir / "stage.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s"))
    log.addHandler(fh)
    return log

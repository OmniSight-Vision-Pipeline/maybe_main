from __future__ import annotations

import os
from pathlib import Path


def configure_ultralytics_dir(base_dir: str | Path | None = None) -> str:
    """
    Ensure Ultralytics writes settings into a writable local directory.
    """
    root = Path(base_dir) if base_dir is not None else Path.cwd()
    target = root / ".ultralytics"
    target.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", target.as_posix())
    return target.as_posix()

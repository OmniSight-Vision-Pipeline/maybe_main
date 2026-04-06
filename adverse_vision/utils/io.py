from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def find_image_files(root: str | Path) -> List[Path]:
    """Discover image files recursively under root."""
    base = Path(root)
    if not base.exists():
        raise FileNotFoundError(f"Image root does not exist: {base}")
    files = [p for p in base.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]
    if not files:
        raise FileNotFoundError(f"No image files found under: {base}")
    return sorted(files)


def ensure_parent_dir(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def as_posix_paths(paths: Iterable[Path]) -> list[str]:
    return [p.as_posix() for p in paths]

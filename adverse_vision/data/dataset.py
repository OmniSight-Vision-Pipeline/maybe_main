from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import cv2
import torch
from torch.utils.data import Dataset

from adverse_vision.data.corruption import apply_corruption, build_deterministic_config
from adverse_vision.data.split import load_split_manifest, split_paths
from adverse_vision.utils.io import find_image_files

SplitName = Literal["train", "val", "test"]


class DrivingRestorationDataset(Dataset):
    """
    Dataset returning a corrupted input frame and its clean target frame.
    """

    def __init__(
        self,
        root: str | Path,
        split: SplitName,
        img_size: tuple[int, int] = (640, 384),
        seed: int = 42,
        split_manifest: str | Path | None = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.img_size = img_size
        self.seed = seed

        if split_manifest:
            split_data = load_split_manifest(split_manifest)
            self.paths = [Path(p) for p in split_data[split]]
        else:
            all_images = find_image_files(self.root)
            split_data = split_paths([p.as_posix() for p in all_images], seed=seed)
            self.paths = [Path(p) for p in split_data[split]]
        if not self.paths:
            raise ValueError(f"No images available for split '{split}' in {self.root}")

    def __len__(self) -> int:
        return len(self.paths)

    def _load_rgb(self, path: Path) -> torch.Tensor:
        bgr = cv2.imread(path.as_posix(), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Failed to read image: {path}")
        resized = cv2.resize(bgr, self.img_size, interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        return tensor

    def __getitem__(self, index: int) -> dict:
        image_path = self.paths[index]
        clean = self._load_rgb(image_path)
        clean_np = (clean.permute(1, 2, 0).numpy() * 255.0).astype("uint8")

        config = build_deterministic_config(index=index, seed=self.seed + self._split_offset())
        corrupted_np, metadata = apply_corruption(clean_np, config)
        corrupted = torch.from_numpy(corrupted_np).permute(2, 0, 1).float() / 255.0

        return {
            "corrupted": corrupted,
            "clean": clean,
            "metadata": json.dumps(metadata, sort_keys=True),
            "path": image_path.as_posix(),
        }

    def _split_offset(self) -> int:
        return {"train": 0, "val": 1_000_000, "test": 2_000_000}[self.split]

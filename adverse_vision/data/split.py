from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np


def split_paths(
    paths: Iterable[str],
    seed: int = 42,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> dict[str, list[str]]:
    ratios = np.array([train_ratio, val_ratio, test_ratio], dtype=float)
    if not np.isclose(ratios.sum(), 1.0):
        raise ValueError("Split ratios must sum to 1.0")

    paths_list = sorted(paths)
    rng = np.random.default_rng(seed)
    indices = np.arange(len(paths_list))
    rng.shuffle(indices)

    n_total = len(paths_list)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val

    train = [paths_list[i] for i in indices[:n_train]]
    val = [paths_list[i] for i in indices[n_train : n_train + n_val]]
    test = [paths_list[i] for i in indices[n_train + n_val : n_train + n_val + n_test]]
    return {"train": train, "val": val, "test": test}


def build_split_manifest(
    image_paths: Iterable[str],
    output_path: str | Path,
    seed: int = 42,
) -> dict[str, list[str]]:
    split = split_paths(image_paths, seed=seed)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"seed": seed, "ratios": {"train": 0.7, "val": 0.15, "test": 0.15}, "split": split}
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return split


def load_split_manifest(path: str | Path) -> dict[str, list[str]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["split"]

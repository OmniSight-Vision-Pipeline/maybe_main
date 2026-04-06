from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from adverse_vision.data.dataset import DrivingRestorationDataset
from adverse_vision.data.split import build_split_manifest
from adverse_vision.models import CompositeRestorationLoss, UNetAutoencoder
from adverse_vision.train import TrainConfig, Trainer
from adverse_vision.utils.io import find_image_files


def _create_toy_images(root: Path, n: int = 12) -> None:
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(123)
    for i in range(n):
        image = rng.integers(0, 256, size=(96, 160, 3), dtype=np.uint8)
        cv2.imwrite((root / f"img_{i:03d}.jpg").as_posix(), image)


def test_smoke_train_one_epoch(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    out_dir = tmp_path / "out"
    _create_toy_images(data_root, n=14)
    split_manifest = out_dir / "split.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    build_split_manifest([p.as_posix() for p in find_image_files(data_root)], split_manifest, seed=42)

    train_ds = DrivingRestorationDataset(data_root, split="train", img_size=(160, 96), split_manifest=split_manifest)
    val_ds = DrivingRestorationDataset(data_root, split="val", img_size=(160, 96), split_manifest=split_manifest)
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=0)

    model = UNetAutoencoder()
    loss_fn = CompositeRestorationLoss()
    loss_fn.perceptual.enabled = False

    optimizer = AdamW(model.parameters(), lr=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=1)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        out_dir=out_dir,
        config=TrainConfig(epochs=1, patience=1, device="cpu"),
    )
    results = trainer.fit(train_loader, val_loader)

    assert Path(results["best_checkpoint"]).exists()
    assert Path(results["last_checkpoint"]).exists()
    assert (out_dir / "history.json").exists()

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from adverse_vision.data.dataset import DrivingRestorationDataset
from adverse_vision.data.split import build_split_manifest
from adverse_vision.models import CompositeRestorationLoss, UNetAutoencoder
from adverse_vision.train import TrainConfig, Trainer
from adverse_vision.utils.io import find_image_files
from adverse_vision.utils.seed import set_seed


def parse_img_size(raw: str) -> tuple[int, int]:
    raw = raw.lower().replace("x", ",")
    width, height = raw.split(",")
    return int(width.strip()), int(height.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train restoration autoencoder.")
    parser.add_argument("--data-root", required=True, type=str, help="Dataset root containing clear driving images.")
    parser.add_argument("--out-dir", required=True, type=str, help="Output directory for checkpoints and logs.")
    parser.add_argument("--img-size", default="640x384", type=str, help="Image size as WIDTHxHEIGHT.")
    parser.add_argument("--epochs", default=80, type=int)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--patience", default=10, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--num-workers", default=0, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    img_size = parse_img_size(args.img_size)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_images = find_image_files(args.data_root)
    split_manifest_path = out_dir / "split_manifest.json"
    build_split_manifest([p.as_posix() for p in all_images], split_manifest_path, seed=args.seed)

    train_ds = DrivingRestorationDataset(
        root=args.data_root,
        split="train",
        img_size=img_size,
        seed=args.seed,
        split_manifest=split_manifest_path,
    )
    val_ds = DrivingRestorationDataset(
        root=args.data_root,
        split="val",
        img_size=img_size,
        seed=args.seed,
        split_manifest=split_manifest_path,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNetAutoencoder()
    loss_fn = CompositeRestorationLoss(w_l1=0.6, w_ssim=0.3, w_perceptual=0.1)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        out_dir=out_dir,
        config=TrainConfig(epochs=args.epochs, patience=args.patience, device=device),
    )

    results = trainer.fit(train_loader=train_loader, val_loader=val_loader)
    summary = {
        "device": device,
        "img_size": {"width": img_size[0], "height": img_size[1]},
        "dataset": {"train_size": len(train_ds), "val_size": len(val_ds)},
        "results": results,
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

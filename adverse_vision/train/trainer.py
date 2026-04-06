from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from adverse_vision.utils.metrics import psnr_torch, ssim_torch


@dataclass
class TrainConfig:
    epochs: int = 80
    patience: int = 10
    grad_clip: float = 1.0
    device: str = "cpu"


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler | None,
        loss_fn: nn.Module,
        out_dir: str | Path,
        config: TrainConfig,
    ) -> None:
        self.model = model.to(config.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.config = config

        self.history: list[dict[str, Any]] = []
        self.best_score = float("-inf")
        self.best_epoch = -1
        self._epochs_without_improve = 0

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> dict[str, Any]:
        for epoch in range(1, self.config.epochs + 1):
            train_metrics = self._run_epoch(train_loader, train=True, epoch=epoch)
            val_metrics = self._run_epoch(val_loader, train=False, epoch=epoch)

            if self.scheduler is not None:
                self.scheduler.step()

            epoch_metrics = {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
                "lr": self.optimizer.param_groups[0]["lr"],
            }
            self.history.append(epoch_metrics)
            self._save_history()

            score = self._validation_score(val_metrics)
            if score > self.best_score:
                self.best_score = score
                self.best_epoch = epoch
                self._epochs_without_improve = 0
                self._save_checkpoint(epoch, val_metrics, best=True)
            else:
                self._epochs_without_improve += 1

            self._save_checkpoint(epoch, val_metrics, best=False)
            if self._epochs_without_improve >= self.config.patience:
                break

        return {
            "best_epoch": self.best_epoch,
            "best_score": self.best_score,
            "history_path": (self.out_dir / "history.json").as_posix(),
            "best_checkpoint": (self.out_dir / "best.pt").as_posix(),
            "last_checkpoint": (self.out_dir / "last.pt").as_posix(),
        }

    def _run_epoch(self, loader: DataLoader, train: bool, epoch: int) -> dict[str, float]:
        mode = "train" if train else "val"
        self.model.train(mode=train)
        totals = {"loss": 0.0, "l1": 0.0, "ssim": 0.0, "perceptual": 0.0, "psnr": 0.0}
        steps = 0

        iterator = tqdm(loader, desc=f"{mode} epoch {epoch}", leave=False)
        for batch in iterator:
            corrupted = batch["corrupted"].to(self.config.device, non_blocking=True)
            clean = batch["clean"].to(self.config.device, non_blocking=True)

            if train:
                self.optimizer.zero_grad(set_to_none=True)

            with torch.set_grad_enabled(train):
                restored = self.model(corrupted)
                loss, components = self.loss_fn(restored, clean)
                psnr_value = psnr_torch(restored, clean)
                ssim_value = ssim_torch(restored, clean)
                if train:
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                    self.optimizer.step()

            totals["loss"] += float(loss.detach().cpu().item())
            totals["l1"] += components["l1"]
            totals["ssim"] += float(ssim_value.detach().cpu().item())
            totals["perceptual"] += components["perceptual"]
            totals["psnr"] += float(psnr_value.detach().cpu().item())
            steps += 1

            iterator.set_postfix({"loss": totals["loss"] / max(steps, 1), "psnr": totals["psnr"] / max(steps, 1)})

        if steps == 0:
            return {key: 0.0 for key in totals}
        return {key: value / steps for key, value in totals.items()}

    def _validation_score(self, val_metrics: dict[str, float]) -> float:
        return val_metrics["psnr"] + (20.0 * val_metrics["ssim"])

    def _save_checkpoint(self, epoch: int, val_metrics: dict[str, float], best: bool) -> None:
        payload = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
            "val_metrics": val_metrics,
            "config": asdict(self.config),
        }
        path = self.out_dir / ("best.pt" if best else "last.pt")
        torch.save(payload, path)

    def _save_history(self) -> None:
        (self.out_dir / "history.json").write_text(json.dumps(self.history, indent=2), encoding="utf-8")

from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from adverse_vision.utils.metrics import ssim_torch


class PerceptualLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.enabled = False
        self.features: nn.Module | None = None
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

        try:
            from torchvision.models import VGG16_Weights, vgg16

            try:
                backbone = vgg16(weights=VGG16_Weights.IMAGENET1K_FEATURES)
            except Exception:
                backbone = vgg16(weights=None)
            self.features = backbone.features[:16].eval()
            for param in self.features.parameters():
                param.requires_grad = False
            self.enabled = True
        except Exception:
            self.features = None
            self.enabled = False

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if not self.enabled or self.features is None:
            return pred.new_tensor(0.0)

        features = self.features.to(pred.device)
        pred_n = (pred - self.mean.to(pred.device)) / self.std.to(pred.device)
        target_n = (target - self.mean.to(target.device)) / self.std.to(target.device)
        pred_f = features(pred_n)
        target_f = features(target_n)
        return nn.functional.l1_loss(pred_f, target_f)


class CompositeRestorationLoss(nn.Module):
    """0.6*L1 + 0.3*(1-SSIM) + 0.1*Perceptual."""

    def __init__(self, w_l1: float = 0.6, w_ssim: float = 0.3, w_perceptual: float = 0.1) -> None:
        super().__init__()
        self.w_l1 = w_l1
        self.w_ssim = w_ssim
        self.w_perceptual = w_perceptual
        self.l1 = nn.L1Loss()
        self.perceptual = PerceptualLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, Dict[str, float]]:
        l1_value = self.l1(pred, target)
        ssim_value = ssim_torch(pred, target)
        ssim_loss = 1.0 - ssim_value
        perceptual_value = self.perceptual(pred, target)
        total = self.w_l1 * l1_value + self.w_ssim * ssim_loss + self.w_perceptual * perceptual_value
        components = {
            "l1": float(l1_value.detach().cpu().item()),
            "ssim": float(ssim_value.detach().cpu().item()),
            "ssim_loss": float(ssim_loss.detach().cpu().item()),
            "perceptual": float(perceptual_value.detach().cpu().item()),
            "total": float(total.detach().cpu().item()),
        }
        return total, components

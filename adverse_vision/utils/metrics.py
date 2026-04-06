from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np
import torch
import torch.nn.functional as F


def psnr_torch(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mse = F.mse_loss(pred, target, reduction="mean")
    return 10.0 * torch.log10(1.0 / (mse + eps))


def _gaussian_window(window_size: int, sigma: float, channels: int, device: torch.device) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    kernel_1d = torch.exp(-(coords**2) / (2 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    return kernel_2d.expand(channels, 1, window_size, window_size).contiguous()


def ssim_torch(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    c1: float = 0.01**2,
    c2: float = 0.03**2,
) -> torch.Tensor:
    channels = pred.shape[1]
    window = _gaussian_window(window_size, sigma, channels, pred.device)
    padding = window_size // 2

    mu_pred = F.conv2d(pred, window, padding=padding, groups=channels)
    mu_target = F.conv2d(target, window, padding=padding, groups=channels)
    mu_pred_sq = mu_pred.pow(2)
    mu_target_sq = mu_target.pow(2)
    mu_pred_target = mu_pred * mu_target

    sigma_pred_sq = F.conv2d(pred * pred, window, padding=padding, groups=channels) - mu_pred_sq
    sigma_target_sq = F.conv2d(target * target, window, padding=padding, groups=channels) - mu_target_sq
    sigma_pred_target = F.conv2d(pred * target, window, padding=padding, groups=channels) - mu_pred_target

    numerator = (2 * mu_pred_target + c1) * (2 * sigma_pred_target + c2)
    denominator = (mu_pred_sq + mu_target_sq + c1) * (sigma_pred_sq + sigma_target_sq + c2)
    ssim_map = numerator / (denominator + 1e-12)
    return ssim_map.mean()


def box_iou_xyxy(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def _compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    indices = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[indices + 1] - mrec[indices]) * mpre[indices + 1]))


@dataclass
class DetectionRecord:
    image_id: str
    class_id: int
    bbox: np.ndarray
    score: float = 1.0


def compute_detection_ap(
    predictions: Iterable[DetectionRecord],
    targets: Iterable[DetectionRecord],
    class_ids: Iterable[int],
    iou_threshold: float = 0.5,
) -> Dict[int, float]:
    """Compute AP@0.5 per class using one-to-one target matching."""
    pred_by_class: dict[int, list[DetectionRecord]] = {cid: [] for cid in class_ids}
    gt_by_class: dict[int, list[DetectionRecord]] = {cid: [] for cid in class_ids}

    for pred in predictions:
        if pred.class_id in pred_by_class:
            pred_by_class[pred.class_id].append(pred)
    for gt in targets:
        if gt.class_id in gt_by_class:
            gt_by_class[gt.class_id].append(gt)

    ap_per_class: Dict[int, float] = {}
    for class_id in class_ids:
        preds = sorted(pred_by_class[class_id], key=lambda x: x.score, reverse=True)
        gts = gt_by_class[class_id]
        if not gts:
            ap_per_class[class_id] = 0.0
            continue

        gt_lookup: dict[str, list[DetectionRecord]] = {}
        for gt in gts:
            gt_lookup.setdefault(gt.image_id, []).append(gt)
        matched = {img_id: np.zeros(len(items), dtype=bool) for img_id, items in gt_lookup.items()}

        tp = np.zeros(len(preds), dtype=float)
        fp = np.zeros(len(preds), dtype=float)

        for idx, pred in enumerate(preds):
            image_targets = gt_lookup.get(pred.image_id, [])
            if not image_targets:
                fp[idx] = 1.0
                continue

            ious = np.array([box_iou_xyxy(pred.bbox, gt.bbox) for gt in image_targets], dtype=float)
            best_i = int(np.argmax(ious))
            best_iou = ious[best_i]
            if best_iou >= iou_threshold and not matched[pred.image_id][best_i]:
                matched[pred.image_id][best_i] = True
                tp[idx] = 1.0
            else:
                fp[idx] = 1.0

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recall = tp_cum / max(len(gts), 1)
        precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
        ap_per_class[class_id] = _compute_ap(recall, precision) if preds else 0.0

    return ap_per_class

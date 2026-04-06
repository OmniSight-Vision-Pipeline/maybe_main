from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import torch

from adverse_vision.data.corruption import CorruptionConfig, apply_corruption
from adverse_vision.models.autoencoder import UNetAutoencoder
from adverse_vision.utils.runtime import configure_ultralytics_dir


class RestorationPipeline:
    def __init__(
        self,
        weights_path: str | Path | None = None,
        device: str = "cpu",
        img_size: tuple[int, int] = (640, 384),
        detector_model: str = "yolov8n.pt",
        enable_detector: bool = True,
    ) -> None:
        self.device = torch.device(device)
        self.img_size = img_size

        self.model = UNetAutoencoder().to(self.device).eval()
        if weights_path is not None:
            checkpoint = torch.load(Path(weights_path), map_location=self.device)
            state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
            self.model.load_state_dict(state_dict, strict=True)

        self.detector = None
        if enable_detector:
            try:
                configure_ultralytics_dir()
                from ultralytics import YOLO

                self.detector = YOLO(detector_model)
            except Exception:
                self.detector = None

    def _preprocess(self, frame: np.ndarray) -> tuple[torch.Tensor, tuple[int, int]]:
        original_hw = frame.shape[:2]
        resized = cv2.resize(frame, self.img_size, interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        return tensor.to(self.device), original_hw

    def _postprocess(self, pred: torch.Tensor, original_hw: tuple[int, int]) -> np.ndarray:
        img = pred.squeeze(0).detach().cpu().clamp(0.0, 1.0)
        img_np = (img.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        return cv2.resize(img_np, (original_hw[1], original_hw[0]), interpolation=cv2.INTER_CUBIC)

    def restore_frame(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Input frame must be HxWx3")
        tensor, original_hw = self._preprocess(frame)
        with torch.no_grad():
            pred = self.model(tensor)
        return self._postprocess(pred, original_hw)

    def detect(self, frame: np.ndarray) -> list[dict]:
        if self.detector is None:
            return []
        result = self.detector.predict(source=frame, device="cpu", verbose=False, conf=0.001)[0]
        if result.boxes is None:
            return []
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        return [
            {"bbox": boxes[i].astype(float).tolist(), "score": float(scores[i]), "class_id": int(classes[i])}
            for i in range(len(boxes))
        ]

    def run_pipeline(self, frame: np.ndarray, corruption_config: CorruptionConfig | None = None) -> dict:
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Input frame must be HxWx3")
        timings: dict[str, float] = {}

        working = frame
        metadata = None
        if corruption_config is not None:
            t0 = time.perf_counter()
            working, metadata = apply_corruption(frame, corruption_config)
            timings["corruption_ms"] = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        restored = self.restore_frame(working)
        timings["restore_ms"] = (time.perf_counter() - t1) * 1000.0

        t2 = time.perf_counter()
        detections = self.detect(restored)
        timings["detect_ms"] = (time.perf_counter() - t2) * 1000.0
        timings["total_ms"] = timings.get("corruption_ms", 0.0) + timings["restore_ms"] + timings["detect_ms"]

        return {
            "restored_frame": restored,
            "detections": detections,
            "timings_ms": timings,
            "corruption_metadata": metadata,
        }


_DEFAULT_PIPELINE: RestorationPipeline | None = None


def _get_default_pipeline() -> RestorationPipeline:
    global _DEFAULT_PIPELINE
    if _DEFAULT_PIPELINE is None:
        _DEFAULT_PIPELINE = RestorationPipeline(weights_path=None, enable_detector=False)
    return _DEFAULT_PIPELINE


def restore_frame(frame: np.ndarray, pipeline: RestorationPipeline | None = None) -> np.ndarray:
    active_pipeline = pipeline or _get_default_pipeline()
    return active_pipeline.restore_frame(frame)


def run_pipeline(
    frame: np.ndarray,
    pipeline: RestorationPipeline | None = None,
    corruption_config: CorruptionConfig | None = None,
) -> dict:
    active_pipeline = pipeline or _get_default_pipeline()
    return active_pipeline.run_pipeline(frame, corruption_config=corruption_config)

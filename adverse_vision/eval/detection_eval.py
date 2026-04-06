from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from adverse_vision.data.corruption import apply_corruption, build_deterministic_config
from adverse_vision.inference.pipeline import RestorationPipeline
from adverse_vision.utils.io import find_image_files
from adverse_vision.utils.metrics import DetectionRecord, compute_detection_ap
from adverse_vision.utils.runtime import configure_ultralytics_dir


def _load_rgb(path: Path, img_size: tuple[int, int]) -> np.ndarray:
    bgr = cv2.imread(path.as_posix(), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read image: {path}")
    bgr = cv2.resize(bgr, img_size, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _resolve_label_path(image_path: Path, test_root: Path) -> Path:
    rel = image_path.relative_to(test_root)
    if "images" in rel.parts:
        parts = list(rel.parts)
        idx = parts.index("images")
        parts[idx] = "labels"
        return test_root.joinpath(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def _load_yolo_labels(label_path: Path, image_id: str, width: int, height: int) -> list[DetectionRecord]:
    if not label_path.exists():
        return []
    records: list[DetectionRecord] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        values = line.split()
        if len(values) != 5:
            continue
        class_id = int(float(values[0]))
        cx, cy, bw, bh = [float(v) for v in values[1:]]
        x1 = (cx - bw / 2.0) * width
        y1 = (cy - bh / 2.0) * height
        x2 = (cx + bw / 2.0) * width
        y2 = (cy + bh / 2.0) * height
        records.append(DetectionRecord(image_id=image_id, class_id=class_id, bbox=np.array([x1, y1, x2, y2], dtype=float)))
    return records


class DetectorAdapter:
    def __init__(self, detector_model: str = "yolov8n.pt", device: str = "cpu") -> None:
        try:
            configure_ultralytics_dir()
            from ultralytics import YOLO
        except Exception as exc:
            raise RuntimeError("Ultralytics is required for detection evaluation.") from exc
        self.model = YOLO(detector_model)
        self.device = device

    def predict(self, image_rgb: np.ndarray) -> list[dict]:
        result = self.model.predict(
            source=image_rgb,
            device=self.device,
            verbose=False,
            conf=0.001,
            imgsz=max(image_rgb.shape[:2]),
        )[0]
        if result.boxes is None:
            return []
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        cls_ids = result.boxes.cls.cpu().numpy().astype(int)
        payload = []
        for box, score, class_id in zip(boxes, scores, cls_ids):
            payload.append({"bbox": box.astype(float), "score": float(score), "class_id": int(class_id)})
        return payload


def _to_records(predictions: Iterable[dict], image_id: str) -> list[DetectionRecord]:
    return [
        DetectionRecord(
            image_id=image_id,
            class_id=int(pred["class_id"]),
            bbox=np.array(pred["bbox"], dtype=float),
            score=float(pred["score"]),
        )
        for pred in predictions
    ]


def evaluate_detection_paths(
    test_root: str | Path,
    restore_weights: str | Path,
    detector_model: str = "yolov8n.pt",
    class_ids: tuple[int, ...] = (0, 1, 2),
    img_size: tuple[int, int] = (640, 384),
    seed: int = 42,
    metadata_path: str | Path | None = None,
) -> dict:
    test_root_path = Path(test_root)
    image_paths = find_image_files(test_root_path)
    detector = DetectorAdapter(detector_model=detector_model, device="cpu")
    pipeline = RestorationPipeline(weights_path=restore_weights, detector_model=detector_model, enable_detector=False, img_size=img_size)

    targets: list[DetectionRecord] = []
    clean_preds: list[DetectionRecord] = []
    corrupt_preds: list[DetectionRecord] = []
    restored_preds: list[DetectionRecord] = []
    metadata_records: list[dict] = []

    for index, image_path in enumerate(image_paths):
        image_id = image_path.as_posix()
        clean = _load_rgb(image_path, img_size=img_size)
        height, width = clean.shape[:2]

        label_path = _resolve_label_path(image_path, test_root_path)
        targets.extend(_load_yolo_labels(label_path, image_id=image_id, width=width, height=height))

        config = build_deterministic_config(index=index, seed=seed)
        corrupted, metadata = apply_corruption(clean, config)
        metadata_records.append({"image_id": image_id, "label_path": label_path.as_posix(), **metadata})

        clean_preds.extend(_to_records(detector.predict(clean), image_id=image_id))
        corrupt_preds.extend(_to_records(detector.predict(corrupted), image_id=image_id))
        restored = pipeline.restore_frame(corrupted)
        restored_preds.extend(_to_records(detector.predict(restored), image_id=image_id))

    if metadata_path:
        metadata_path = Path(metadata_path)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(metadata_records, indent=2), encoding="utf-8")

    clean_ap = compute_detection_ap(clean_preds, targets, class_ids)
    corrupt_ap = compute_detection_ap(corrupt_preds, targets, class_ids)
    restored_ap = compute_detection_ap(restored_preds, targets, class_ids)

    clean_map = float(np.mean([clean_ap[c] for c in class_ids])) if class_ids else 0.0
    corrupt_map = float(np.mean([corrupt_ap[c] for c in class_ids])) if class_ids else 0.0
    restored_map = float(np.mean([restored_ap[c] for c in class_ids])) if class_ids else 0.0
    denominator = clean_map - corrupt_map
    recovered_fraction = (restored_map - corrupt_map) / denominator if denominator > 1e-9 else 0.0

    class_name_map = {0: "person", 1: "bicycle", 2: "car"}
    class_metrics = {
        class_name_map.get(cid, f"class_{cid}"): {
            "clean_ap50": clean_ap[cid],
            "corrupted_ap50": corrupt_ap[cid],
            "restored_ap50": restored_ap[cid],
        }
        for cid in class_ids
    }

    return {
        "dataset": {"test_root": test_root_path.as_posix(), "num_images": len(image_paths)},
        "overall": {
            "clean_map50": clean_map,
            "corrupted_map50": corrupt_map,
            "restored_map50": restored_map,
            "recovered_map_drop_fraction": float(recovered_fraction),
        },
        "classes": class_metrics,
        "metadata_samples": metadata_records[:20],
    }

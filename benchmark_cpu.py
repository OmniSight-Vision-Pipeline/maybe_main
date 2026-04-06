from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from adverse_vision.inference import RestorationPipeline, benchmark_onnx_cpu, export_to_onnx
from adverse_vision.utils.io import find_image_files
from adverse_vision.utils.timing import latency_summary


def parse_img_size(raw: str) -> tuple[int, int]:
    raw = raw.lower().replace("x", ",")
    width, height = raw.split(",")
    return int(width.strip()), int(height.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark CPU restoration + detection latency.")
    parser.add_argument("--weights", required=True, type=str, help="Restoration checkpoint path.")
    parser.add_argument("--detector", default="yolov8n.pt", type=str, help="YOLO detector path.")
    parser.add_argument("--num-frames", default=100, type=int)
    parser.add_argument("--img-size", default="640x384", type=str)
    parser.add_argument("--test-root", default=None, type=str, help="Optional frame source root.")
    parser.add_argument("--warmup", default=10, type=int)
    parser.add_argument("--compare-onnx", action="store_true", help="Export model and benchmark ONNX Runtime CPU.")
    parser.add_argument("--onnx-path", default="artifacts/restore.onnx", type=str)
    parser.add_argument("--report-json", default=None, type=str)
    return parser.parse_args()


def _load_frames(test_root: str | None, num_frames: int, img_size: tuple[int, int]) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    if test_root:
        image_paths = find_image_files(test_root)
        for path in image_paths[:num_frames]:
            bgr = cv2.imread(path.as_posix(), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            bgr = cv2.resize(bgr, img_size, interpolation=cv2.INTER_AREA)
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    if not frames:
        frames = [np.random.randint(0, 256, (img_size[1], img_size[0], 3), dtype=np.uint8) for _ in range(num_frames)]
    return frames[:num_frames]


def main() -> None:
    args = parse_args()
    img_size = parse_img_size(args.img_size)
    frames = _load_frames(args.test_root, args.num_frames, img_size)

    pipeline = RestorationPipeline(
        weights_path=args.weights,
        device="cpu",
        img_size=img_size,
        detector_model=args.detector,
        enable_detector=True,
    )

    for frame in frames[: args.warmup]:
        _ = pipeline.restore_frame(frame)
        _ = pipeline.detect(frame)
        _ = pipeline.run_pipeline(frame)

    restore_lat = []
    detect_lat = []
    end_to_end_lat = []
    for frame in frames:
        t0 = time.perf_counter()
        _ = pipeline.restore_frame(frame)
        restore_lat.append((time.perf_counter() - t0) * 1000.0)

        t1 = time.perf_counter()
        _ = pipeline.detect(frame)
        detect_lat.append((time.perf_counter() - t1) * 1000.0)

        t2 = time.perf_counter()
        _ = pipeline.run_pipeline(frame)
        end_to_end_lat.append((time.perf_counter() - t2) * 1000.0)

    report: dict = {
        "frames": len(frames),
        "restore_cpu_ms": latency_summary(restore_lat),
        "detect_cpu_ms": latency_summary(detect_lat),
        "restore_plus_detect_cpu_ms": latency_summary(end_to_end_lat),
    }

    if args.compare_onnx:
        onnx_path = export_to_onnx(args.weights, args.onnx_path, img_size=img_size)
        report["onnx_path"] = onnx_path
        report["restore_onnx_cpu_ms"] = benchmark_onnx_cpu(onnx_path, num_iters=max(len(frames), 20), img_size=img_size)

    if args.report_json:
        report_path = Path(args.report_json)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

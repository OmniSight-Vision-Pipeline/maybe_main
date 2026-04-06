from __future__ import annotations

import argparse

import cv2
import numpy as np

from adverse_vision.inference import RestorationPipeline


def parse_img_size(raw: str) -> tuple[int, int]:
    raw = raw.lower().replace("x", ",")
    width, height = raw.split(",")
    return int(width.strip()), int(height.strip())


def draw_detections(image_bgr: np.ndarray, detections: list[dict]) -> np.ndarray:
    out = image_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        class_id = det["class_id"]
        score = det["score"]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, f"id:{class_id} {score:.2f}", (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime restoration + detection demo.")
    parser.add_argument("--weights", required=True, type=str, help="Restoration checkpoint path.")
    parser.add_argument("--detector", default="yolov8n.pt", type=str, help="YOLO detector weights path.")
    parser.add_argument("--source", default="0", type=str, help="Video source index or file path.")
    parser.add_argument("--img-size", default="640x384", type=str)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    img_size = parse_img_size(args.img_size)
    source = int(args.source) if args.source.isdigit() else args.source

    pipeline = RestorationPipeline(
        weights_path=args.weights,
        device="cpu",
        img_size=img_size,
        detector_model=args.detector,
        enable_detector=True,
    )

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {args.source}")

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = pipeline.run_pipeline(frame_rgb)
        restored_rgb = result["restored_frame"]
        detections = result["detections"]

        restored_bgr = cv2.cvtColor(restored_rgb, cv2.COLOR_RGB2BGR)
        overlay_bgr = draw_detections(restored_bgr, detections)
        panel = np.hstack(
            [
                cv2.resize(frame_bgr, (img_size[0], img_size[1]), interpolation=cv2.INTER_AREA),
                cv2.resize(overlay_bgr, (img_size[0], img_size[1]), interpolation=cv2.INTER_AREA),
            ]
        )
        cv2.putText(
            panel,
            f"restore {result['timings_ms']['restore_ms']:.1f} ms | detect {result['timings_ms']['detect_ms']:.1f} ms",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("Adverse Vision Demo (left: input, right: restored+detections)", panel)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

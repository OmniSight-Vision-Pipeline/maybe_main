from __future__ import annotations

import argparse
import json
from pathlib import Path

from adverse_vision.eval import evaluate_detection_paths


def parse_img_size(raw: str) -> tuple[int, int]:
    raw = raw.lower().replace("x", ",")
    width, height = raw.split(",")
    return int(width.strip()), int(height.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate clean/corrupt/restored detection mAP50.")
    parser.add_argument("--weights", required=True, type=str, help="Restoration model checkpoint path.")
    parser.add_argument("--detector", default="yolov8n.pt", type=str, help="YOLO detector weights.")
    parser.add_argument("--test-root", required=True, type=str, help="Test image root.")
    parser.add_argument("--report-json", required=True, type=str, help="Output JSON report path.")
    parser.add_argument("--img-size", default="640x384", type=str)
    parser.add_argument("--metadata-json", default=None, type=str, help="Optional corruption metadata output file.")
    parser.add_argument("--seed", default=42, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    img_size = parse_img_size(args.img_size)
    report = evaluate_detection_paths(
        test_root=args.test_root,
        restore_weights=args.weights,
        detector_model=args.detector,
        img_size=img_size,
        seed=args.seed,
        metadata_path=args.metadata_json,
    )
    out_path = Path(args.report_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["overall"], indent=2))


if __name__ == "__main__":
    main()

# Adverse Vision MVP (Beginner Friendly)

This project restores adverse-weather frames (rain, fog, low light) before detection.  
It is designed as a learning-first research MVP with a clean structure and reproducible steps.

## 1) What You Are Building

We train a restoration ANN (autoencoder with skip connections) that learns:
- Input: corrupted road scene image
- Output: clean version of the same scene

Then we measure whether running YOLO on the restored frame recovers detection quality compared to:
- Clean frame -> YOLO (upper bound)
- Corrupted frame -> YOLO (failure baseline)
- Corrupted frame -> Restoration ANN -> YOLO (rescue path)

## 2) Project Structure

```text
adverse_vision/
  data/
    corruption.py        # Synthetic rain/fog/dark generation + metadata
    dataset.py           # Dataset for clean/corrupted pairs
    split.py             # 70/15/15 split utilities
  models/
    autoencoder.py       # U-Net style autoencoder
    losses.py            # Composite loss: L1 + SSIM + perceptual
  train/
    trainer.py           # Training loop, checkpoints, history
  eval/
    detection_eval.py    # mAP50 evaluation for clean/corrupt/restored paths
  inference/
    pipeline.py          # Public APIs: restore_frame, run_pipeline
    onnx_utils.py        # ONNX export and CPU benchmark
  utils/
    io.py                # File discovery helpers
    metrics.py           # SSIM/PSNR + AP computation
    seed.py              # Determinism helpers
    timing.py            # p50/p95 latency summary

train_restore.py
eval_restore_and_detect.py
benchmark_cpu.py
run_realtime_demo.py
tests/
docs/
```

## 3) Environment Setup (Inside `soft_comp`)

### Windows

```powershell
.\scripts\setup_windows.ps1
.\soft_comp\Scripts\activate
```

### macOS / Linux

```bash
bash ./scripts/setup_unix.sh
source soft_comp/bin/activate
```

Manual fallback:

```bash
python3 -m venv soft_comp
source soft_comp/bin/activate
python -m pip install -r requirements.txt
```

Verify install:

```bash
python -m pytest -q
```

Note: this project auto-creates a local `.ultralytics` folder in the workspace so YOLO settings stay writable.

## 4) Data Layout

For easiest use, keep clear driving images under one folder:

```text
D:\datasets\bdd100k\images\100k\train
D:\datasets\bdd100k\images\100k\val
```

For detection evaluation, labels should be YOLO-format `.txt` files.  
If your path contains an `images` folder, the evaluator will automatically look for matching files under `labels`.

Expected YOLO label line format:

```text
class_id center_x center_y width height
```

All values except `class_id` should be normalized to `[0,1]`.

## 5) Train the Restoration Model

Example:

```powershell
python train_restore.py `
  --data-root D:\datasets\bdd100k\images\100k\train `
  --out-dir artifacts\run1 `
  --img-size 640x384 `
  --epochs 80 `
  --batch-size 16 `
  --lr 1e-4 `
  --patience 10 `
  --seed 42
```

Artifacts generated:
- `artifacts/run1/best.pt`
- `artifacts/run1/last.pt`
- `artifacts/run1/history.json`
- `artifacts/run1/train_summary.json`
- `artifacts/run1/split_manifest.json`

## 6) Evaluate Detection Recovery

Example:

```powershell
python eval_restore_and_detect.py `
  --weights artifacts\run1\best.pt `
  --detector yolov8n.pt `
  --test-root D:\datasets\bdd100k\images\100k\val `
  --report-json artifacts\eval\report.json `
  --metadata-json artifacts\eval\corruption_meta.json `
  --img-size 640x384 `
  --seed 42
```

Read `report.json` and focus on:
- `overall.clean_map50`
- `overall.corrupted_map50`
- `overall.restored_map50`
- `overall.recovered_map_drop_fraction`

Goal interpretation:
- If `recovered_map_drop_fraction >= 0.70`, your restoration model recovered 70% of the mAP drop caused by corruption.

## 7) CPU Latency Benchmark

Example:

```powershell
python benchmark_cpu.py `
  --weights artifacts\run1\best.pt `
  --detector yolov8n.pt `
  --num-frames 200 `
  --img-size 640x384 `
  --test-root D:\datasets\bdd100k\images\100k\val `
  --compare-onnx `
  --onnx-path artifacts\bench\restore.onnx `
  --report-json artifacts\bench\cpu.json
```

Check:
- `restore_cpu_ms.p50_ms`
- `detect_cpu_ms.p50_ms`
- `restore_plus_detect_cpu_ms.p50_ms`
- Optional ONNX numbers when `--compare-onnx` is enabled.

## 8) Realtime Demo

Webcam:

```powershell
python run_realtime_demo.py --weights artifacts\run1\best.pt --detector yolov8n.pt --source 0 --img-size 640x384
```

Video file:

```powershell
python run_realtime_demo.py --weights artifacts\run1\best.pt --detector yolov8n.pt --source D:\videos\drive.mp4 --img-size 640x384
```

Press `q` or `Esc` to exit.

## 9) Public APIs

From Python:

```python
import numpy as np
from adverse_vision.data.corruption import CorruptionConfig
from adverse_vision.inference.pipeline import restore_frame, run_pipeline, RestorationPipeline

frame = np.zeros((720, 1280, 3), dtype=np.uint8)
pipeline = RestorationPipeline(weights_path="artifacts/run1/best.pt", enable_detector=False)

restored = restore_frame(frame, pipeline=pipeline)
result = run_pipeline(frame, pipeline=pipeline, corruption_config=CorruptionConfig(fog_level=3, seed=42))
```

## 10) Testing

Run all tests:

```powershell
python -m pytest -q
```

Tests include:
- Corruption determinism
- Severity monotonicity checks
- Model shape/range invariants
- One-epoch training smoke test
- End-to-end pipeline interface test

## 11) Beginner Troubleshooting

If `torch` install fails:
- Confirm Python version compatibility with your Torch build.
- Try CPU-only Torch wheel commands from official PyTorch instructions.

If YOLO weights fail to auto-download:
- Download `yolov8n.pt` manually and pass full path via `--detector`.

If you get an Ultralytics permission error on Windows:
- Keep running commands from this project root.
- The code will route Ultralytics config to `D:\Soft computing\.ultralytics`.

If dataset read fails:
- Ensure images are valid and file extensions are supported (`.jpg`, `.png`, ...).
- Confirm `--data-root` and `--test-root` paths are correct.

If runtime is slow:
- Reduce `--img-size` (for example `512x320`).
- Use smaller batch size for training.
- Disable detector in pipeline when benchmarking restoration only.

## 12) Learning Path

Recommended order for beginners:
1. Run tests
2. Train on a tiny subset
3. Run evaluation
4. Run benchmark
5. Run realtime demo
6. Tune corruption severity and compare mAP recovery

Detailed beginner docs are in:
- `docs/BEGINNER_QUICKSTART.md`
- `docs/ARCHITECTURE_EXPLAINED.md`
- `docs/COMMAND_REFERENCE.md`
- `docs/CROSS_PLATFORM_SETUP.md`
- `docs/GITHUB_SETUP.md`

## 13) GitHub Collaboration

This repository is set up with GitHub community standards:
- `LICENSE` (MIT)
- `CONTRIBUTING.md`
- `CODE_OF_CONDUCT.md`
- `SECURITY.md`
- PR template and issue templates under `.github/`
- CI workflow at `.github/workflows/ci.yml`
- Dependabot config at `.github/dependabot.yml`

Recommended flow:
1. Create a branch for one focused change.
2. Run `python -m pytest -q`.
3. Open a PR and fill out the PR template.

## 14) What Not To Commit

The `.gitignore` is configured to exclude local-only files such as:
- `soft_comp/` (virtual environment)
- `artifacts/` (generated training/eval outputs)
- `__pycache__/` and test caches
- `.ultralytics/` local YOLO settings/cache

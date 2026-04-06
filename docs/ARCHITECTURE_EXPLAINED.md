# Architecture Explained (For Beginners)

## End-to-End Flow

1. Start with a clean driving frame.
2. Apply synthetic corruption (rain/fog/dark).
3. Feed corrupted frame into restoration ANN.
4. ANN predicts restored frame.
5. YOLO detector runs on restored frame.
6. Compare detector quality across clean/corrupt/restored paths.

## Data Pipeline

`adverse_vision/data/split.py`
- Creates deterministic train/val/test split with seed 42.
- Ratio is fixed to 70/15/15.

`adverse_vision/data/corruption.py`
- `CorruptionConfig` describes severity and mode.
- `apply_corruption` returns:
  - corrupted frame
  - metadata dictionary used for reproducibility

Corruption operators:
- Rain: synthetic streak overlay + blur
- Fog: atmospheric scattering approximation
- Dark: gamma reduction + shot/read noise

`adverse_vision/data/dataset.py`
- Loads clean images.
- Builds deterministic corruption config per sample index.
- Returns dictionary with:
  - `corrupted` tensor
  - `clean` tensor
  - `metadata`
  - `path`

## Model

`adverse_vision/models/autoencoder.py`
- U-Net style autoencoder.
- Encoder channels:
  - 32 -> 64 -> 128 -> 256
- Bottleneck:
  - 512
- Decoder mirrors encoder with skip connections.
- Final activation is sigmoid (output range [0,1]).

Why skip connections help:
- They preserve local detail (edges, object boundaries).
- Important for detector-friendly restoration.

## Loss Function

`adverse_vision/models/losses.py`

Composite objective:
- `0.6 * L1`
- `0.3 * (1 - SSIM)`
- `0.1 * Perceptual`

Meaning:
- L1 stabilizes pixel reconstruction.
- SSIM encourages structural similarity.
- Perceptual encourages feature-level realism.

## Training

`adverse_vision/train/trainer.py`

Key defaults:
- Optimizer: AdamW
- LR: 1e-4
- Scheduler: CosineAnnealingLR
- Early stop patience: 10
- Checkpoint selection metric:
  - `val_psnr + 20 * val_ssim`

Artifacts:
- `best.pt`
- `last.pt`
- `history.json`

## Evaluation

`adverse_vision/eval/detection_eval.py`

Three-path comparison:
- Clean -> YOLO
- Corrupted -> YOLO
- Corrupted -> Restore -> YOLO

Outputs:
- class-level AP@0.5
- overall mAP@0.5
- recovered drop fraction

Recovered fraction formula:

```text
(restored_map - corrupted_map) / (clean_map - corrupted_map)
```

## Inference APIs

`adverse_vision/inference/pipeline.py`

Main interfaces:
- `restore_frame(frame) -> restored_frame`
- `run_pipeline(frame, corruption_config=None) -> dict`

`run_pipeline` returns:
- `restored_frame`
- `detections`
- `timings_ms`
- `corruption_metadata`

## ONNX and Benchmarking

`adverse_vision/inference/onnx_utils.py`
- Export PyTorch model to ONNX
- Benchmark ONNX Runtime on CPU

`benchmark_cpu.py`
- Reports p50/p95 latency for:
  - restore only
  - detect only
  - restore + detect

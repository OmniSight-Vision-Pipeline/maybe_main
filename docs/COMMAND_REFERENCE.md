# Command Reference

All commands below assume:
- Current directory is `D:\Soft computing`
- Virtual environment is activated: `.\soft_comp\Scripts\activate`

## Install Dependencies

```powershell
python -m pip install -r requirements.txt
```

## Train

```powershell
python train_restore.py --data-root <path_to_clear_images> --out-dir artifacts\run1 --img-size 640x384 --epochs 80 --batch-size 16
```

### Important Flags
- `--data-root`: required, clear images
- `--out-dir`: required, output folder
- `--img-size`: WIDTHxHEIGHT
- `--epochs`: default 80
- `--batch-size`: default 16
- `--lr`: default 1e-4
- `--patience`: default 10
- `--seed`: default 42

## Evaluate Recovery

```powershell
python eval_restore_and_detect.py --weights artifacts\run1\best.pt --detector yolov8n.pt --test-root <path_to_test_images> --report-json artifacts\eval\report.json --metadata-json artifacts\eval\corruption_meta.json
```

### Important Flags
- `--weights`: required checkpoint
- `--detector`: YOLO weights path/name
- `--test-root`: required test images root
- `--report-json`: required output report
- `--metadata-json`: optional corruption log output

## Benchmark CPU

```powershell
python benchmark_cpu.py --weights artifacts\run1\best.pt --detector yolov8n.pt --num-frames 200 --img-size 640x384 --compare-onnx --report-json artifacts\bench\cpu.json
```

### Important Flags
- `--num-frames`: benchmark sample size
- `--warmup`: warmup frames before timing
- `--compare-onnx`: include ONNX export + benchmark
- `--onnx-path`: output ONNX file path

## Realtime Demo

```powershell
python run_realtime_demo.py --weights artifacts\run1\best.pt --detector yolov8n.pt --source 0 --img-size 640x384
```

## Run Tests

```powershell
python -m pytest -q
```

## Helpful Environment Checks

```powershell
python --version
python -m pip --version
python -c "import torch; print(torch.__version__)"
python -c "import cv2; print(cv2.__version__)"
```

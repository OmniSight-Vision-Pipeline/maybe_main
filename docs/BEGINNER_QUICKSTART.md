# Beginner Quickstart

This guide is intentionally verbose and practical. Follow it line-by-line.

## Step 0: Open the Correct Folder

Use a terminal opened in:

```text
D:\Soft computing
```

Check:

```powershell
Get-Location
```

## Step 1: Create The Virtual Environment

On Windows:

```powershell
.\scripts\setup_windows.ps1
.\soft_comp\Scripts\activate
```

On macOS / Linux:

```bash
bash ./scripts/setup_unix.sh
source soft_comp/bin/activate
```

You should see `(soft_comp)` in your terminal prompt after activation.

## Step 2: Verify The Environment

```bash
python -m pytest -q
```

## Step 3: Run Tests First

If tests pass, your setup is healthy.

## Step 4: Prepare a Small Dataset Subset

For first run, do not use full dataset. Use 200 to 500 images.

Example folder:

```text
D:\datasets\bdd100k\images\toy_train
```

## Step 5: Train a Small Model Run

```powershell
python train_restore.py `
  --data-root D:\datasets\bdd100k\images\toy_train `
  --out-dir artifacts\toy_run `
  --img-size 640x384 `
  --epochs 5 `
  --batch-size 4
```

## Step 6: Evaluate Detection Recovery

```powershell
python eval_restore_and_detect.py `
  --weights artifacts\toy_run\best.pt `
  --detector yolov8n.pt `
  --test-root D:\datasets\bdd100k\images\100k\val `
  --report-json artifacts\toy_eval\report.json `
  --metadata-json artifacts\toy_eval\corruption_meta.json
```

Open `artifacts\toy_eval\report.json` and check `overall`.

## Step 7: Benchmark CPU Speed

```powershell
python benchmark_cpu.py `
  --weights artifacts\toy_run\best.pt `
  --detector yolov8n.pt `
  --num-frames 100 `
  --img-size 640x384 `
  --report-json artifacts\toy_bench\cpu.json
```

## Step 8: Realtime Demo

```powershell
python run_realtime_demo.py --weights artifacts\toy_run\best.pt --detector yolov8n.pt --source 0
```

## Common Beginner Mistakes

- Running commands without activating `soft_comp`
- Wrong dataset path
- Missing YOLO labels in evaluation
- Using very large image size on CPU
- Starting with too many epochs before a smoke run

## Suggested Next Experiments

- Train with only fog corruption and evaluate
- Train with only low-light corruption and evaluate
- Compare `640x384` vs `512x320` latency
- Try combined corruption and measure recovered mAP drop fraction

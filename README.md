# OmniSight BDD100k-10k — Image Restoration

My contribution to the OmniSight project: training the adverse vision restoration model on the BDD100k 10k subset.

## Results
| Metric | Value |
|--------|-------|
| Epochs | 20 |
| Best Epoch | 19 |
| PSNR | ~23 dB |
| SSIM | ~0.80 |
| Train Size | 4,900 images |
| Val Size | 1,050 images |

## Dataset
BDD100k-10k subset via Kaggle: `solesensei/solesensei_bdd100k`

## Training
```bash
python train_restore.py \
    --data-root /path/to/bdd100k/images/10k/train \
    --out-dir artifacts/v2 \
    --img-size 320x192 \
    --epochs 20 \
    --batch-size 8
```

## Inference
```python
from adverse_vision.inference.pipeline import RestorationPipeline
pipeline = RestorationPipeline(weights_path='artifacts/v2/best.pt', device='cuda')
restored = pipeline.restore_frame(img_array)
```

## Part of
[OmniSight](https://github.com/Gunjan00001/OmniSight-Vis) — Adverse Vision Restoration System

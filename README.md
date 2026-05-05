# Clear Vision / OmniSight: Robust Autonomous Vision Restoration

This repository contains the OmniSight pipeline and advanced training scripts for robust autonomous vision restoration. It is designed to remove rain, fog, and lens distortion from camera feeds to improve downstream object detection (e.g., YOLOv8) performance in adverse weather conditions. The codebase utilizes state-of-the-art **Dual-Teacher Distillation** to compress heavy Restormer models into a fast, lightweight U-Net architecture for edge deployment.

## Key Features
*   **Dual-Teacher Distillation**: Uses an ensemble of multiple pre-trained Restormer models (e.g., a generic restorer and a Cityscapes-specific model) to generate high-quality pseudo-ground-truth targets for a student U-Net.
*   **Weight Blending**: Includes utilities to merge the weights of multiple checkpoints to create unified hybrid models.
*   **Temporal U-Net Student**: A lightweight 6-channel U-Net designed to process consecutive frames (T and T-1) for temporally consistent and real-time video restoration.

## Project Structure

- `train_restore.py`: Base script for training a Restormer Teacher model on paired clean and synthetically corrupted images.
- `train_cityscapes.py`: Fine-tunes a base Restormer Teacher specifically on the Cityscapes dataset using `CharbonnierLoss` to improve edge restoration.
- `train_unet.py`: Basic script for training a U-Net Student from scratch without a pretrained teacher (baseline).
- `train_unet_from_pretrained.py`: Distills a single Teacher's knowledge (or a blended teacher) into the U-Net Student model using a combination of Charbonnier, MSE, and SSIM-proxy losses.
- `train_dual_teacher.py`: The advanced distillation script. Passes noisy inputs to TWO Teacher models, averages their outputs, and forces the Student U-Net to mimic this ensemble prediction.
- `blend_models.py`: A utility script to perform weight-averaging between two PyTorch models, creating a hybrid checkpoint (`blended_model.pt`).
- `compare_models.py`: Evaluates and compares the visual output of the Student U-Net against the heavy Restormer Teacher models.
- `test_shapes.py`: Utility script to verify tensor dimension transformations through the Restormer and U-Net pipelines.
- `run_image_restoration.py`: Standalone inference script to run the trained model on an image and visualize the results.
- `eval_restore_and_detect.py`: Evaluates the baseline (YOLO directly on corrupted) vs. OmniSight pipeline (Student U-Net -> YOLO) on video streams using ByteTrack tracking.
- `data/`: Dataset handling, corruption scripts (adding rain, fog, lens distortion).
- `models/`: Architectures for Restormer and UNet.
- `inference/`: OmniSight pipeline combining U-Net restoration and YOLOv8 tracking.

## Hardware & Environment Setup

This project is optimized to run on standard PyTorch (CPU or CUDA). It has been carefully structured to avoid dependency conflicts on Windows environments. 

### 1. Create a Virtual Environment
It is highly recommended to use a virtual environment (`soft_venv`) to isolate dependencies.
```powershell
python -m venv soft_venv
.\soft_venv\Scripts\Activate.ps1
```

### 2. Install Stable Dependencies
To prevent compatibility issues (specifically `Numpy is not available` crashes when PyTorch native bindings clash with Numpy 2.x), install the exact pinned versions below:
```powershell
pip install numpy==1.26.4 opencv-python-headless==4.8.1.78 albumentations==1.3.1
pip install torch torchvision torchaudio
```

## Git LFS (Large File Storage)
Pre-trained model weights (`*.pt`) are tracked using **Git LFS**. To ensure you download the actual model files and not text pointers, ensure Git LFS is installed and pull the objects:
```bash
git lfs install
git lfs pull
```
*   `teacher_best.pt`: Best performing base Restormer model.
*   `cityscapes_final.pt`: Restormer model fine-tuned on the Cityscapes dataset.
*   `blended_model.pt`: An averaged hybrid model created by blending weights.

## How to Train

### 1. Data Preparation
The dataset loader expects the following directory structure for training (e.g., BDD100k or Cityscapes):
```text
data/
  bdd100k/
    image1.jpg
    image2.jpg
    ...
```

### 2. Fine-Tuning a Teacher (Optional)
If you have a base Restormer (`teacher_best.pt`) and want to adapt it to a new environment like Cityscapes:
```bash
python train_cityscapes.py --batch_size 4 --epochs 30 --lr 1e-5
```
This uses a lower learning rate and `CharbonnierLoss` to preserve edges, outputting `cityscapes_final.pt`.

### 3. Model Weight Blending (Optional)
To merge two different teacher models (e.g., Base and Cityscapes) into a single unified checkpoint without actively running two models during inference:
```bash
python blend_models.py teacher_best.pt cityscapes_final.pt blended_model.pt 0.5
```
This creates a 50/50 state dictionary average.

### 4. Knowledge Distillation

**Option A: Advanced Single-Teacher Distillation**
Train the U-Net Student from the blended model using `PerceptualDistillationLoss` (MSE + Charbonnier + SSIM-proxy) and Automatic Mixed Precision (AMP) for speed.
```bash
python train_unet_from_pretrained.py --teacher_weights blended_model.pt --batch_size 8 --epochs 50
```

**Option B: Dual-Teacher Ensemble Distillation**
Alternatively, load both `teacher_best.pt` and `cityscapes_final.pt` into GPU memory, freeze them, and use them as a live ensemble "oracle" to teach the lightweight U-Net student.
```bash
python train_dual_teacher.py --batch_size 4 --epochs 50
```
This script handles 6-channel input concatenation (T and T-1 frames) and saves `dual_distilled_unet_best.pt`.

## Evaluation and Inference

To visually evaluate the models' pixel-level restoration capabilities on a sample image:
```bash
python compare_models.py
```
This runs an image through the frozen Teachers and the Student U-Net side-by-side.

To run end-to-end evaluation on a sample video to compare the baseline (YOLOv8 directly on corrupted) against the OmniSight pipeline (Student U-Net -> YOLOv8 + ByteTrack):
```bash
python eval_restore_and_detect.py --video path/to/your/video.mp4
```
This script tracks detections using ByteTrack and outputs the total valid detections to help compute mAP/MOTA metrics. If no video is provided, it generates a dummy video for pipeline validation.

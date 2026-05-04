# Clear Vision / OmniSight: Robust Autonomous Vision Restoration

This repository contains the OmniSight pipeline and advanced training scripts for robust autonomous vision restoration. It is designed to remove rain, fog, and lens distortion from camera feeds to improve downstream object detection (e.g., YOLOv8) performance in adverse weather conditions. The codebase utilizes state-of-the-art **Dual-Teacher Distillation** to compress heavy Restormer models into a fast, lightweight U-Net architecture for edge deployment.

## Key Features
*   **Dual-Teacher Distillation**: Uses an ensemble of multiple pre-trained Restormer models (e.g., a generic restorer and a Cityscapes-specific model) to generate high-quality pseudo-ground-truth targets for a student U-Net.
*   **Weight Blending**: Includes utilities to merge the weights of multiple checkpoints to create unified hybrid models.
*   **Temporal U-Net Student**: A lightweight 6-channel U-Net designed to process consecutive frames (T and T-1) for temporally consistent and real-time video restoration.

## Project Structure

- `train_restore.py`: Base script for training a Restormer Teacher model on paired clean and synthetically corrupted images.
- `train_unet_from_pretrained.py`: Distills a single Teacher's knowledge into the U-Net Student model.
- `train_dual_teacher.py`: The advanced distillation script. Passes noisy inputs to TWO Teacher models, averages their outputs, and forces the Student U-Net to mimic this ensemble prediction.
- `blend_models.py`: A utility script to perform weight-averaging between two PyTorch models, creating a hybrid checkpoint (`blended_model.pt`).
- `compare_models.py`: Evaluates and compares the visual output of the Student U-Net against the heavy Restormer Teacher models.
- `run_image_restoration.py`: Standalone inference script to run the trained model on an image and visualize the results.
- `eval_restore_and_detect.py`: Evaluates the baseline (YOLO directly on corrupted) vs. OmniSight pipeline (Student U-Net -> YOLO) on video streams.
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
The dataset loader expects the following directory structure for training (e.g., BDD100k):
```text
data/
  bdd100k/
    image1.jpg
    image2.jpg
    ...
```

### 2. Dual-Teacher Distillation (Recommended)
Instead of training a U-Net from scratch or using a single teacher, use the `train_dual_teacher.py` script. This script loads both `teacher_best.pt` and `cityscapes_final.pt`, freezes them, and uses them as an ensemble "oracle" to teach the lightweight U-Net student.

```bash
python train_dual_teacher.py --batch_size 4 --epochs 50
```
This script handles 6-channel input concatenation (T and T-1 frames) and will automatically save `dual_distilled_unet_best.pt` upon improvement. It runs on the CPU by default unless a CUDA-capable GPU is detected.

### 3. Model Weight Blending (Optional)
If you want to permanently merge two models into a single checkpoint instead of doing active ensemble inference during training, use the blending script:
```bash
python blend_models.py
```
This merges `teacher_best.pt` and `cityscapes_final.pt` into a single `blended_model.pt` via a 50/50 state dictionary average.

## Evaluation and Inference

To visually evaluate how well the models are performing on an image:
```bash
python compare_models.py
```
This script will run an image through the frozen Teacher and the Student U-Net side-by-side, allowing you to gauge the distillation quality.

To run standard restoration on a sample video to compare the baseline YOLOv8 with the OmniSight pipeline:
```bash
python eval_restore_and_detect.py --video path/to/your/video.mp4
```
If no video is provided, the script will generate a dummy video for testing to ensure the pipeline runs correctly.

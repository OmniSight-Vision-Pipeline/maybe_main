# Clear Vision / OmniSight: Robust Autonomous Vision Restoration

This repository contains the OmniSight pipeline and training scripts for robust autonomous vision restoration. It is designed to remove rain, fog, and lens distortion from video feeds to improve downstream object detection (e.g., YOLOv8) performance in adverse weather conditions. The codebase is designed to be fully compatible with Linux environments for seamless cloud training and deployment.

## 👩‍💻 Contributors

### Anushka
- Contributed to early development of the restoration pipeline  
- Trained baseline models on the BDD100k-10k subset (Kaggle)  
- Established initial performance benchmarks (PSNR ~23 dB, SSIM ~0.80)  
- Worked on training setup, experimentation, and validation  
- Contributed to the foundation used for later architectural improvements  

🔗 Related work:
- https://github.com/Anushka-XD/omnisight-bdd10k  
- https://github.com/Anushka-XD/Restormer-architecture_omnisight-  

---

### Gunjan
- Led development of the current version of the project  
- Improved model architecture and overall system design  
- Expanded dataset usage and refined training strategy  
- Implemented temporal distillation and pipeline optimization  
- Structured and documented the repository in its current form 

## Project Structure

- `train_restore.py`: Trains the Restormer Teacher model on paired clean and synthetically corrupted images.
- `train_distillation.py`: Distills the Teacher's knowledge into a fast U-Net Student model using temporal pairs (T and T-1).
- `eval_restore_and_detect.py`: Evaluates the baseline (YOLO directly on corrupted) vs. OmniSight pipeline (Student U-Net -> YOLO) on a video.
- `data/`: Dataset handling, corruption scripts (adding rain, fog, lens distortion).
- `models/`: Architectures for Restormer and UNet.
- `inference/`: OmniSight pipeline combining U-Net restoration and YOLOv8 tracking.

## Requirements

The project uses Python and standard deep learning libraries. To install the dependencies, run:

```bash
pip install -r requirements.txt
```

*(Note: Install PyTorch according to your system's CUDA version from the [official PyTorch site](https://pytorch.org/).)*

## How to Train

### 1. Data Preparation

The dataset loader expects the following directory structure for training:

```text
data/
  bdd100k/
    image1.jpg
    image2.jpg
    ...
  real_rain/
    image1.jpg
    image2.jpg
    ...
```

You can adjust the dataset directory paths passed to `OmniSightDataset` inside `train_restore.py` and `train_distillation.py` if your data is located elsewhere.

### 2. Train the Teacher Model (Restormer)

The Teacher model learns to restore single frames from heavy corruption using a Charbonnier loss against clean ground truth. It acts as the oracle for the student model later.

Run the training script:

```bash
python train_restore.py --batch_size 4 --epochs 50 --lr 2e-4
```

This script will run and automatically save `teacher_best.pt` when the validation/training loss improves.

### 3. Train the Student Model (Temporal U-Net Distillation)

The Student model (U-Net) is designed to be lightweight and fast for edge deployment. It takes two consecutive frames (T and T-1) to restore the current frame. It is trained to mimic the frozen Teacher model's output on the same frames, so it doesn't need clean ground truth during distillation.

Ensure `teacher_best.pt` is present in the root directory (or update the script if it's stored elsewhere).

Run the distillation script:

```bash
python train_distillation.py --batch_size 4 --epochs 50 --lr 1e-4
```

This script will save `distilled_unet_best.pt` upon improvement.

## Evaluation and Inference

You can test the pipeline on a sample video to compare the baseline YOLOv8 with the OmniSight pipeline (Restoration + YOLOv8).

```bash
python eval_restore_and_detect.py --video path/to/your/video.mp4
```

If no video is provided or the specified video is not found, the script will generate a dummy video for testing to ensure the pipeline runs correctly.

The `inference/pipeline.py` script also provides a `run_video` method to visually see the side-by-side comparison in a desktop environment.

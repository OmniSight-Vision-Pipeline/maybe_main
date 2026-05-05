# OmniSight Code Architecture & Flow Details

This document breaks down the internal workings, algorithms, and execution flow of the OmniSight Vision Restoration project.

## 1. Overall System Flow

The pipeline operates on the principle of **Knowledge Distillation**. We have powerful but slow "Teacher" models (Restormer architecture) and a fast but simple "Student" model (U-Net architecture). 

The goal is to teach the Student to mimic the Teacher so that the Student can be deployed in real-time edge environments to clean noisy inputs before object detection (YOLOv8).

### High-Level Training Flow:
1. Load noisy frames (current frame `T`, previous frame `T-1`).
2. Pass noisy frame `T` to the heavy Teacher model(s) to get a high-quality restored image (the "Pseudo-Ground-Truth").
3. Concatenate frames `T` and `T-1` into a 6-channel input.
4. Pass the 6-channel input to the lightweight Student U-Net.
5. Calculate the loss (difference) between the Student's output and the Teacher's output.
6. Update the Student's weights.

---

## 2. Detailed Breakdown of Scripts

### A. `train_dual_teacher.py` (Ensemble Distillation)
This is the core script for advanced distillation using two distinct teachers.
* **Flow**:
  1. Initializes the `OmniSightDataset` (BDD100k).
  2. Loads **two** `Restormer` models (`teacher_best.pt` and `cityscapes_final.pt`) onto the device.
  3. Freezes both Teachers (`requires_grad = False`).
  4. Loads the `UNet` Student model.
  5. **Training Loop**:
     - Passes the noisy input `T` through both Teachers without tracking gradients (`torch.no_grad()`).
     - Averages their outputs to create a combined target: `ensemble_clean_target = (t1_clean + t2_clean) / 2.0`.
     - Passes the 6-channel temporal input (`T`, `T-1`) through the Student.
     - Calculates standard **MSE Loss** between Student and the Ensemble target.
     - Backpropagates and updates Student weights using the Adam optimizer.
     - Saves the best model as `dual_distilled_unet_best.pt`.

### B. `train_unet_from_pretrained.py` (Advanced Single-Teacher Distillation)
This script is used when distilling from a single teacher (often the `blended_model.pt` which is a physical combination of the two teachers). It includes highly advanced training techniques tailored for image restoration.
* **Flow**:
  1. Sets up Train/Validation splits (90/10) using `random_split`.
  2. Loads a single frozen `Restormer` Teacher and a trainable `UNet` Student.
  3. Uses an advanced **PerceptualDistillationLoss** instead of simple MSE. This custom loss combines:
     * **Charbonnier Loss (Pixel-level)**: A smooth L1 loss that handles sharp image edges better than MSE, preventing blurry outputs.
     * **MSE Loss (Baseline)**: Provides a stable baseline gradient for overall structural matching.
     * **SSIM-Proxy (Structural)**: Computes the mean and variance differences between Student and Teacher outputs to implicitly match their structural similarity indices.
  4. Implements **Automatic Mixed Precision (AMP)** via `torch.cuda.amp`. This scales gradients and uses float16 where possible, significantly speeding up training on modern GPUs and reducing VRAM usage.
  5. Uses a **CosineAnnealingLR** learning rate scheduler to decay the learning rate smoothly over epochs.
  6. Implements **Early Stopping** based on validation loss to prevent overfitting.

### C. `train_cityscapes.py` (Teacher Fine-Tuning)
This script takes a base Restormer Teacher and fine-tunes it on a specific dataset (Cityscapes) to handle domain-specific corruptions (like dense urban fog or rain).
* **Flow**:
  1. Loads multiple subsets of the Cityscapes dataset using `ConcatDataset`.
  2. Loads the pre-trained `teacher_best.pt` base Restormer weights.
  3. Uses a very low learning rate (e.g., `1e-5`) because it is fine-tuning, not training from scratch.
  4. Optimizes using only **CharbonnierLoss** to ensure sharp, highly detailed edge restorations of urban environments.
  5. Saves the best fine-tuned weights as `cityscapes_final.pt`.

### D. `blend_models.py` (Weight Averaging)
A utility script to physically merge two trained PyTorch models without running inference.
* **Flow**:
  1. Takes two `.pt` model paths and an `alpha` blending ratio (default `0.5`).
  2. Loads the `state_dict` of both models.
  3. Iterates through the dictionary keys. If a key (layer weight) exists in both, it calculates the new weight as: `(weight1 * alpha) + (weight2 * (1 - alpha))`.
  4. Saves this new aggregated dictionary as `blended_model.pt`.

### E. `run_image_restoration.py` (Single Image Inference)
A standalone script to test the Restormer models on a single image.
* **Flow**:
  1. Reads an image using OpenCV and converts it to a PyTorch Tensor `(1, C, H, W)`.
  2. **Crucial Step**: Because the Restormer uses downsampling (factors of 2), it pads the image dimensions to be a multiple of 4 using `reflect` padding.
  3. Passes the padded tensor through the chosen Restormer model.
  4. Removes the padding.
  5. Converts the tensor back to a standard BGR numpy array and saves it.

### F. `compare_models.py` (Visual Evaluation)
Used to visually benchmark the different Teacher checkpoints.
* **Flow**:
  1. Takes an input image and resizes it to a manageable width (640px).
  2. Loads three models sequentially: Base Teacher, Cityscapes Teacher, and Blended Teacher.
  3. Runs the padded inference for each model.
  4. Concatenates the original image and all three restored outputs horizontally (`cv2.hconcat`) and saves it as a single comparison strip.

### G. `eval_restore_and_detect.py` (End-to-End Pipeline Evaluation)
This script evaluates the real-world impact of the OmniSight restoration by measuring how much it improves object detection.
* **Flow**:
  1. Loads the `OmniSightPipeline` (which encompasses the Student U-Net and YOLOv8 tracker).
  2. **Baseline Mode**: Takes raw corrupted video frames and runs YOLOv8 + ByteTrack directly on them, counting the number of valid tracking boxes.
  3. **OmniSight Mode**: First passes the corrupted frames through the Student U-Net to "clean" them, and *then* runs YOLOv8 + ByteTrack on the cleaned frames.
  4. Compares the total detections/tracks. A successful restoration pipeline will yield a significantly higher number of stable bounding boxes.

---

## 3. The Models

1. **Restormer (Teacher)**: A large Transformer-based network that processes images at high fidelity. It uses Multi-Dconv Head Transposed Attention (MDTA) and Gated-Dconv Feed-Forward Network (GDFN). Very accurate, but very slow.
2. **Temporal U-Net (Student)**: A Convolutional Neural Network with an encoder-decoder structure and skip connections. It accepts 6 channels (two frames) to maintain temporal consistency in video. Highly optimized for speed.

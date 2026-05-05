# OmniSight / Clear Vision: Viva Preparation Guide

This document is designed to help beginners prepare for a project viva or presentation on the **OmniSight (Clear Vision)** project. It breaks down complex concepts into simple, easy-to-understand questions and answers.

## 1. Project Overview

**Q: What is the main goal of the OmniSight project?**
**A:** The main goal is to improve the vision of autonomous systems (like self-driving cars or drones) in bad weather. It does this by taking low-quality, corrupted images (due to rain, fog, or lens distortion) and "cleaning" or "restoring" them before passing them to an object detection model (like YOLOv8).

**Q: Why do we need to restore images before object detection?**
**A:** Object detection models like YOLO are usually trained on clean, clear images. If you feed them a blurry or foggy image, they struggle to find objects (like cars or pedestrians). By restoring the image first, we help the object detector perform much better.

## 2. Core Concepts & Terminology

**Q: What is "Knowledge Distillation"?**
**A:** Knowledge Distillation is a machine learning technique where a large, complex, and slow model (the "Teacher") teaches a smaller, simpler, and faster model (the "Student"). The Student learns to mimic the Teacher's predictions.

**Q: Who are the "Teacher" and "Student" in our project?**
**A:** 
*   **Teacher:** A heavy model called **Restormer**. It is very good at restoring images but is too slow to run in real-time on edge devices (like a Raspberry Pi or a basic camera processor).
*   **Student:** A lightweight model called **U-Net**. It is much faster and smaller, making it perfect for real-time video processing.

**Q: What is a "Temporal U-Net"?**
**A:** A standard image model processes one frame at a time. Our "Temporal U-Net" looks at the *current* frame (T) and the *previous* frame (T-1) together (as 6 channels instead of 3). This helps the model understand movement and keeps the video looking smooth and consistent over time, avoiding flickering.

**Q: What is "Weight Blending"?**
**A:** It is a technique to combine the "knowledge" of two different trained models into one. In our project, we take the weights (the learned numbers) of a general teacher model and a city-specific teacher model, and simply average them together to create a single, well-rounded hybrid model.

## 3. Project Architecture & Workflow

**Q: Can you explain the "Dual-Teacher Distillation" process used in this project?**
**A:** Yes. Instead of learning from one teacher, our Student U-Net learns from an "ensemble" (a group) of two teachers. 
1. We pass a noisy image to Teacher 1 and Teacher 2.
2. We average their restored output to get a very high-quality "pseudo-ground-truth".
3. We pass the same noisy image to our Student U-Net.
4. We train the Student to make its output look exactly like the averaged output from the two Teachers.

**Q: Why don't we just use the Restormer (Teacher) for the final application?**
**A:** The Restormer is computationally very expensive. If a self-driving car needs to process 30 frames per second, the Restormer might be too slow, causing a dangerous delay. The U-Net Student is designed for speed (edge deployment) while still retaining most of the Restormer's accuracy.

## 4. Code & Implementation Details

**Q: Which library/framework is used for building the neural networks in this project?**
**A:** The project is built using **PyTorch**, a popular deep learning framework in Python.

**Q: What does the script `train_dual_teacher.py` do?**
**A:** It is the core training script. It loads the two heavy teacher models, freezes them so they don't change, and then trains the lightweight Student U-Net to mimic their combined outputs.

**Q: What does `blend_models.py` do?**
**A:** It takes two PyTorch model files (like `teacher_best.pt` and `cityscapes_final.pt`), calculates the exact mathematical average of their internal weights, and saves this as a new file (`blended_model.pt`).

**Q: What is the `PerceptualDistillationLoss` and why not just use MSE?**
**A:** Simple Mean Squared Error (MSE) often leads to blurry images because it averages out high-frequency details. `PerceptualDistillationLoss` (used in our advanced U-Net training) combines:
1. **Charbonnier Loss**: A robust L1 loss that keeps edges sharp.
2. **MSE**: For overall structural stability.
3. **SSIM-Proxy**: To match the structural similarity (variance and mean) between the Teacher's perfect output and the Student's output.

**Q: Why do we have a `train_cityscapes.py` script?**
**A:** A base Restormer is good at general noise, but real-world autonomous driving deals with specific environments (like dense urban scenes). `train_cityscapes.py` fine-tunes the base model on the Cityscapes dataset using a low learning rate and Charbonnier Loss to make it a specialist in urban environments.

**Q: What is Automatic Mixed Precision (AMP)?**
**A:** AMP is a technique used during training (specifically in `train_unet_from_pretrained.py`) where PyTorch automatically switches between 32-bit and 16-bit floating-point numbers. This drastically speeds up training on GPUs and reduces memory usage without losing accuracy.

**Q: How do we evaluate if the project is successful in real-time?**
**A:** We use `eval_restore_and_detect.py`. It runs the video through our U-Net, passes the cleaned frames to **YOLOv8** for object detection, and uses **ByteTrack** to track the objects across frames. We compare the total number of stable tracks (using metrics like MOTA) against a baseline where the video was NOT cleaned.

## 5. Typical "Trick" Questions for Viva

**Q: Does Knowledge Distillation always guarantee the Student will be as good as the Teacher?**
**A:** No. A Student model has less "capacity" (fewer parameters) than the Teacher. The goal isn't to be 100% as good, but rather to be 95% as good while being 10x faster. 

**Q: Why use a U-Net architecture specifically for the Student?**
**A:** U-Net has "skip connections" that link the early layers directly to the later layers. This is excellent for image-to-image tasks (like restoration) because it preserves the fine spatial details (like edges and textures) that might otherwise get lost in the network.

# ⚽ Soccer Event Classification via Deep CNNs & Fine-Tuned Transfer Learning

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Hardware Support](https://img.shields.io/badge/Hardware-Dual%20Tesla%20T4%20GPUs-green.svg)](#)

An automated, end-to-end computer vision pipeline that classifies high-resolution soccer match broadcast images into 7 tactical event categories. Designed explicitly for high-throughput execution on multi-GPU deep learning instances (e.g., dual Tesla T4 GPUs) using PyTorch DataParallel distribution.

---

## 📌 Project Overview & Highlights

Manual match indexing and highlight compilation in sports networks are incredibly labor-intensive. This project implements an intelligent, automated frame-level event classifier to detect seven critical association football actions:
* `Corner Kick`, `Free Kick`, `Penalty Kick`
* `Yellow Card`, `Red Card`
* `Substitute`, `Tackle`

### 🛠️ Advanced Sports Engineering Features
* **Defensive Truncation Fail-Safes:** Includes an unreadable/truncated image byte bypass framework to prevent training loop runtime failures (`OSError: image file is truncated`).
* **Aspect-Ratio Square Padding:** Eliminates player body posture and skeletal distortion by symmetrically padding images with black borders before downsampling to 224x224.
* **Adaptive Contrast Optimization (CLAHE):** Isolates the illumination channels using OpenCV LAB transformations to neutralize harsh stadium floodlight glare and deep stadium pitch shadows.
* **Dynamic Engine Augmentation Tracking:** Plots on-the-fly computational variances showing the exact volumetric scaling across epochs.
* **Explainable AI (XAI):** Employs custom Grad-CAM layer hooks to output spatial focus overlays displaying what features (referee hand, player limbs, ball position) the models use to make decisions.

---

## 🏗️ Model Architectures Compared

1. **Custom CNN (Baseline):** A 5-stage sequential convolutional feature extraction network built from scratch, incorporating Batch Normalization, ReLU activations, Max Pooling, and a 50% Dropout dense classifier.
2. **ResNet18 (Transfer Learning V1):** A frozen feature extractor utilization paradigm relying on standard pre-trained ImageNet filters.
3. **EfficientNet_B0 (Transfer Learning V2 - Fine-Tuned):** An advanced setup featuring differential learning rates. Keeps early layers frozen while unfreezing blocks 6 and 7 to tune model attention directly to granular soccer field artifacts.

---

## 🚀 Instructions for Running the Code

1. Navigate to the dataset webpage on Kaggle: Soccer Event Classification Image Data.

2. Click "New Notebook" in the upper-right corner.

3. In your active notebook's right-hand settings panel, change the Accelerator option to "GPU T4 x2" (Dual T4 setup).

4. Copy and paste the script configurations sequentially or import the .ipynb file.

5. Click "Run All". The notebook will download, evaluate, train, validate, and write your figures autonomously.

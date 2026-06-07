# ⚽ Soccer Event Classification via Deep CNNs & Fine-Tuned Transfer Learning An
automated, end-to-end computer vision pipeline that classifies high-resolution soccer match
broadcast images into 7 tactical event categories. Designed explicitly for high-throughput
execution on multi-GPU deep learning instances (e.g., dual Tesla T4 GPUs) using PyTorch
DataParallel distribution. --- ## 📌 Project Overview & Highlights Manual match indexing and
highlight compilation in sports networks are incredibly labor-intensive. This project implements
an intelligent, automated frame-level event classifier to detect seven critical association football
actions: * `Corner Kick`, `Free Kick`, `Penalty Kick` * `Yellow Card`, `Red Card` * `Substitute`,
`Tackle` ### ️ Advanced Sports Engineering Features * **Defensive Truncation Fail-Safes:**
Includes an unreadable/truncated image byte bypass framework to prevent training loop runtime
failures (`OSError: image file is truncated`). * **Aspect-Ratio Square Padding:** Eliminates
player body posture and skeletal distortion by symmetrically padding images with black borders
before downsampling to 224x224. * **Adaptive Contrast Optimization (CLAHE):** Isolates the
illumination channels using OpenCV LAB transformations to neutralize harsh stadium floodlight
glare and deep stadium pitch shadows. * **Dynamic Engine Augmentation Tracking:** Plots
on-the-fly computational variances showing the exact volumetric scaling across epochs. *
**Explainable AI (XAI):** Employs custom Grad-CAM layer hooks to output spatial focus
overlays displaying what features (referee hand, player limbs, ball position) the models use to
make decisions. --- ## ️ Model Architectures Compared 1. **Custom CNN (Baseline):** A
5-stage sequential convolutional feature extraction network built from scratch, incorporating
Batch Normalization, ReLU activations, Max Pooling, and a 50% Dropout dense classifier. 2.
**ResNet18 (Transfer Learning V1):** A frozen feature extractor utilization paradigm relying on
standard pre-trained ImageNet filters. 3. **EfficientNet_B0 (Transfer Learning V2 -
Fine-Tuned):** An advanced setup featuring differential learning rates. Keeps early layers frozen
while unfreezing blocks 6 and 7 to tune model attention directly to granular soccer field artifacts.
--- ## 📂 Repository Structure ```text ├── README.md # Project technical documentation ├──
soccer_event_classification_pipeline.ipynb # Core runnable Kaggle / Jupyter Notebook ├──
requirements.txt # Pinpointed package dependencies └── expected_exports/ # Visual assets
zipped by the export utility ├── preprocessing_evolution_sample_1.png # Step-by-step contrast
and padding transformations ├── data_augmentation_distribution_fixed.png # Data
amplification chart ├── custom_cnn_confusion_matrix.png # Class-by-class error checking
mapping ├── final_model_comparison_matrix.png # Comparative plot across key metrics └──
gradcam_xai_sample_1.png # XAI heatmaps showing attention regions ``` --- ## 🚀 Instructions
for Running the Code ### Option A: Quick-Run on Kaggle (Recommended) 1. Navigate to the
dataset webpage on [Kaggle: Soccer Event Classification Image
Data](https://www.kaggle.com/datasets/rishintiw/soccer-event-classification-image-data-cnn-and
-llm). 2. Click **"New Notebook"** in the upper-right corner. 3. In your active notebook's
right-hand settings panel, change the **Accelerator** option to **"GPU T4 x2"** (Dual T4 setup).
4. Copy and paste the script configurations sequentially or import the `.ipynb` file. 5. Click
**"Run All"**. The notebook will download, evaluate, train, validate, and write your figures
autonomously. ### Option B: Local / Cloud Server Setup #### 1. Prerequisites & Dataset
Procurement Ensure your machine houses a modern NVIDIA GPU setup or an equivalent
compute grid with CUDA configured. Download the dataset files directly from Kaggle and extract
them locally. #### 2. Environment Initialization Clone this repository and establish an isolated
virtual environment to prevent package version conflicts: ```bash # Create and activate
environment python3 -m venv venv source venv/bin/activate # On Windows:
venv\Scripts\activate # Install required dependencies pip install -r requirements.txt ``` #### 3.
Update Path Mapping Configurations Open the pipeline notebook or corresponding script

configurations and adjust the path parameter inside the global `CONFIG` dictionary block to
match your local directory structure: ```python CONFIG = { 'input_shape': (224, 224),
'batch_size': 64, 'epochs': 15, 'lr': 0.001, 'num_classes': 7, 'dataset_path':
'./your_local_path/Soccer Event Classification' } ``` #### 4. Train & Evaluate the Pipeline
Execute the notebook cells in linear sequence. The execution loop will run completely on its
own, providing visual telemetry updates at each stage. Section 14 will zip and create a localized
package download token (`soccer_classification_report_figures.zip`) containing all your plots. ---
## 📊 Expected Performance Matrix | Model Architecture | Test Accuracy | Macro Precision |
Macro Recall | Macro F1-Score | | :--- | :---: | :---: | :---: | :---: | | **Custom CNN Baseline** | ~63%
- 66% | ~0.63 | ~0.64 | ~0.63 | | **ResNet18 Feature Extractor** | ~80% - 83% | ~0.81 | ~0.80 |
~0.80 | | **Fine-Tuned EfficientNet_B0** | **~88% - 91%**| **0.89** | **0.88** | **0.89** |

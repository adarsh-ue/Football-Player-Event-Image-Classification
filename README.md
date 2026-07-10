# ⚽ Football Player Event Image Classification Using Explainable Deep Learning

## 📌 Project Overview

This project develops an automated **football event image classification system** using deep learning and explainable artificial intelligence techniques. The objective is to classify football match images into different event categories using Convolutional Neural Networks (CNNs) and transfer learning architectures.

The study compares a **Custom CNN**, **ResNet18**, and **EfficientNet-B0** to evaluate the effectiveness of transfer learning for sports image understanding. In addition to classification performance, explainability methods including **Grad-CAM** and **SHAP** are applied to analyze the visual regions influencing model decisions.

The final framework provides an interpretable computer vision pipeline suitable for applications such as automated sports analytics, image tagging, and football content organization.

---

# 🎯 Objectives

The main objectives of this project are:

- Develop a CNN-based framework for football event image classification
- Compare custom CNN learning with transfer learning models
- Analyze class-level recognition performance
- Improve model robustness using preprocessing and augmentation
- Apply explainable AI techniques to understand model predictions
- Evaluate accuracy, efficiency, and deployment feasibility

---

# 🏟️ Dataset

The project uses a football event image classification dataset containing seven event categories:

| Class No. | Football Event |
|----------|----------------|
| 1 | Corner Kick |
| 2 | Free Kick |
| 3 | Penalty Kick |
| 4 | Red Card |
| 5 | Yellow Card |
| 6 | Substitute |
| 7 | Tackle |

Images contain different football match situations captured under varying conditions such as:

- Player positions
- Camera viewpoints
- Lighting conditions
- Field regions
- Match scenarios

---

# 🔄 Project Workflow

The complete deep learning workflow consists of:

```text
Football Image Dataset
          |
          ↓
Data Verification
          |
          ↓
Image Preprocessing
(RGB Conversion, CLAHE, Resize, Normalization)
          |
          ↓
Data Augmentation
(Flip, Rotation, Color Adjustment)
          |
          ↓
Train / Validation / Test Split
          |
          ↓
Model Development
(Custom CNN | ResNet18 | EfficientNet-B0)
          |
          ↓
Model Training
          |
          ↓
Performance Evaluation
          |
          ↓
Explainable AI Analysis
(Grad-CAM + SHAP)
          |
          ↓
Football Event Prediction
```

---

# 🧹 Image Preprocessing

The preprocessing pipeline includes:

### RGB Standardization
All images are converted into three-channel RGB format.

### CLAHE Enhancement
Contrast Limited Adaptive Histogram Equalization improves local image contrast.

### Image Resizing
Images are resized to:

```text
224 × 224 × 3
```

to support CNN and transfer learning architectures.

### Normalization
Pixel values are normalized before model input.

---

# 🔁 Data Augmentation

To improve generalization, training images are augmented using:

- Random horizontal flipping
- Random rotation
- Brightness adjustment
- Color transformation

Augmentation increases visual diversity and reduces overfitting.

---

# 🧠 Deep Learning Models

## 1. Custom CNN

A baseline CNN was developed from scratch.

Architecture:

```text
Input Image (224×224×3)

↓
Convolution Block 1
Conv2D + BatchNorm + ReLU + MaxPool

↓
Convolution Block 2
Conv2D + BatchNorm + ReLU + MaxPool

↓
Convolution Block 3
Conv2D + BatchNorm + ReLU + MaxPool

↓
Convolution Block 4
Conv2D + BatchNorm + ReLU + MaxPool

↓
Convolution Block 5
Conv2D + BatchNorm + ReLU + MaxPool

↓
Global Average Pooling

↓
Fully Connected Layer

↓
Softmax Classifier

↓
7 Football Event Classes
```

---

## 2. ResNet18 Transfer Learning

ResNet18 uses pretrained ImageNet weights.

Features:

- Residual connections
- Deep feature extraction
- Transfer learning adaptation
- Modified final classifier layer

---

## 3. EfficientNet-B0 Fine-Tuning

EfficientNet-B0 uses:

- Compound scaling
- Efficient feature extraction
- ImageNet pretrained weights
- Fine-tuning strategy

---

# ⚙️ Experimental Configuration

| Parameter | Value |
|-|-|
| Framework | PyTorch |
| Image Size | 224 × 224 |
| Classes | 7 |
| Epochs | 50 |
| Learning Rate | 0.0001 |
| Batch Size | 20 |
| Optimizer | SGD |
| Momentum | 0.9 |
| Weight Decay | 0.0002 |
| Loss Function | Cross Entropy Loss |

---

# 📊 Evaluation Metrics

Models were evaluated using:

## Accuracy

Measures the percentage of correctly classified football images.

## Precision

Measures reliability of predictions for each football event.

## Recall

Measures ability to identify all images belonging to each event.

## F1-Score

Balances precision and recall.

Additional evaluation:

- Confusion Matrix
- ROC-AUC Score
- Training curves
- Validation curves

---

# 📈 Model Performance

| Model | Approach |
|---|---|
| Custom CNN | Training from scratch |
| ResNet18 | Transfer Learning |
| EfficientNet-B0 | Fine-Tuning |

The models were compared based on:

- Classification performance
- Training stability
- Parameter efficiency
- Inference capability

---

# 🔥 Explainable AI (XAI)

Explainability techniques were applied to understand CNN decisions.

---

## Grad-CAM

Gradient-weighted Class Activation Mapping highlights important image regions.

Used to identify:

- Players
- Referees
- Football regions
- Event-specific objects

Example outputs:

```text
Input Image
     ↓
CNN Feature Maps
     ↓
Gradient Analysis
     ↓
Heatmap Visualization
```

---

## SHAP

SHAP explains predictions using feature contribution values.

Provides:

- Positive evidence regions
- Negative evidence regions
- Class contribution analysis
- Prediction transparency

---

# 🚀 Deployment

A lightweight deployment pipeline was created for football image prediction.

Deployment considerations:

- Model inference
- Image upload
- Prediction confidence
- Explainability output

---

# 🛠️ Technologies Used

- Python
- PyTorch
- TorchVision
- NumPy
- Pandas
- Matplotlib
- Scikit-Learn
- OpenCV
- Grad-CAM
- SHAP

---

# 📂 Repository Structure

```text
Football-Event-Classification/

│
├── Dataset/
│   └── Dataset information
│
├── Models/
│   ├── CustomCNN
│   ├── ResNet18
│   └── EfficientNetB0
│
├── Notebooks/
│   └── Training notebooks
│
├── Results/
│   ├── Accuracy curves
│   ├── Confusion matrices
│   └── Evaluation results
│
├── Explainability/
│   ├── GradCAM outputs
│   └── SHAP outputs
│
├── requirements.txt
│
└── README.md
```

---

# 📌 Key Contributions

✔ Football event classification using CNN architectures  
✔ Transfer learning comparison using ResNet18 and EfficientNet-B0  
✔ Complete preprocessing and augmentation workflow  
✔ Class-level performance analysis  
✔ Grad-CAM and SHAP explainability integration  
✔ Deployment-ready image classification pipeline  

---

# 🔮 Future Improvements

Possible extensions include:

- Larger multi-league football datasets
- Video-based event recognition
- Vision Transformer models
- Real-time match analysis
- Object detection integration
- Mobile deployment optimization

---

# 📜 License

This project is developed for academic and research purposes.

---

# 👤 Author

**Football Player Event Image Classification Project**

Deep Learning | Computer Vision | Sports Analytics | Explainable AI

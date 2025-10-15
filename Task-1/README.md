# 🤖 Kung Fu Scroll Classifier

### 📘 Overview  
This project aims to develop a **machine learning classifier** capable of identifying **correct (Real)** and **incorrect (Fake)** symbols for both **R1 (Red)** and **R2 (Blue)** categories.  
The reference patterns are based on the *Robocon 2026 – Kung Fu Scroll Recognition Patterns*.

---

### 🎯 Objective  
Create a vision-based model that can:
- Classify whether a symbol is **Real** or **Fake**.  
- Identify whether it belongs to **R1 (Red)** or **R2 (Blue)**.  

---

### 🧩 Task Breakdown  

#### 1. Dataset Preparation  
- Extract and label each symbol from the provided reference image.  
- Ensure proper naming and structure
- Preprocess images (crop, resize, normalize).  
- Optionally apply data augmentation for better model generalization.

#### 2. Model Development  
- Use a **Convolutional Neural Network (CNN)** or any efficient image classification architecture.  
- Frameworks allowed: **TensorFlow**, **Keras**, **PyTorch**, or **OpenCV + ML** approach.  
- Train the model to distinguish all symbol classes with high accuracy.

#### 3. Evaluation  
- Evaluate model performance using:  
- Accuracy, Precision, Recall, F1-Score  
- Confusion Matrix visualization  
- Test on unseen data to verify robustness.  

#### 4. Documentation  
- Include a short report (`report.md` or `report.txt`) describing:  
- Dataset and preprocessing  
- Model architecture and training parameters  
- Results and insights  
- Add comments and clear variable names in all scripts.  

---

### 📦 Deliverables  
- ✅ Trained model file (`model.h5` or `model.pt`)  
- ✅ Training and testing code (`train.py`, `test.py`, or `notebook.ipynb`)  
- ✅ Final report with metrics and visualizations  

---

### 🗓️ Deadline  
`19 October 2025`

---

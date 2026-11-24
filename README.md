# 🐾 Cat vs Dog Image Classifier

A high-performance deep learning image classification project implemented in **PyTorch**, accelerated with Apple Silicon GPU (`mps`).

---

## 📊 Model Comparison & Results

| Model Architecture | Approach | Epochs | Training Time | Validation Accuracy | Kitten (`Unknown.webp`) Prediction |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **MobileNetV3** *(Active Default)* | Transfer Learning (Pre-trained on ImageNet) | 5 | ~1.35 min | **96.70%** | 🐱 **Cat (99.87%)** ✅ |
| **Custom 4-Block CNN** | Trained from Scratch | 15 | ~9.6 min | **81.30%** | 🐶 Dog (80.39%) ❌ |

- The fine-tuned model checkpoint is saved at [best_model.pth](best_model.pth).
- The original from-scratch checkpoint is preserved at `best_model_scratch.pth`.

---

## 📁 Project Structure

```text
image_classifier/
├── model.py                # Dual model architectures (MobileNetV3 & Custom CNN)
├── download_data.py        # Automated dataset download and extraction script
├── train_transfer.py       # Transfer learning fine-tuning script (MobileNetV3)
├── train.py                # Scratch CNN training script
├── predict.py              # CLI prediction script (auto-detects files & directories)
├── app.py                  # Interactive drag-and-drop Streamlit Web UI
├── best_model.pth          # Active best model weights (MobileNetV3)
├── best_model_scratch.pth  # Original scratch model weights
├── test_data/              # Folder for testing your custom images
├── data/                   # Cats & Dogs dataset (train and validation splits)
└── venv/                   # Python virtual environment
```

---

## 🚀 How to Use

### 1. Run Predictions from Terminal

Test an entire directory or a single image file:

```bash
# Predict all images in test_data folder:
python3 predict.py --image test_data

# Or predict a single image:
python3 predict.py --image test_data/Unknown.webp
```

Example Output:
```text
=============================================
File:       Unknown.webp
Prediction: 🐱 Cat (99.87% confidence)
Probabilities:
  • 🐱 Cat: 99.87%
  • 🐶 Dog: 0.13%
=============================================
```

### 2. Launch the Interactive Web App (Streamlit)

```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser. Drag and drop any image to see real-time classifications with confidence scores and progress bars.

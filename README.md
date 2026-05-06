# 🔍 DeepFake Detection System

A deep learning-based web application to detect whether an image is **real or AI-generated (deepfake)**, built using **EfficientNetB7 with a custom Attention Mechanism** and deployed via **Streamlit**.

---

## 🚀 Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)

---

## 📌 About

With the rise of generative AI tools like Stable Diffusion, DALL-E, and GANs, detecting synthetic faces has become critical. This project tackles that problem using a powerful CNN backbone enhanced with spatial attention — making the model focus on the most suspicious regions of a face.

---

## ✨ Features

- 🧠 **EfficientNetB7 + Attention Mechanism** for high-accuracy detection
- 📊 **Confidence gauge charts** — Real vs Fake probability
- 🔥 **Attention heatmap** — shows exactly where the model looked
- 🔄 **Preprocessing pipeline visualization** — step-by-step image transformation
- 🧠 **Model pipeline diagram** — layer-by-layer inference flow
- ⚙️ **3 preprocessing modes** — Training Match, Simple Norm, EfficientNet ImageNet
- 🐛 **Debug mode** — tensor shapes, value ranges, raw probabilities

---

## 🗂️ Project Structure

```
deepfake-detection-streamlit/
├── src/
│   ├── app.py                  # Main Streamlit application
│   ├── model/
│   │   ├── architecture.py     # EfficientNetB7 + Attention model
│   │   └── loader.py           # Model loading from Hugging Face
│   ├── preprocessing/
│   │   └── image_processor.py  # 3 preprocessing pipelines
│   └── ui/
│       ├── components.py       # Header, sidebar, footer
│       └── styles.py           # Custom CSS
├── data/
│   ├── Train/Real/
│   ├── Train/Fake/
│   ├── Test/Real/
│   └── Test/Fake/
├── docs/
├── requirements.txt
└── README.md
```

---

## 🧠 Model Architecture

```
Input (128×128×3)
      ↓
EfficientNetB7 Backbone   ← Pre-trained, extracts deep features
      ↓
Batch Normalization
      ↓
Spatial Attention Block   ← Focuses on suspicious face regions
      ↓
Dropout (0.5) + Dense 64
      ↓
Softmax Output            → P(Real), P(Fake)
```

The **attention block** uses a series of 1×1 convolutions to generate a spatial mask, highlighting regions most relevant for classification — typically face boundaries, eyes, and skin texture in deepfakes.

---

## 📦 Dataset

Dataset used: [Deepfake and Real Images — Kaggle](https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images)

- ~190,000 images total
- Balanced 50/50 split: Real faces and AI-generated fake faces
- Sources: GAN-generated, Stable Diffusion, face-swap methods
- Split into Train / Test / Validation sets

---

## 🔥 Visualizations

| Tab               | What it shows                                      |
| ----------------- | -------------------------------------------------- |
| 📊 Confidence     | Gauge charts + bar chart of Real vs Fake %         |
| 🔥 Attention Map  | Heatmap overlay — red = high attention, blue = low |
| 🔄 Preprocessing  | 4-step image transformation pipeline               |
| 🧠 Model Pipeline | Layer-by-layer inference diagram                   |

**Heatmap color guide:**

- 🔴 **Red/Yellow** — Model focused heavily here (suspicious region)
- 🔵 **Blue/Green** — Background, less relevant to decision

---

## ⚙️ Installation

```bash
# Clone the repo
git clone https://github.com/gazalsg/deepfake-detection-streamlit.git
cd deepfake-detection-streamlit

# Create virtual environment
conda create -n deepfake python=3.11 -y
conda activate deepfake

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run src/app.py
```

---

## 📋 Requirements

```
streamlit>=1.32.0
tensorflow>=2.13.0
pillow>=10.0.0
opencv-python-headless>=4.8.0
numpy>=1.24.0
huggingface-hub>=0.20.0
matplotlib>=3.8.0
```

---

## 🌐 Deployment

- **App:** Streamlit Community Cloud
- **Model weights:** Hugging Face Model Hub (`~240MB`, auto-downloaded at runtime)

---

## 🤔 Why EfficientNetB7?

| Model              | Accuracy | Parameters |
| ------------------ | -------- | ---------- |
| VGG16              | ~92%     | 138M       |
| ResNet50           | ~93%     | 25M        |
| InceptionV3        | ~93%     | 23M        |
| **EfficientNetB7** | **~97%** | **66M**    |

EfficientNetB7 uses **compound scaling** — balancing depth, width, and resolution — designed via Neural Architecture Search. It achieves the best accuracy for detecting subtle facial manipulation artifacts.

---

## 👩‍💻 Author

**Gazal Keshwani**
B.E. Computer Engineering
Vivekanand Education Society's Institute Of Technology, Mumbai

---

## 📄 License

MIT License — © 2026 Gazal Keshwani

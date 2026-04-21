# 🌊 DBR-Net: Dual-Branch Refinement Network for Underwater Image Enhancement

**Author:** Abhyuday Singh Rathore  
**Co-authors:** Anurag Bindra, Nupur Nandanwar, Harsh Shinde, Siyona Chandanshiv, Hrishika Jawasa  
**University:** Ajeenkya DY Patil University, School of Engineering, Pune  
**Conference:** ICSICE-2026 | Paper ID: ICSICE-475  

---

## 📌 What is this project?

Underwater images suffer from severe color distortion, 
low contrast, and blurring caused by light absorption 
and scattering in water. This makes it hard for both 
humans and AI systems to understand underwater scenes.

**DBR-Net** is a lightweight deep learning model that 
automatically enhances degraded underwater images in 
real-time. It uses two parallel processing branches:

- **Detail Branch** — recovers edges and textures 
  using standard convolutions
- **Context Branch** — fixes global color distortion 
  using dilated convolutions

Both branches are combined and added back to the 
original image (residual learning) to produce a 
clean, enhanced output.

---

## 📊 Results

Evaluated on the **EUVP dataset**:

| Method | PSNR (dB) | SSIM |
|--------|-----------|------|
| Raw Input | 14.23 | 0.62 |
| CLAHE (traditional) | 16.45 | 0.69 |
| **DBR-Net (ours)** | **18.58** | **0.78** |

Higher is better for both metrics. DBR-Net 
significantly outperforms traditional methods.

---

## 🗂️ Project Structure

DBR-Net/
├── app.py                  # Streamlit demo app
├── models/
│   └── dbr_net.py         # DBR-Net architecture
├── training/
│   └── train.py           # Training script
├── datasets/              # Dataset loading
├── losses/                # Loss functions
├── evaluate/              # Evaluation metrics
├── inference/             # Run model on images
├── results/
│   └── checkpoints/       # Saved model weights
└── data/                  # Training data

---

## 🚀 How to Run (Step by Step)

### Step 1 — Clone the repository
```bash
git clone https://github.com/abhyudaysr/DBR-Net.git
cd DBR-Net
```

### Step 2 — Create virtual environment
```bash
python -m venv venv
```

### Step 3 — Activate it
```bash
# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### Step 4 — Install dependencies
```bash
pip install streamlit torch torchvision streamlit-image-comparison torchmetrics Pillow numpy
```

### Step 5 — Run the demo
```bash
streamlit run app.py
```

### Step 6 — Open browser
Go to `http://localhost:8501` — upload any 
underwater image and see the enhancement live!

---

## 🧠 Model Architecture

Input Image (3×H×W)
↓
Shallow Feature Extraction (Conv 3×3, 32ch)
↓
┌──┴──┐
↓     ↓
Detail  Context
Branch  Branch
(std    (dilated
conv)   conv d=2)
↓     ↓
└──┬──┘
↓
Concatenate (64ch)
↓
Fusion (1×1 Conv)
↓
Final Conv (3ch)
↓

Input (Residual)
↓
Enhanced Image

---

## ⚙️ Training Details

- **Dataset:** EUVP (Enhancing Underwater Visual Perception)
- **Loss:** 0.85 × L1 + 0.15 × (1 − SSIM)
- **Optimizer:** Adam (lr=1e-4, β1=0.9, β2=0.999)
- **Epochs:** 100
- **Batch size:** 8

---

## 📚 References

1. C. Li et al., "Underwater Image Enhancement 
   Benchmark", IEEE TIP, 2019
2. M. J. Islam et al., "FUnIE-GAN", IEEE RA-L, 2020
3. D. Akkaynak & T. Treibitz, "Revised Underwater 
   Imaging Model", CVPR, 2018
4. K. Panetta et al., "UIQM Metric", IEEE JOE, 2016


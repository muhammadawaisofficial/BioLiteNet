<div align="center">

# 🏥 BioLiteNet

### Lightweight Biomedical Waste Classification via MobileNetV3 + CBAM + Hazard-Aware Focal Loss

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green)
![Accuracy](https://img.shields.io/badge/Test%20Accuracy-96.97%25-brightgreen)
![Macro F1](https://img.shields.io/badge/Macro--F1-0.9680-brightgreen)
![Sharps F1](https://img.shields.io/badge/Sharps--F1-0.9864-brightgreen)

</div>

---

## 📌 Abstract

**BioLiteNet** is a lightweight, safety-aware deep learning model for classifying biomedical waste into **23 categories** based on the WHO biomedical waste taxonomy. It combines:

- **MobileNetV3-Large** as the efficient backbone
- **CBAM** (Convolutional Block Attention Module) inserted at stages 3, 6, and 12 for fine-grained spatial and channel attention
- **HAFL** (Hazard-Aware Focal Loss), a novel loss function that penalizes misclassification of high-risk waste categories (e.g., sharps, pathological) more heavily than general waste

BioLiteNet achieves **96.97% accuracy** and a **Sharps-F1 of 0.9864** on the background-removed split of the Biomedical Waste Dataset — making it suitable for real-time deployment in resource-constrained hospital settings.

---

## 🏆 State-of-the-Art Comparison (2025-2026)
BioLiteNet mathematically out-performs recent architectures:
- **Vs. Khonjun et al. (2025):** Matches their 98.4% precision but with a fraction of the computational bloat of their heavy reinforcement learning ensemble.
- **Vs. Swin Transformers (2026):** Bypasses the immense inference bottlenecks of modern transformer models, enabling ultra-fast 18ms processing.
- **Vs. Reddy et al. CNNs (2026):** Dynamically accounts for extreme class imbalance through HAFL, correctly prioritizing dangerous sharp objects.

---

## 🏗️ Architecture

```
Input Image (224×224×3)
        │
        ▼
┌─────────────────────────────────────────┐
│         MobileNetV3-Large Backbone      │
│                                         │
│  Stage 0  ──▶  Stage 1  ──▶  Stage 2   │
│                                         │
│  Stage 3  ──▶ [CBAM] ──▶  ...          │  ← Channel + Spatial Attention
│                                         │
│  Stage 6  ──▶ [CBAM] ──▶  ...          │  ← Channel + Spatial Attention
│                                         │
│  Stage 12 ──▶ [CBAM] ──▶  Stage 13-16  │  ← Channel + Spatial Attention
└─────────────────────────────────────────┘
        │
        ▼
  AdaptiveAvgPool2d(1)
        │
        ▼
   Dropout(0.4)
        │
        ▼
   Linear(960 → 23)
        │
        ▼
  HAFL Loss: L = -h_c · α_t · (1 - p_t)^γ · log(p_t)
             where h_c = WHO hazard weight per class
```

### CBAM Detail
```
Feature Map (B, C, H, W)
        │
        ▼
  ChannelGate  →  scale by important channels
        │
        ▼
  SpatialGate  →  scale by important spatial regions
        │
        ▼
 Refined Feature Map
```

---

## 📊 Results

### Ablation Study

| Variant        | CBAM | HAFL | Accuracy | Macro-F1 | Sharps-F1 |
|----------------|:----:|:----:|:--------:|:--------:|:---------:|
| A — Baseline   |  ✗   |  ✗   |    —     |    —     |     —     |
| B — CBAM only  |  ✓   |  ✗   |    —     |    —     |     —     |
| C — HAFL only  |  ✗   |  ✓   |    —     |    —     |     —     |
| **D — BioLiteNet** | **✓** | **✓** | **96.97%** | **0.9680** | **0.9864** |

> Run `python main.py --all-variants` to reproduce the full ablation study.

### Per-Class F1 Highlights (Variant D)

| Class | F1 | Risk Category |
|-------|----|---------------|
| stitch_removal_scissors | 1.0000 | 🔴 Sharps |
| used_masks | 1.0000 | 🟡 General |
| uncategorized_or_overlapping | 0.9978 | 🟡 General |
| hemostats | 0.9948 | 🔴 Sharps |
| episiotomy_scissors | 0.9948 | 🔴 Sharps |
| scalpels | 0.9948 | 🔴 Sharps |
| used_syringes | 0.9897 | 🔴 Sharps |
| forceps | 0.9843 | 🔴 Sharps |
| ... | ... | ... |
| human_organs | 0.9026 | 🟠 Pathological |
| waterbottles | 0.9149 | 🟡 General |

---

## 📦 Dataset

This project uses the **Biomedical Waste Dataset** available on Kaggle:

🔗 **[Download from Kaggle](https://www.kaggle.com/datasets/)**

After downloading:
1. Place `archive.zip` inside the `data/` folder:
   ```
   BioLiteNet/
   └── data/
       └── archive.zip   ← place here
   ```
2. The training script will auto-extract and split the dataset (70/15/15).

**Dataset statistics (per class, 22 classes × 640 images + uncategorized × 1520):**

| Split | Images |
|-------|--------|
| Train | ~10,360 |
| Val   | ~2,220  |
| Test  | ~2,340  |

---

## 🚀 Quickstart

### 1. Clone the repo
```bash
git clone https://github.com/muhammadawaisofficial/BioLiteNet.git
cd BioLiteNet
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Place the dataset
```
data/archive.zip
```

### 4. Train BioLiteNet (Variant D — full model)
```bash
python main.py --variant D
```

### 5. Or open the notebook
```bash
jupyter notebook BioLiteNet.ipynb
```
Run all cells — training auto-resumes from checkpoint if one exists.

---

## ⚙️ Configuration

All hyperparameters are in [`config.py`](config.py):

| Parameter | Value | Description |
|-----------|-------|-------------|
| `IMG_SIZE` | 224 | Input resolution |
| `BATCH_SIZE` | 16 | Training batch size |
| `NUM_EPOCHS` | 30 | Total training epochs |
| `LR` | 1e-3 | Initial learning rate (AdamW) |
| `LR_MIN` | 1e-6 | Cosine annealing minimum LR |
| `DROPOUT_RATE` | 0.4 | Classifier dropout |
| `FOCAL_GAMMA` | 2.0 | Focal loss focusing parameter |
| `CBAM_STAGES` | [3, 6, 12] | MobileNetV3 stages where CBAM is inserted |
| `SEED` | 42 | Global random seed |

---

## 📁 Project Structure

```
BioLiteNet/
│
├── BioLiteNet.ipynb        # Self-contained notebook (recommended entry point)
│
├── config.py               # All hyperparameters and paths
├── model.py                # BioLiteNet, CBAM, ChannelGate, SpatialGate
├── loss.py                 # HazardAwareFocalLoss (HAFL) + FocalLoss
├── dataset.py              # Dataset extraction, splitting, dataloaders
├── trainer.py              # Training loop, evaluation, checkpointing
├── main.py                 # CLI entry point (argparse)
├── results_table.py        # Ablation study table generator
│
├── data/                   # Dataset (not tracked by git — see .gitignore)
│   └── archive.zip         ← place Kaggle download here
│
├── outputs/                # Training outputs (results tracked, checkpoints not)
│   ├── split_stats.json
│   └── D_BioLiteNet/
│       ├── training_log.csv
│       ├── test_report.txt
│       ├── per_class_f1.json
│       └── confusion_matrix.npy
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🧠 CLI Usage

```bash
# Train the full BioLiteNet model (Variant D)
python main.py --variant D

# Train all 4 ablation variants
python main.py --all-variants

# Skip dataset extraction (if already extracted)
python main.py --variant D --skip-extract

# Train a specific variant
python main.py --variant A   # Baseline (no CBAM, no HAFL)
python main.py --variant B   # CBAM only
python main.py --variant C   # HAFL only
python main.py --variant D   # BioLiteNet (CBAM + HAFL)
```

**Training auto-resumes** from the last saved checkpoint — so interrupting and restarting is safe.

---

## 🔬 HAFL — Hazard-Aware Focal Loss

Standard focal loss treats all misclassifications equally. HAFL introduces a per-class **hazard weight** $h_c$ based on WHO biomedical waste risk guidelines:

$$\mathcal{L}_{\text{HAFL}} = -h_c \cdot \alpha_t \cdot (1 - p_t)^\gamma \cdot \log(p_t)$$

| Risk Category | Classes | $h_c$ |
|---------------|---------|--------|
| 🔴 Sharps (highest) | used_syringes, scalpels, scissors, forceps, hemostats | 2.8 – 3.0 |
| 🟠 Pathological | human_organs, blood_soaked_bandages | 2.5 |
| 🟡 Chemical | disinfectant_bottles, ampoules | 2.0 – 2.2 |
| 🟢 Pharmaceutical | expired_tablets, vials, syrup_bottles | 1.5 |
| ⚪ General | masks, gloves, paper, organic waste | 1.0 – 1.3 |

This ensures the model is more heavily penalized for missing high-hazard classes like sharps (direct injury + bloodborne pathogen risk).

---

## 📋 Requirements

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
scikit-learn>=1.3.0
Pillow>=10.0.0
```

> **Windows note:** Set `NUM_WORKERS = 0` in `config.py` (already the default) to avoid multiprocessing issues with PyTorch DataLoader on Windows.

---

## 📄 Citation

If you use BioLiteNet in your research, please cite:

```bibtex
@article{biolite2026,
  title     = {BioLiteNet: Lightweight Biomedical Waste Classification with
               Attention and Hazard-Aware Focal Loss},
  author    = {Your Name and Co-authors},
  journal   = {Journal / Conference Name},
  year      = {2026},
  url       = {https://github.com/muhammadawaisofficial/BioLiteNet}
}
```

---

## 📜 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

<div align="center">
Made for safer biomedical waste management 🏥
</div>

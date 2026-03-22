# Diabetic Retinopathy Detection
### APTOS 2019 Blindness Detection — Kaggle Competition

A deep learning pipeline for automated detection and grading of diabetic retinopathy severity from retinal fundus images, built with PyTorch and EfficientNet-B4.

---

## Overview

Diabetic retinopathy (DR) is the leading cause of blindness among working-age adults. Early detection is critical — treatment is most effective before vision loss occurs. This project automates DR grading from retinal fundus photographs, enabling scalable screening in rural and resource-limited settings.

The model predicts DR severity on a 0–4 ordinal scale:

| Grade | Label | Description |
|-------|-------|-------------|
| 0 | No DR | Healthy retina |
| 1 | Mild | Microaneurysms only |
| 2 | Moderate | More than just microaneurysms |
| 3 | Severe | Extensive hemorrhages, no proliferative signs |
| 4 | Proliferative DR | Neovascularization present |

**Final Result: QWK 0.9159** on validation set (single 80/20 split, EfficientNet-B4).

---

## Dataset

[APTOS 2019 Blindness Detection](https://www.kaggle.com/competitions/aptos2019-blindness-detection)

- 3,662 training images with expert-graded labels
- 1,928 test images
- Images captured across rural clinics in India using varying equipment
- Significant class imbalance (class 0: 49.3%, class 3: 5.3%)
- Distribution shift between train and test sets due to different camera equipment

---

## Project Structure

```
aptos2019-blindness-detection/
│
├── notebook.ipynb              # Main training notebook
└── README.md
```

---

## Pipeline

### 1. Exploratory Data Analysis
- Class distribution analysis — reveals 9.3× imbalance between majority and minority classes
- Raw pixel statistics across train and test sets — reveals distribution shift due to different cameras
- Image size distribution — train images up to 4288×2848, test images mostly 640×480

### 2. Preprocessing

Every image goes through a three-step pipeline before any augmentation:

```
load image (PIL RGB)
    → crop_black_borders()     remove empty black rows/cols
    → apply_circular_mask()    zero out non-retinal area, set border to 128
    → ben_graham()             normalize lighting, enhance local structure
    → PIL Image                ready for transforms
```

**Ben Graham normalization** is the most important step. The formula `4 × original − 4 × blurred + 128` removes global brightness and color cast (fixing the camera distribution shift) while preserving local structure — the blood vessels, hemorrhages and exudates that carry the diagnostic signal.

The circular mask sets non-retinal pixels to `128` (not `0`) to match the Ben Graham baseline and avoid edge artifacts during the Gaussian blur.

### 3. Class Imbalance Handling

Two complementary tools working at different levels:

**WeightedRandomSampler** — fixes frequency. Each image gets a weight of `1 / class_count`, so minority classes are drawn proportionally more often. Result: all classes appear roughly equally per epoch despite the original 9.3× imbalance.

**Per-class augmentation** — fixes repetition. Because the sampler repeats minority images 3–4× per epoch, augmentation ensures each repeat looks different:

| Class | Level | Transforms |
|-------|-------|------------|
| 0 — No DR | Light | Horizontal flip, subtle color jitter |
| 1, 2 — Mild/Moderate | Medium | + Vertical flip, rotation ±20° |
| 3, 4 — Severe/Proliferative | Heavy | + Rotation ±45°, Gaussian blur |

All augmentations are clinically validated — only transformations physically possible from real camera/operator variation. No elastic distortion, no random erasing, no hue/saturation shifts (which would corrupt the diagnostic color signal of hemorrhages and exudates).

### 4. Model Architecture

```
Input [batch, 3, 384, 384]
    ↓
EfficientNet-B4 backbone        (17.5M params — pretrained ImageNet)
    ↓
Global average pooling          [batch, 1792]
    ↓
BatchNorm1d(1792)
    ↓
Linear(1792 → 256)
    ↓
ReLU + Dropout(0.4)
    ↓
Linear(256 → 1)                 regression output
    ↓
squeeze → round → clamp(0, 4)   final DR grade prediction
```

Regression is used instead of classification because DR severity is ordinal — predicting class 4 for a class 0 patient should be penalized much more than predicting class 1. Regression naturally captures this ordering; classification treats all mistakes equally.

### 5. Training Strategy

**Progressive unfreezing:**
- Epochs 0–2: backbone frozen, only head trains (462K params). Model learns task-specific features fast.
- Epoch 3+: backbone unfrozen, full fine-tuning (18M params). Pretrained features adapt to retinal fundus images.

**Loss:** `SmoothL1` with label smoothing (`smoothing=0.1`) — softens targets slightly toward the batch mean to handle the known label noise in this dataset.

**Optimizer:** `AdamW` with separate learning rates — backbone `lr=1e-4`, head `lr=3e-4`, `weight_decay=1e-2`.

**Scheduler:** `CosineAnnealingLR` — smoothly decays from peak to `eta_min=1e-6` over 15 epochs.

**Additional regularization:**
- `Dropout(0.4)` in regression head
- Gradient clipping (`max_norm=1.0`)
- Mixed precision training (`GradScaler` + `autocast`) — faster training, less GPU memory

**Early stopping:** patience=5, min_delta=0.001 — monitors validation QWK.

### 6. Evaluation

**Primary metric:** Quadratic Weighted Kappa (QWK) — the official Kaggle competition metric. Penalizes predictions proportionally to the square of the error distance, making large mistakes much more costly than small ones.

```
Off by 1 → penalty = 1
Off by 2 → penalty = 4   (4× worse)
Off by 3 → penalty = 9   (9× worse)
```

---

## Results

| Metric | Score |
|--------|-------|
| Val QWK | 0.9159 |
| Val Accuracy | 0.7858 |
| Val F1 (weighted) | 0.7913 |

**Per-class performance:**

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| 0 — No DR | 0.99 | 0.94 | 0.96 |
| 1 — Mild | 0.41 | 0.55 | 0.47 |
| 2 — Moderate | 0.77 | 0.77 | 0.77 |
| 3 — Severe | 0.36 | 0.72 | 0.48 |
| 4 — Proliferative | 0.84 | 0.27 | 0.41 |

Classes 1, 3, 4 are minority classes with known visual ambiguity — even expert ophthalmologists disagree on borderline cases in this dataset.

---

## Requirements

```
torch >= 2.0
torchvision
timm
opencv-python
Pillow
pandas
numpy
scikit-learn
matplotlib
seaborn
tqdm
```

---

## Reproducing Results

1. Download the dataset from Kaggle and place it at `/kaggle/input/competitions/aptos2019-blindness-detection/`
2. Run all cells in `notebook.ipynb` in order
3. `best_model.pth` and `submission.csv` will be generated automatically

**Hardware:** Tesla T4 GPU (16GB), ~3.5 hours training time for 15 epochs.

---

## Further Development

The following improvements are recommended to push performance toward and beyond the top Kaggle submissions (QWK ~0.93–0.94):

### High Impact

**5-Fold Cross Validation + Ensemble**
The most reliable way to improve score. Train one model per fold and average all 5 models' raw predictions before rounding. Each model sees a different 80% of data — averaging cancels individual model errors. Expected gain: +0.005–0.015 QWK.

**Larger Backbone**
EfficientNet-B5 or B6 instead of B4. Larger models extract richer features at the cost of more memory and training time. Expected gain: +0.005–0.010 QWK.

### Medium Impact

**Pre-save Preprocessed Images to Disk**
Currently preprocessing runs on CPU for every image every batch, creating a data loading bottleneck that keeps the GPU underutilized. Saving preprocessed images to disk once before training would significantly speed up epoch time and allow `num_workers` to be pushed higher.

**Cosine Annealing with Warm Restarts (SGDR)**
Replace the current single-cycle cosine scheduler with warm restarts (`CosineAnnealingWarmRestarts`). The periodic LR spikes help escape local minima and can find better solutions than a single monotonic decay.

---

## References

- [APTOS 2019 Blindness Detection Competition](https://www.kaggle.com/competitions/aptos2019-blindness-detection)
- [EfficientNet: Rethinking Model Scaling for CNNs](https://arxiv.org/abs/1905.11946)
- [Ben Graham's 2015 Diabetic Retinopathy Solution](https://www.kaggle.com/c/diabetic-retinopathy-detection/discussion/15801)
- [Quadratic Weighted Kappa](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html)
- [timm — PyTorch Image Models](https://github.com/huggingface/pytorch-image-models)

---

## License

This project is for educational and research purposes. The dataset is subject to the [APTOS 2019 competition rules](https://www.kaggle.com/competitions/aptos2019-blindness-detection/rules).
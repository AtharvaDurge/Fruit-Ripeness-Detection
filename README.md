# 🍎🍌🍊 Fruit Ripeness Detection System

A fruit ripeness classifier built entirely with **classical image processing** — no neural networks, no deep learning. Given a photo of an **apple, banana or orange**, the system decides **which fruit it is** and **whether it is ripe or unripe**, using handcrafted colour, texture and shape features and a two-stage reference-vector classifier. It ships with an interactive Streamlit dashboard for prediction, exploratory data analysis (EDA) and evaluation.

---

## Table of Contents

1. [Highlights](#highlights)
2. [How It Works](#how-it-works)
3. [Feature Set](#feature-set)
4. [Classification Method](#classification-method)
5. [Project Structure](#project-structure)
6. [Dataset Layout](#dataset-layout)
7. [Installation](#installation)
8. [Usage](#usage)
9. [Using a Different Dataset](#using-a-different-dataset)
10. [The Dashboard](#the-dashboard)
11. [Evaluation](#evaluation)
12. [Limitations](#limitations)
13. [Tech Stack](#tech-stack)
14. [Future Work](#future-work)

---

## Highlights

- **Strictly traditional pipeline** — HSV/Otsu segmentation, GLCM/LBP texture, shape moments. Fully explainable, every decision traces back to a distance between feature vectors.
- **Two-stage classification** — Stage 1 identifies the fruit, Stage 2 decides ripe vs. unripe using only that fruit's references.
- **108 handcrafted features** across five groups (fruit colour, fruit texture, shape, whole-image colour, whole-image texture).
- **Robust to imperfect masks** — whole-image features act as a safety net when fruit and background (e.g. green fruit among leaves) cannot be separated by thresholding.
- **Automatic feature-group selection** per stage using cross-validation on the training data only.
- **Honest evaluation** on unseen images (separate test folder, or a stratified hold-out split).
- **Interactive dashboard** with prediction, EDA, performance metrics and reference vectors.
- **Dataset-agnostic** — switch to another dataset directly from the sidebar, no code changes.

---

## How It Works

```
Photo
  │
  ├─ 1. Preprocessing       resize (256×256) → Gaussian blur → contrast stretching
  │
  ├─ 2. Segmentation        RGB → HSV → S channel → Otsu threshold → binary mask
  │                         → morphological cleaning → largest blob
  │                         → mask × image = segmented fruit
  │
  ├─ 3. Feature extraction  colour + texture + shape (fruit)  +  colour + texture (whole image)
  │                         → 108-dimensional feature vector
  │
  ├─ 4. Training            6 reference vectors (class means) + shrunk Mahalanobis distance
  │                         + cross-validated feature-group selection per stage
  │
  └─ 5. Prediction          Stage 1: nearest reference → fruit
                            Stage 2: ripe vs. unripe reference of that fruit → ripeness
```

### 1. Preprocessing

| Step | Detail | Purpose |
|---|---|---|
| Resize | 256 × 256 | Comparable features, fast processing |
| Gaussian blur | 5×5 kernel, σ = 1 | Suppresses noise before thresholding |
| Contrast stretching | 2nd–98th percentile → 0–255, same range for all channels | Normalises dull/bright photos without shifting hue |

The stretched image is used **only to build the mask**; all features are measured on the resized original so true colours are preserved.

### 2. Segmentation

1. Convert to **HSV** and take the **saturation (S) channel**.
2. Apply **Otsu thresholding** to obtain a binary mask.
3. Ensure the object is white (invert if the border is mostly white).
4. Clean the mask: morphological opening/closing, hole filling, keep the largest connected component.
5. If the mask is implausible (< 4 % or > 92 % of the image), fall back to a Lab-space distance-from-border-colour mask.
6. **Segmented fruit = mask × original image.**
7. Erode the mask by 2 px (the *core mask*) so blended edge pixels do not pollute colour/texture statistics.
8. A reliability check (solidity, border contact, area fraction) flags masks that look wrong.

---

## Feature Set

Each image is converted into a **108-dimensional feature vector** in five groups.

### Group 1 — Fruit colour (18)

| Features | Count | Notes |
|---|---|---|
| Hue sin-mean, cos-mean, circular std | 3 | Hue is circular (red wraps around 0°/360°), so it is averaged on a circle |
| Saturation mean, std, skew | 3 | Vividness |
| Value mean, std, skew | 3 | Brightness, dark/shiny regions |
| Lab *a* and *b* mean, std | 4 | Green↔red and blue↔yellow axes; shift strongly during ripening |
| Colour-band fractions | 5 | Fraction of red, orange, yellow, green and dark pixels |

### Group 2 — Fruit texture (25)

| Features | Count | Notes |
|---|---|---|
| GLCM (distances 1 and 3, angles 0°/45°/90°/135°) | 14 | Contrast, dissimilarity, homogeneity, energy, correlation, ASM, entropy — computed on the fruit only (32 gray levels, background ignored) |
| LBP histogram | 10 | Uniform local binary patterns (P = 8, R = 1) |
| Edge density | 1 | Canny edge fraction inside the fruit — captures spots and blemishes |

### Group 3 — Shape (10)

Area fraction, normalised perimeter, circularity, aspect ratio (minimum-area rotated rectangle), eccentricity, solidity, extent and Hu moments 1–3.

### Group 4 — Global colour (30)

The Group 1 colour features computed over the **whole frame**, plus a 12-bin hue histogram.

### Group 5 — Global texture (25)

The Group 2 texture features computed over the **whole frame**.

> **Why global features?** In orchard-style photos, green fruit and green leaves are equally saturated, so an S-channel threshold cannot isolate the fruit. Whole-image features still describe the scene and give the classifier something reliable. When masks are good, the fruit-only groups can be selected instead.

---

## Classification Method

### Reference vectors
For each of the **6 classes** (apple/banana/orange × ripe/unripe), the **mean feature vector** of its training images is stored as that class's reference vector (prototype).

### Distance metric — shrunk Mahalanobis
1. Features are z-scored.
2. A **pooled within-class covariance** matrix is estimated.
3. The covariance is **shrunk towards the identity** by a factor α for numerical stability with many features.
4. Distance = Mahalanobis distance to the reference vector, which accounts for feature scale and correlation.

### Two stages
- **Stage 1 — which fruit?** A fruit's distance is the smaller of its ripe/unripe reference distances. The nearest fruit wins.
- **Stage 2 — ripe or unripe?** Only the two references of the predicted fruit are compared, with features standardised using that fruit's own statistics. A softmax over the negative squared distances gives a ripeness probability.

### Automatic feature-group selection
For each stage, all 31 combinations of the 5 feature groups are scored with **5-fold stratified cross-validation on the training data only**. The best combination is kept (fewer groups win ties), and α is then tuned over `{0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8}`. The test set is never used for selection.

---

## Project Structure

```
FruitRipeness/
├── app.py              # Streamlit dashboard (front end)
├── main.py             # Core pipeline: segmentation, features, classifier, evaluation
├── requirements.txt    # Python dependencies
├── README.md
├── .gitignore
├── dataset/            # Your image data (not tracked by Git)
│   └── dataset1/
│       ├── train/
│       └── test/
└── .fruit_cache/       # Auto-generated feature cache (not tracked by Git)
```

> If your backend module has a different file name, make sure the `import` line at the top of `app.py` matches it.

---

## Dataset Layout

The loader reads labels from **folder names**:

```
dataset/
└── dataset1/
    ├── train/
    │   ├── apples/
    │   │   ├── raw/      (or unripe/)
    │   │   └── ripe/
    │   ├── banana/
    │   │   ├── raw/
    │   │   └── ripe/
    │   └── orange/
    │       ├── raw/
    │       └── ripe/
    └── test/             # optional – same structure
```

**Recognised folder names**

| Meaning | Accepted names (case-insensitive) |
|---|---|
| Apple | `apple`, `apples` |
| Banana | `banana`, `bananas` |
| Orange | `orange`, `oranges` |
| Unripe | `raw`, `unripe`, `green` |
| Ripe | `ripe` |

Supported image formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`.

If no `test` folder is found, the app automatically uses a **stratified 80/20 hold-out split** of the training folder.

---

## Installation

**Requirements:** Python 3.9+

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# 2. (Recommended) create a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

**Dependencies:** `numpy`, `opencv-python`, `scikit-image`, `scikit-learn`, `scipy`, `pandas`, `matplotlib`, `seaborn`, `joblib`, `streamlit`.

Place your dataset in the `dataset/` folder following the [layout above](#dataset-layout).

---

## Usage

```bash
streamlit run app.py
```

The dashboard opens in your browser. On the **first run** the app extracts features from every image and caches them in `.fruit_cache/`, so later launches are fast. It then builds the reference vectors and (if enabled) selects the best feature groups for each stage.

**To classify an image:** open the **Predict** tab and upload a photo.

---

## Using a Different Dataset

You do **not** need to change any code. Everything is controlled from the sidebar:

1. Put the new dataset in the project, e.g. `dataset/dataset2/` with `train/` (and optionally `test/`) using the [same folder structure](#dataset-layout).
2. In the sidebar, set:
   - **Training dataset directory** → `dataset/dataset2/train`
   - **Test dataset directory (optional)** → `dataset/dataset2/test`
3. Click **Reload dataset / clear cache** (recommended when switching datasets).

The app re-extracts features, rebuilds the reference vectors and re-runs feature selection for the new data. Absolute paths (for example `D:/data/fruits/train`) also work, and paths are resolved relative to the folder where you ran `streamlit run`.

To change the **default** paths shown when the app opens, edit the two `st.sidebar.text_input` lines near the top of `app.py`.

---

## The Dashboard

| Tab | What it shows |
|---|---|
| **🔍 Predict** | Upload a photo → predicted fruit and ripeness with probability, the feature groups used by each stage, an unreliable-mask warning when relevant, distance charts for both stages, and the full feature vector |
| **📊 EDA** | Class counts, the cleanest segmentation example per class (raw / mask / segmented), feature-group correlation heatmaps, per-class boxplots for any feature, and a PCA scatter plot |
| **📈 Performance** | Stage 1, Stage 2 and overall accuracy on unseen images, confusion matrices and classification reports, and the cross-validation table behind feature-group selection |
| **📐 Reference vectors** | The six class-mean reference vectors across all features |

**Sidebar controls:** training/test dataset paths, automatic feature-group selection toggle, and a reload/clear-cache button.

---

## Evaluation

All reported metrics are computed on **images not used for training** (your `test` folder, or a stratified hold-out split).

| Metric | Meaning |
|---|---|
| Stage 1 accuracy | Fruit identified correctly |
| Stage 2 accuracy (true fruit) | Ripeness correct when the fruit is given — isolates Stage 2 |
| Stage 2 accuracy (pipeline) | Ripeness correct using Stage 1's prediction |
| Overall 6-class accuracy | Both fruit and ripeness correct |

Confusion matrices and per-class precision / recall / F1 are available in the **Performance** tab.

**Results** *(fill in with your own run)*

| Dataset | Stage 1 | Stage 2 | Overall |
|---|---|---|---|
| dataset1 | XX.X % | XX.X % | XX.X % |

---

## Tech Stack

- **Language:** Python
- **Image processing:** OpenCV, scikit-image, SciPy
- **Numerics / data:** NumPy, pandas
- **Parallelism:** joblib
- **Metrics / validation:** scikit-learn (cross-validation splits, accuracy, confusion matrices — the classifier itself is plain NumPy)
- **Visualisation:** Matplotlib, seaborn
- **Front end:** Streamlit

---



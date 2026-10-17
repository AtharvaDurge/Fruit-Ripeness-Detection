<div align="center">

# 🍎 🍌 🍊 Fruit Ripeness Detection System

**Is it an apple, a banana or an orange — and is it ripe?**
Answered with classical image processing alone. No neural networks, no black boxes.

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Image%20Processing-5C3EE8?logo=opencv&logoColor=white)
![scikit-image](https://img.shields.io/badge/scikit--image-Features-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)
![Approach](https://img.shields.io/badge/Approach-No%20Deep%20Learning-success)

</div>

---

## 📖 Overview

Give the system a photo of an **apple, banana or orange** and it answers two questions:

1. **Which fruit is it?**
2. **Is it ripe or unripe?**

It does this with handcrafted **colour, texture and shape features** and a **two-stage reference-vector classifier**. Every decision can be traced back to a distance between feature vectors, so the whole system is fully explainable. An interactive **Streamlit dashboard** covers prediction, exploratory data analysis (EDA) and evaluation.

### ⚡ At a glance

| | |
|---|---|
| **Fruits** | Apple · Banana · Orange |
| **Output** | Fruit type + ripe / unripe |
| **Segmentation** | HSV saturation channel + Otsu thresholding |
| **Features** | 108 handcrafted features in 5 groups |
| **Classifier** | Two-stage nearest-reference-vector, shrunk Mahalanobis distance |
| **Evaluation** | Unseen test images, confusion matrices, cross-validated feature selection |
| **Interface** | Streamlit dashboard |

---

## 📑 Table of Contents

- [✨ Highlights](#-highlights)
- [🔬 How It Works](#-how-it-works)
- [🧬 Feature Set](#-feature-set)
- [🎯 Classification Method](#-classification-method)
- [📁 Project Structure](#-project-structure)
- [🗂️ Dataset Layout](#️-dataset-layout)
- [🚀 Installation](#-installation)
- [▶️ Usage](#️-usage)
- [🔄 Using a Different Dataset](#-using-a-different-dataset)
- [🖥️ The Dashboard](#️-the-dashboard)
- [📊 Evaluation](#-evaluation)
- [🛠️ Tech Stack](#️-tech-stack)

---

## ✨ Highlights

- 🧠 **Strictly traditional pipeline.** HSV/Otsu segmentation, GLCM and LBP texture, shape descriptors. Nothing is learned by a neural network.
- 🪜 **Two-stage classification.** Stage 1 identifies the fruit. Stage 2 decides ripe vs. unripe using only that fruit's references.
- 🔢 **108 handcrafted features** across five groups: fruit colour, fruit texture, shape, whole-image colour and whole-image texture.
- 🛡️ **Robust to imperfect masks.** Whole-image features act as a safety net when the fruit cannot be separated from the background, for example green fruit among green leaves.
- 🤖 **Automatic feature-group selection** for each stage, using cross-validation on the training data only.
- ✅ **Honest evaluation** on images the model has never seen (a separate test folder, or a stratified hold-out split).
- 🖥️ **Interactive dashboard** with prediction, EDA, performance metrics and reference vectors.
- 🔄 **Dataset-agnostic.** Switch datasets from the sidebar, with no code changes.

---

## 🔬 How It Works

```mermaid
flowchart LR
    A[📷 Photo] --> B[1. Preprocessing<br/>resize · blur · contrast stretch]
    B --> C[2. Segmentation<br/>HSV S-channel · Otsu · clean mask]
    C --> D[3. Feature extraction<br/>108 features]
    D --> E[Stage 1<br/>Which fruit?]
    E --> F[Stage 2<br/>Ripe or unripe?]
    F --> G[✅ Result]
```

### 1️⃣ Preprocessing

| Step | Detail | Purpose |
|---|---|---|
| **Resize** | 256 × 256 | Comparable features and fast processing |
| **Gaussian blur** | 5×5 kernel, σ = 1 | Suppresses noise before thresholding |
| **Contrast stretching** | 2nd–98th percentile → 0–255, same range for all channels | Normalises dull or bright photos without shifting hue |

> The stretched image is used **only to build the mask**. All features are measured on the resized original, so true colours are preserved.

### 2️⃣ Segmentation

1. Convert to **HSV** and take the **saturation (S) channel**.
2. Apply **Otsu thresholding** to get a binary mask.
3. Make sure the object is white (invert if the border is mostly white).
4. Clean the mask: morphological opening and closing, hole filling, keep the largest connected component.
5. If the mask is implausible (< 4 % or > 92 % of the image), fall back to a Lab-space distance-from-border-colour mask.
6. **Segmented fruit = mask × original image.**
7. Erode the mask by 2 px (the *core mask*) so blended edge pixels don't pollute colour and texture statistics.
8. A reliability check (solidity, border contact, area fraction) flags masks that look wrong.

### 3️⃣ Feature extraction

Each image becomes a **108-dimensional feature vector** (details in the next section).

### 4️⃣ Training

- Build **6 reference vectors**, the mean feature vector of each class.
- Estimate a **shrunk Mahalanobis** distance.
- Select the best feature groups for each stage by cross-validation.

### 5️⃣ Prediction

- **Stage 1:** the nearest reference vector gives the fruit.
- **Stage 2:** the ripe vs. unripe reference of that fruit gives the ripeness.

---

## 🧬 Feature Set

Each image is converted into a **108-dimensional feature vector** in five groups.

| Group | Features | Measured on |
|---|:---:|---|
| 🎨 **1. Fruit colour** | 18 | Fruit only |
| 🔍 **2. Fruit texture** | 25 | Fruit only |
| 📐 **3. Shape** | 10 | Binary mask |
| 🌍 **4. Global colour** | 30 | Whole image |
| 🌐 **5. Global texture** | 25 | Whole image |
| **Total** | **108** | |

<details>
<summary><b>🎨 Group 1: Fruit colour (18)</b></summary>

| Features | Count | Notes |
|---|:---:|---|
| Hue sin-mean, cos-mean, circular std | 3 | Hue is circular (red wraps around 0°/360°), so it is averaged on a circle |
| Saturation mean, std, skew | 3 | Vividness |
| Value mean, std, skew | 3 | Brightness, dark or shiny regions |
| Lab *a* and *b* mean, std | 4 | Green↔red and blue↔yellow axes, which shift strongly during ripening |
| Colour-band fractions | 5 | Fraction of red, orange, yellow, green and dark pixels |

</details>

<details>
<summary><b>🔍 Group 2: Fruit texture (25)</b></summary>

| Features | Count | Notes |
|---|:---:|---|
| **GLCM** (distances 1 and 3, angles 0°/45°/90°/135°) | 14 | Contrast, dissimilarity, homogeneity, energy, correlation, ASM and entropy, computed on the fruit only (32 gray levels, background ignored) |
| **LBP histogram** | 10 | Uniform local binary patterns (P = 8, R = 1) |
| **Edge density** | 1 | Canny edge fraction inside the fruit, which captures spots and blemishes |

</details>

<details>
<summary><b>📐 Group 3: Shape (10)</b></summary>

Area fraction, normalised perimeter, circularity, aspect ratio (minimum-area rotated rectangle), eccentricity, solidity, extent and Hu moments 1–3.

</details>

<details>
<summary><b>🌍 Group 4: Global colour (30)</b></summary>

The Group 1 colour features computed over the **whole frame**, plus a 12-bin hue histogram.

</details>

<details>
<summary><b>🌐 Group 5: Global texture (25)</b></summary>

The Group 2 texture features computed over the **whole frame**.

</details>

> [!NOTE]
> **Why global features?** In orchard-style photos, green fruit and green leaves are equally saturated, so an S-channel threshold cannot isolate the fruit. Whole-image features still describe the scene and give the classifier something reliable. When masks are good, the fruit-only groups can be selected instead.

---

## 🎯 Classification Method

### 📌 Reference vectors
For each of the **6 classes** (apple / banana / orange × ripe / unripe), the **mean feature vector** of its training images is stored as that class's reference vector (its prototype).

### 📏 Distance metric: shrunk Mahalanobis
1. Features are z-scored.
2. A **pooled within-class covariance** matrix is estimated.
3. The covariance is **shrunk towards the identity** by a factor α, for numerical stability with many features.
4. Distance = Mahalanobis distance to the reference vector, which accounts for feature scale and correlation.

### 🪜 Two stages

| Stage | Question | How it decides |
|---|---|---|
| **Stage 1** | Which fruit? | A fruit's distance is the smaller of its ripe and unripe reference distances. The nearest fruit wins. |
| **Stage 2** | Ripe or unripe? | Only the two references of the predicted fruit are compared, with features standardised using that fruit's own statistics. A softmax over the negative squared distances gives a ripeness probability. |

### 🤖 Automatic feature-group selection
For each stage, all **31 combinations** of the 5 feature groups are scored with **5-fold stratified cross-validation on the training data only**. The best combination is kept (fewer groups win ties), and α is then tuned over `{0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8}`.

> [!IMPORTANT]
> The test set is **never** used for selection or tuning.

---

## 📁 Project Structure

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

## 🗂️ Dataset Layout

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
    └── test/             # optional, same structure
```

**Recognised folder names** (case-insensitive)

| Meaning | Accepted names |
|---|---|
| 🍎 Apple | `apple`, `apples` |
| 🍌 Banana | `banana`, `bananas` |
| 🍊 Orange | `orange`, `oranges` |
| 🟢 Unripe | `raw`, `unripe`, `green` |
| 🟠 Ripe | `ripe` |

Supported image formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`.

If no `test` folder is found, the app automatically uses a **stratified 80/20 hold-out split** of the training folder.

---

## 🚀 Installation

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

Place your dataset in the `dataset/` folder following the [layout above](#️-dataset-layout).

---

## ▶️ Usage

```bash
streamlit run app.py
```

The dashboard opens in your browser. On the **first run** the app extracts features from every image and caches them in `.fruit_cache/`, so later launches are fast. It then builds the reference vectors and, if enabled, selects the best feature groups for each stage.

**To classify an image:** open the **Predict** tab and upload a photo.

---

## 🔄 Using a Different Dataset

> [!TIP]
> You do **not** need to change any code. Everything is controlled from the sidebar.

1. Put the new dataset in the project, for example `dataset/dataset2/`, with `train/` (and optionally `test/`) using the [same folder structure](#️-dataset-layout).
2. In the sidebar, set:
   - **Training dataset directory** → `dataset/dataset2/train`
   - **Test dataset directory (optional)** → `dataset/dataset2/test`
3. Click **Reload dataset / clear cache** (recommended when switching datasets).

The app re-extracts features, rebuilds the reference vectors and re-runs feature selection for the new data. Absolute paths (for example `D:/data/fruits/train`) also work, and paths are resolved relative to the folder where you ran `streamlit run`.

To change the **default** paths shown when the app opens, edit the two `st.sidebar.text_input` lines near the top of `app.py`.

---

## 🖥️ The Dashboard

| Tab | What it shows |
|---|---|
| **🔍 Predict** | Upload a photo → predicted fruit and ripeness with probability, the feature groups used by each stage, an unreliable-mask warning when relevant, distance charts for both stages, and the full feature vector |
| **📊 EDA** | Class counts, the cleanest segmentation example per class (raw / mask / segmented), feature-group correlation heatmaps, per-class boxplots for any feature, and a PCA scatter plot |
| **📈 Performance** | Stage 1, Stage 2 and overall accuracy on unseen images, confusion matrices and classification reports, and the cross-validation table behind feature-group selection |
| **📐 Reference vectors** | The six class-mean reference vectors across all features |

**Sidebar controls:** training and test dataset paths, an automatic feature-group selection toggle, and a reload / clear-cache button.

---

## 📊 Evaluation

All reported metrics are computed on **images not used for training** (your `test` folder, or a stratified hold-out split).

| Metric | Meaning |
|---|---|
| **Stage 1 accuracy** | Fruit identified correctly |
| **Stage 2 accuracy (true fruit)** | Ripeness correct when the fruit is given, which isolates Stage 2 |
| **Stage 2 accuracy (pipeline)** | Ripeness correct using Stage 1's prediction |
| **Overall 6-class accuracy** | Both fruit and ripeness correct |

Confusion matrices and per-class precision / recall / F1 are available in the **Performance** tab.

### 🏆 Results

*Fill in with your own run.*

| Dataset | Stage 1 | Stage 2 | Overall |
|---|:---:|:---:|:---:|
| dataset1 | XX.X % | XX.X % | XX.X % |

---

## 🛠️ Tech Stack

| Area | Tools |
|---|---|
| **Language** | Python |
| **Image processing** | OpenCV, scikit-image, SciPy |
| **Numerics / data** | NumPy, pandas |
| **Parallelism** | joblib |
| **Metrics / validation** | scikit-learn (cross-validation splits, accuracy, confusion matrices; the classifier itself is plain NumPy) |
| **Visualisation** | Matplotlib, seaborn |
| **Front end** | Streamlit |
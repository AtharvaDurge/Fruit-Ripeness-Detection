import os
import cv2
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.stats import skew
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.measure import label, regionprops

IMG_SIZE = 256
FRUITS = ["apple", "banana", "orange"]
CLASSES = [f"{f}_{c}" for f in FRUITS for c in ("ripe", "unripe")]   # 6 classes
FRUIT_ALIAS = {"apple": "apple", "apples": "apple", "banana": "banana", "bananas": "banana",
               "orange": "orange", "oranges": "orange"}
COND_ALIAS = {"raw": "unripe", "unripe": "unripe", "green": "unripe", "ripe": "ripe"}
IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
GLCM_LEVELS, GLCM_DIST = 32, [1, 3]


# ----------------------------------------------------------------------------
# PREPROCESSING (steps 1-4)
# ----------------------------------------------------------------------------
def resize_square(bgr, size=IMG_SIZE):                      # step 2: 256 x 256
    h, w = bgr.shape[:2]
    scale = size / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    small = cv2.resize(bgr, (nw, nh), interpolation=interp)
    border = np.concatenate([small[0], small[-1], small[:, 0], small[:, -1]])
    colour = [int(c) for c in np.median(border, axis=0)]
    top, left = (size - nh) // 2, (size - nw) // 2
    return cv2.copyMakeBorder(small, top, size - nh - top, left, size - nw - left,
                              cv2.BORDER_CONSTANT, value=colour)


def contrast_stretch(img, pct=(2, 98)):                     # step 4
    lo, hi = np.percentile(img, pct)
    if hi - lo < 1:
        return img.copy()
    return np.clip((img.astype(np.float32) - lo) * 255 / (hi - lo), 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------
# SEGMENTATION (steps 5-10)
# ----------------------------------------------------------------------------
def _border_white(mask):
    b = np.concatenate([mask[0], mask[-1], mask[:, 0], mask[:, -1]])
    return (b > 0).mean()


def _clean(mask):                                           # step 8
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)                    # a) opening
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)                   # b) closing
    mask = ndi.binary_fill_holes(mask > 0).astype(np.uint8)             # c) hole filling
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n > 1:                                                           # d) largest component
        mask = (lab == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
    return mask * 255


def segment(bgr):
    raw = resize_square(bgr)                                            # step 2
    blur = cv2.GaussianBlur(raw, (5, 5), 1.0)                           # step 3
    stretched = contrast_stretch(blur)                                  # step 4
    s = cv2.cvtColor(stretched, cv2.COLOR_BGR2HSV)[..., 1]              # step 5 + S-channel
    thr, otsu = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)   # step 6

    m = otsu.copy()
    if _border_white(m) > 0.5:                                          # step 7: polarity check
        m = 255 - m
    m = _clean(m)                                                       # step 8

    core = cv2.erode(m, np.ones((3, 3), np.uint8), iterations=2)        # step 9: erode by 2 px
    if (core > 0).sum() < 100:
        core = m

    segmented = cv2.bitwise_and(raw, raw, mask=m)
    props = regionprops(label(m > 0))
    solidity = max(props, key=lambda p: p.area).solidity if props else 0.0
    touch = float(_border_white(m))
    frac = float((m > 0).mean())
    reliable = bool(solidity > 0.80 and touch < 0.25 and 0.03 < frac < 0.80)
    # step 10: raw (resized RGB original) + core mask (+ cleaned mask for shape)
    return dict(raw=raw, blur=blur, stretched=stretched, s=s, otsu=otsu, thr=float(thr),
                mask=m, core=core, segmented=segmented,
                solidity=float(solidity), border_touch=touch, area_frac=frac, reliable=reliable)


# ----------------------------------------------------------------------------
# FEATURE NAMES  (A 18 + B 10 + C 25 + D 18 + E 12 + F 25 = 108)
# ----------------------------------------------------------------------------
COLOR_NAMES = ["hue_sin", "hue_cos", "hue_cstd", "s_mean", "s_std", "s_skew", "v_mean", "v_std", "v_skew",
               "lab_a_mean", "lab_b_mean", "lab_a_std", "lab_b_std",
               "frac_red", "frac_orange", "frac_yellow", "frac_green", "frac_dark"]
SHAPE_NAMES = ["area_frac", "perimeter_norm", "circularity", "aspect_ratio", "eccentricity",
               "solidity", "extent", "hu1", "hu2", "hu3"]
GLCM_PROPS = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
TEXTURE_NAMES = ([f"glcm_{p}_d{d}" for d in GLCM_DIST for p in GLCM_PROPS + ["entropy"]] +
                 [f"lbp_{i}" for i in range(10)] + ["edge_density"])
G_COLOR_NAMES = ["g_" + n for n in COLOR_NAMES]
HUE_HIST_NAMES = [f"hue_bin_{i}" for i in range(12)]
G_TEXTURE_NAMES = ["g_" + n for n in TEXTURE_NAMES]

FEATURE_NAMES = (COLOR_NAMES + SHAPE_NAMES + TEXTURE_NAMES +
                 G_COLOR_NAMES + HUE_HIST_NAMES + G_TEXTURE_NAMES)
GROUPS = {"A colour (core)": COLOR_NAMES, "B shape": SHAPE_NAMES, "C texture (core)": TEXTURE_NAMES,
          "D global colour": G_COLOR_NAMES, "E hue histogram": HUE_HIST_NAMES,
          "F global texture": G_TEXTURE_NAMES}
GROUP_OF = {n: g for g, names in GROUPS.items() for n in names}
assert len(FEATURE_NAMES) == 108


# ----------------------------------------------------------------------------
# FEATURE EXTRACTION (step 11)
# ----------------------------------------------------------------------------
def color_features(raw, mask):
    """HSV stats + Lab stats + colour fractions inside `mask`. (18)"""
    m = mask > 0
    hsv = cv2.cvtColor(raw, cv2.COLOR_BGR2HSV)
    h = hsv[..., 0][m].astype(np.float64) * 2.0
    s = hsv[..., 1][m] / 255.0
    v = hsv[..., 2][m] / 255.0
    rad = np.deg2rad(h)
    sm, cm = np.sin(rad).mean(), np.cos(rad).mean()
    R = np.clip(np.hypot(sm, cm), 1e-9, 1)
    f = [sm, cm, np.sqrt(-2 * np.log(R))]
    for ch in (s, v):
        f += [ch.mean(), ch.std(), skew(ch) if ch.std() > 1e-9 else 0.0]
    lab = cv2.cvtColor(raw, cv2.COLOR_BGR2LAB).astype(np.float64)
    a, b = lab[..., 1][m] - 128, lab[..., 2][m] - 128
    f += [a.mean(), b.mean(), a.std(), b.std()]
    ok = (s > 0.2) & (v > 0.2)
    n = len(h)
    red = ok & ((h < 12) | (h >= 340))
    orange = ok & (h >= 12) & (h < 35)
    yellow = ok & (h >= 35) & (h < 65)
    green = ok & (h >= 65) & (h < 170)
    f += [red.sum() / n, orange.sum() / n, yellow.sum() / n, green.sum() / n, (v < 0.25).sum() / n]
    return f


def shape_features(mask):
    """From cleaned binary mask. (10)"""
    r = max(regionprops(label(mask > 0)), key=lambda p: p.area)
    H, W = mask.shape
    A, P = r.area, max(r.perimeter, 1e-6)
    cnt, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    (_, _), (w, h), _ = cv2.minAreaRect(max(cnt, key=cv2.contourArea))
    ar = max(w, h) / max(min(w, h), 1e-6)
    hu = -np.sign(r.moments_hu) * np.log10(np.abs(r.moments_hu) + 1e-30)
    return [A / (H * W), P / (H + W), min(4 * np.pi * A / P ** 2, 1.5), ar, r.eccentricity,
            r.solidity, r.extent, hu[0], hu[1], hu[2]]


def texture_features(raw, mask):
    """Grayscale -> mask -> normalise -> GLCM + LBP + edge density. (25)"""
    gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    m = mask > 0
    lo, hi = np.percentile(gray[m], (2, 98))
    g = np.clip((gray.astype(np.float32) - lo) / max(hi - lo, 1), 0, 1)   # normalise intensities
    ys, xs = np.where(m)
    sl = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
    mm = m[sl]
    L = GLCM_LEVELS
    q = (g[sl] * (L - 2)).astype(np.uint8) + 1
    q[~mm] = 0
    glcm = graycomatrix(q, GLCM_DIST, [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                        levels=L, symmetric=True, normed=False).astype(np.float64)
    glcm[0], glcm[:, 0] = 0, 0
    tot = glcm.sum(axis=(0, 1), keepdims=True)
    glcm = np.divide(glcm, tot, out=np.zeros_like(glcm), where=tot > 0)
    f = []
    for di in range(len(GLCM_DIST)):
        gi = glcm[:, :, di:di + 1, :]
        f += [float(np.nan_to_num(graycoprops(gi, p).mean())) for p in GLCM_PROPS]
        p_ = gi[gi > 0]
        f.append(float(-(p_ * np.log2(p_)).sum() / gi.shape[-1]))
    g8 = (g * 255).astype(np.uint8)
    lbp = local_binary_pattern(g8, 8, 1, "uniform")
    hist = np.bincount(lbp[m].astype(int), minlength=10)[:10].astype(np.float64)
    f += list(hist / max(hist.sum(), 1))
    inner = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=4) > 0
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 40, 120) > 0
    f.append(edges[inner].mean() if inner.any() else 0.0)
    return f


def hue_histogram(raw, bins=12):
    """Raw -> HSV -> hue (0-360) split into 12 bins. (12)"""
    h = cv2.cvtColor(raw, cv2.COLOR_BGR2HSV)[..., 0].astype(np.float64).ravel() * 2.0
    hist = np.histogram(h, bins=bins, range=(0, 360))[0].astype(np.float64)
    return list(hist / h.size)


def extract(bgr):
    seg = segment(bgr)
    if (seg["mask"] > 0).sum() < 100:
        raise ValueError("object not found")
    raw, core, mask = seg["raw"], seg["core"], seg["mask"]
    full = np.full(mask.shape, 255, np.uint8)                 # full mask: all pixels white
    vec = (color_features(raw, core)                          # A  colour (core mask)      18
           + shape_features(mask)                             # B  shape (cleaned mask)    10
           + texture_features(raw, core)                      # C  texture (core mask)     25
           + color_features(raw, full)                        # D  global colour           18
           + hue_histogram(raw)                               # E  hue histogram           12
           + texture_features(raw, full))                     # F  global texture          25
    vec = np.nan_to_num(np.array(vec, dtype=np.float64))
    return dict(zip(FEATURE_NAMES, vec)), seg


# ----------------------------------------------------------------------------
# CLASS FEATURE VECTORS (steps 12-13)
# ----------------------------------------------------------------------------
def list_images(base):
    rows = []
    for root, _, files in os.walk(base):
        parts = [p.lower() for p in os.path.relpath(root, base).split(os.sep)]
        fruit = next((FRUIT_ALIAS[p] for p in parts if p in FRUIT_ALIAS), None)
        cond = next((COND_ALIAS[p] for p in parts if p in COND_ALIAS), None)
        if not (fruit and cond):
            continue
        for fn in files:
            if fn.lower().endswith(IMG_EXT):
                rows.append(dict(path=os.path.join(root, fn), filename=fn, true_fruit=fruit, true_cond=cond))
    return pd.DataFrame(rows)


def build_prototypes(train_dir, save_path="prototypes.npz", progress=None):
    """Step 12: average the 108-feature vectors of each class  ->  6 class vectors.
       Step 13: standardise (scale) those vectors."""
    df = list_images(train_dir)
    per = {k: [] for k in CLASSES}
    for i, row in enumerate(df.itertuples(index=False)):
        try:
            feats, _ = extract(cv2.imread(row.path))
            per[f"{row.true_fruit}_{row.true_cond}"].append([feats[n] for n in FEATURE_NAMES])
        except Exception:
            pass
        if progress:
            progress((i + 1) / len(df))
    missing = [k for k, v in per.items() if not v]
    if missing:
        raise ValueError(f"No usable train images for: {missing}")
    counts = np.array([len(per[k]) for k in CLASSES])
    raw = np.array([np.mean(per[k], axis=0) for k in CLASSES])     # 6 x 108
    mu, sd = raw.mean(axis=0), raw.std(axis=0)
    sd[sd < 1e-9] = 1.0
    proto = dict(classes=np.array(CLASSES), raw=raw, mu=mu, sd=sd, scaled=(raw - mu) / sd, counts=counts)
    np.savez(save_path, **proto)
    return proto


def load_prototypes(path="prototypes.npz"):
    return dict(np.load(path)) if os.path.exists(path) else None


# ----------------------------------------------------------------------------
# CLASSIFICATION: scale the image's vector the same way, pick the closest class vector
# ----------------------------------------------------------------------------
def classify(feats, proto):
    x = np.array([feats[n] for n in FEATURE_NAMES], dtype=np.float64)
    z = (x - proto["mu"]) / proto["sd"]
    d = np.linalg.norm(proto["scaled"] - z, axis=1)
    order = np.argsort(d)
    best = str(proto["classes"][order[0]])
    fruit, cond = best.split("_")
    margin = float(1 - d[order[0]] / (d[order[1]] + 1e-9))
    return dict(fruit=fruit, condition=cond, confidence=margin,
                distances={str(c): float(v) for c, v in zip(proto["classes"], d)})


def evaluate_folder(base, proto, progress=None):
    df = list_images(base)
    if df.empty:
        return df
    out = []
    for i, row in enumerate(df.itertuples(index=False)):
        try:
            feats, seg = extract(cv2.imread(row.path))
            res = classify(feats, proto)
            out.append(dict(row._asdict(), pred_fruit=res["fruit"], pred_cond=res["condition"],
                            confidence=res["confidence"], reliable=seg["reliable"]))
        except Exception:
            pass
        if progress:
            progress((i + 1) / len(df))
    return pd.DataFrame(out)
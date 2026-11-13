"""
Fruit ripeness detection - PURE classical image processing (no AI / ML / deep learning).

Only these techniques are used: resizing, Gaussian blur, contrast stretching, colour-space
conversion (HSV / Lab), Otsu thresholding, morphology, connected components, convex hull,
hand-crafted colour / shape / texture (GLCM, LBP, Canny) features, and nearest-template
matching against class-average feature vectors computed from the reference images.
Nothing is trained: no weights, no gradient descent, no fitted model.
"""
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
FEATURE_VERSION = 3          # bump when feature definitions change -> old prototypes.npz is ignored


# ----------------------------------------------------------------------------
# PREPROCESSING
# ----------------------------------------------------------------------------
def resize_square(bgr, size=IMG_SIZE):
    h, w = bgr.shape[:2]
    interp = cv2.INTER_AREA if max(h, w) > size else cv2.INTER_LINEAR
    return cv2.resize(bgr, (size, size), interpolation=interp)


def contrast_stretch(img, pct=(2, 98)):
    lo, hi = np.percentile(img, pct)
    if hi - lo < 1:
        return img.copy()
    return np.clip((img.astype(np.float32) - lo) * 255 / (hi - lo), 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------
# SEGMENTATION  (several candidate masks -> keep the most reliable one)
# ----------------------------------------------------------------------------
def _border_pixels(a):
    return np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])


def _border_white(mask):
    return float((_border_pixels(mask) > 0).mean())


def _clean(mask):
    """opening -> closing -> hole filling -> keep every component >= 20% of the biggest."""
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = ((mask > 0).astype(np.uint8)) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k7)
    mask = ndi.binary_fill_holes(mask > 0).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        big = areas.max()
        thr = min(big, max(0.2 * big, 150))
        keep = [i + 1 for i, a in enumerate(areas) if a >= thr]
        mask = np.isin(lab, keep).astype(np.uint8)
    return mask * 255


def _hull_fill(mask):
    """Compact blobs (solidity > 0.8) are replaced by their convex hull.
       This recovers bites caused by specular highlights / pale patches on a fruit."""
    out = np.zeros_like(mask)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        hull = cv2.convexHull(c)
        a, ha = cv2.contourArea(c), cv2.contourArea(hull)
        cv2.drawContours(out, [hull if (ha > 0 and a / ha > 0.8) else c], -1, 255, -1)
    return out


def _bg_distance(lab, bg):
    """Distance of every pixel from the border (background) colour; L weighted low so
       soft shadows are not mistaken for fruit."""
    d = lab - bg
    dist = np.sqrt(d[..., 1] ** 2 + d[..., 2] ** 2 + (0.3 * d[..., 0]) ** 2)
    d8 = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def _warm_hue(bgr):
    """Saturated red / orange / yellow pixels - isolates ripe fruit in cluttered scenes
       (e.g. oranges on a tree full of green leaves)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0].astype(np.int32) * 2, hsv[..., 1], hsv[..., 2]
    return (((h < 58) | (h >= 340)) & (s > 110) & (v > 80)).astype(np.uint8) * 255


def _score(m, bonus=0.0):
    props = regionprops(label(m > 0))
    if not props:
        return -9.0
    p = max(props, key=lambda q: q.area)
    frac = float((m > 0).mean())
    pen = 0.0 if 0.02 < frac < 0.85 else 0.6
    return float(p.solidity - 1.5 * _border_white(m) - pen + bonus)


def segment(bgr):
    raw = resize_square(bgr)
    blur = cv2.GaussianBlur(raw, (5, 5), 1.0)
    stretched = contrast_stretch(blur)
    s = cv2.cvtColor(stretched, cv2.COLOR_BGR2HSV)[..., 1]
    thr, otsu = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # is the background plain (studio photo) or cluttered (tree, basket, table)?
    lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB).astype(np.float32)
    border = _border_pixels(lab)
    bg, spread = np.median(border, axis=0), border.std(axis=0)
    uniform = bool(spread[1] < 6 and spread[2] < 6 and spread[0] < 25)

    cands = {"saturation-Otsu": (otsu, 0.0),
             "background-distance": (_bg_distance(lab, bg), 0.10 if uniform else 0.0),
             "warm-hue": (_warm_hue(blur), -0.05 if uniform else 0.15)}
    best = None
    for name, (c, bonus) in cands.items():
        c = c.copy()
        if name != "warm-hue" and _border_white(c) > 0.5:      # polarity check
            c = 255 - c
        m = _hull_fill(_clean(c))
        sc = _score(m, bonus)
        if best is None or sc > best[0]:
            best = (sc, name, c, m)
    _, method, cand_raw, m = best

    core = cv2.erode(m, np.ones((3, 3), np.uint8), iterations=2)
    if (core > 0).sum() < 100:
        core = m
    segmented = cv2.bitwise_and(raw, raw, mask=m)
    props = regionprops(label(m > 0))
    solidity = max(props, key=lambda p: p.area).solidity if props else 0.0
    touch = _border_white(m)
    frac = float((m > 0).mean())
    reliable = bool(solidity > 0.80 and touch < 0.25 and 0.02 < frac < 0.85)
    return dict(raw=raw, blur=blur, stretched=stretched, s=s, cand_raw=cand_raw, method=method,
                uniform_bg=uniform, thr=float(thr), mask=m, core=core, segmented=segmented,
                solidity=float(solidity), border_touch=float(touch), area_frac=frac, reliable=reliable)


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

# Group weights (hand-set, tunable). Each group is further divided by sqrt(group size)
# so that big groups (texture, 25 features) do not swamp small ones (shape, 10).
# Stage 1 decides the FRUIT: shape + hue matter most.
FRUIT_W = {"A colour (core)": 1.0, "B shape": 1.5, "C texture (core)": 0.6,
           "D global colour": 0.6, "E hue histogram": 1.0, "F global texture": 0.3}
# Stage 2 decides RIPE / UNRIPE for that fruit: colour matters most, texture hardly.
COND_W = {"A colour (core)": 1.5, "B shape": 0.2, "C texture (core)": 0.3,
          "D global colour": 1.0, "E hue histogram": 1.5, "F global texture": 0.0}
Z_CLIP = 4.0                 # limit the influence of any single outlier feature


# ----------------------------------------------------------------------------
# FEATURE EXTRACTION
# ----------------------------------------------------------------------------
def color_features(raw, mask):
    """HSV stats + Lab stats + colour fractions inside `mask`. (18)
       Hue statistics ignore colourless pixels (hue is meaningless when S or V is tiny)."""
    m = mask > 0
    hsv = cv2.cvtColor(raw, cv2.COLOR_BGR2HSV)
    h = hsv[..., 0][m].astype(np.float64) * 2.0
    s = hsv[..., 1][m] / 255.0
    v = hsv[..., 2][m] / 255.0
    n = max(len(h), 1)

    good = (s > 0.15) & (v > 0.15)
    hh = h[good] if good.sum() >= 20 else h
    rad = np.deg2rad(hh)
    sm, cm = np.sin(rad).mean(), np.cos(rad).mean()
    R = np.clip(np.hypot(sm, cm), 1e-9, 1)
    f = [sm, cm, np.sqrt(-2 * np.log(R))]
    for ch in (s, v):
        f += [ch.mean(), ch.std(), skew(ch) if ch.std() > 1e-9 else 0.0]

    lab = cv2.cvtColor(raw, cv2.COLOR_BGR2LAB).astype(np.float64)
    a, b = lab[..., 1][m] - 128, lab[..., 2][m] - 128
    f += [a.mean(), b.mean(), a.std(), b.std()]

    ok = (s > 0.2) & (v > 0.2)
    red = ok & ((h < 12) | (h >= 340))
    orange = ok & (h >= 12) & (h < 35)
    yellow = ok & (h >= 35) & (h < 65)
    green = ok & (h >= 65) & (h < 170)
    f += [red.sum() / n, orange.sum() / n, yellow.sum() / n, green.sum() / n, (v < 0.25).sum() / n]
    return f


def shape_features(mask):
    """From the cleaned binary mask (largest component). (10)"""
    lab = label(mask > 0)
    r = max(regionprops(lab), key=lambda p: p.area)
    comp = (lab == r.label).astype(np.uint8)
    H, W = mask.shape
    A, P = r.area, max(r.perimeter, 1e-6)
    cnt, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    (_, _), (w, h), _ = cv2.minAreaRect(max(cnt, key=cv2.contourArea))
    ar = max(w, h) / max(min(w, h), 1e-6)
    hu = -np.sign(r.moments_hu) * np.log10(np.abs(r.moments_hu) + 1e-30)
    return [A / (H * W), P / (H + W), min(4 * np.pi * A / P ** 2, 1.5), ar, r.eccentricity,
            r.solidity, r.extent, hu[0], hu[1], hu[2]]


def texture_features(raw, mask):
    """Grayscale -> mask -> normalise -> GLCM + LBP + edge density. (25)"""
    n_feat = len(TEXTURE_NAMES)
    gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    m = mask > 0
    if m.sum() < 50:
        return [0.0] * n_feat
    lo, hi = np.percentile(gray[m], (2, 98))
    g = np.clip((gray.astype(np.float32) - lo) / max(hi - lo, 1), 0, 1)
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
    inner = cv2.erode(((m.astype(np.uint8)) * 255), np.ones((3, 3), np.uint8), iterations=4) > 0
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 40, 120) > 0
    f.append(edges[inner].mean() if inner.any() else 0.0)
    return f


def hue_histogram(raw, mask, bins=12):
    """Hue (0-360) of FRUIT pixels only, 12 bins, ignoring colourless pixels. (12)
       Normalised by the number of coloured fruit pixels so it always sums to 1."""
    m = mask > 0
    hsv = cv2.cvtColor(raw, cv2.COLOR_BGR2HSV)
    ok = m & (hsv[..., 1] > 40) & (hsv[..., 2] > 40)
    if ok.sum() < 20:
        return [0.0] * bins
    h = hsv[..., 0][ok].astype(np.float64) * 2.0
    hist = np.histogram(h, bins=bins, range=(0, 360))[0].astype(np.float64)
    return list(hist / ok.sum())


def extract(bgr):
    seg = segment(bgr)
    if (seg["mask"] > 0).sum() < 100:
        raise ValueError("object not found")
    raw, core, mask = seg["raw"], seg["core"], seg["mask"]
    vec = (color_features(raw, core)             # A  colour (core mask)            18
           + shape_features(mask)                # B  shape (cleaned mask)          10
           + texture_features(raw, core)         # C  texture (core mask)           25
           + color_features(raw, mask)           # D  colour of the whole fruit     18
           + hue_histogram(raw, mask)            # E  hue histogram (fruit only)    12
           + texture_features(raw, mask))        # F  texture of the whole fruit    25
    vec = np.nan_to_num(np.array(vec, dtype=np.float64))
    return dict(zip(FEATURE_NAMES, vec)), seg


# ----------------------------------------------------------------------------
# REFERENCE (CLASS) FEATURE VECTORS  (built from ONE dataset folder - no train/test split)
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


def build_prototypes(data_dir, save_path="prototypes.npz", progress=None):
    """Average the 108-feature vectors of each class -> 6 class vectors, standardised with
       the mean/std of all dataset images. Images whose segmentation looks unreliable are
       left out of the average (unless that would leave a class with fewer than 3 images)."""
    df = list_images(data_dir)
    if df.empty:
        raise ValueError(f"No images found in '{data_dir}'. Expected <fruit>/<ripe|raw|unripe>/image folders.")
    vecs, labels, rel, files, skipped = [], [], [], [], 0
    for i, row in enumerate(df.itertuples(index=False)):
        try:
            img = cv2.imread(row.path)
            if img is None:
                raise ValueError("unreadable")
            feats, seg = extract(img)
            vecs.append([feats[n] for n in FEATURE_NAMES])
            labels.append(f"{row.true_fruit}_{row.true_cond}")
            rel.append(seg["reliable"])
            files.append(row.filename)
        except Exception:
            skipped += 1
        if progress:
            progress((i + 1) / len(df))
    if not vecs:
        raise ValueError("No usable images.")
    X, labels, rel = np.array(vecs), np.array(labels), np.array(rel)
    missing = [k for k in CLASSES if not (labels == k).any()]
    if missing:
        raise ValueError(f"No usable images for: {missing}")

    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd < 1e-9] = 1.0
    Z = np.clip((X - mu) / sd, -Z_CLIP, Z_CLIP)
    used, idx = np.zeros(len(labels), bool), []
    for k in CLASSES:
        sel = (labels == k) & rel
        sel = sel if sel.sum() >= 3 else (labels == k)
        idx.append(sel)
        used |= sel
    counts = np.array([i.sum() for i in idx])
    raw = np.array([X[i].mean(axis=0) for i in idx])
    scaled = np.array([Z[i].mean(axis=0) for i in idx])
    proto = dict(classes=np.array(CLASSES), raw=raw, mu=mu, sd=sd, scaled=scaled, counts=counts,
                 version=np.array(FEATURE_VERSION), Z=Z, labels=labels, rel=rel, used=used,
                 files=np.array(files), skipped=np.array(skipped))
    np.savez(save_path, **proto)
    return proto


def load_prototypes(path="prototypes.npz"):
    if not os.path.exists(path):
        return None
    p = dict(np.load(path))
    if int(p.get("version", -1)) != FEATURE_VERSION or p["mu"].shape[0] != len(FEATURE_NAMES):
        return None            # built with an older feature set -> must be rebuilt
    return p


# ----------------------------------------------------------------------------
# CLASSIFICATION  (two stages: fruit type, then ripeness of that fruit)
# ----------------------------------------------------------------------------
def _weights(group_w, shape_ok):
    w = np.empty(len(FEATURE_NAMES))
    for i, n in enumerate(FEATURE_NAMES):
        g = GROUP_OF[n]
        gw = 0.0 if (g == "B shape" and not shape_ok) else group_w[g]   # bad mask -> ignore shape
        w[i] = gw / np.sqrt(len(GROUPS[g]))
    return w


def _classify_z(z, scaled, classes, reliable):
    diff = scaled - z
    # stage 1: which fruit?
    d1 = np.linalg.norm(diff * _weights(FRUIT_W, reliable), axis=1)
    per_fruit = {f: min(d1[i] for i, c in enumerate(classes) if c.startswith(f + "_")) for f in FRUITS}
    order = sorted(per_fruit, key=per_fruit.get)
    fruit = order[0]
    fruit_margin = float(1 - per_fruit[order[0]] / (per_fruit[order[1]] + 1e-9))
    # stage 2: ripe or unripe, compared only between that fruit's two reference vectors
    d2 = np.linalg.norm(diff * _weights(COND_W, reliable), axis=1)
    cd = {c: float(d2[i]) for i, c in enumerate(classes) if c.startswith(fruit + "_")}
    dr, du = cd[f"{fruit}_ripe"], cd[f"{fruit}_unripe"]
    cond = "ripe" if dr < du else "unripe"
    cond_margin = float(1 - min(dr, du) / (max(dr, du) + 1e-9))
    return dict(fruit=fruit, condition=cond, confidence=cond_margin, fruit_margin=fruit_margin,
                distances={c: float(v) for c, v in zip(classes, d1)}, cond_distances=cd)


def classify(feats, proto, reliable=True):
    x = np.array([feats[n] for n in FEATURE_NAMES], dtype=np.float64)
    z = np.clip((x - proto["mu"]) / proto["sd"], -Z_CLIP, Z_CLIP)
    return _classify_z(z, proto["scaled"], [str(c) for c in proto["classes"]], reliable)
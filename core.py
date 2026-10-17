"""
Strictly traditional pipeline (no neural nets, no learned classifiers):

 raw BGR -> resize -> Gaussian blur -> contrast stretch -> HSV S-channel -> Otsu mask
         -> mask * raw image = segmented fruit
         -> colour (HSV + Lab + hue-band fractions), texture (masked GLCM + LBP + edge density),
            shape (mask moments) -> feature vector
         -> 6 reference vectors (apple/banana/orange x ripe/unripe) = class mean vectors
         -> Stage 1: which fruit      (Mahalanobis distance to the nearest reference vector)
            Stage 2: ripe or unripe   (Mahalanobis distance to the 2 references of that fruit)
"""
import hashlib
import os
import cv2
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import ndimage as ndi
from scipy.stats import skew
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.measure import label, regionprops

IMG_SIZE = (256, 256)
FRUITS = ["apple", "banana", "orange"]
CONDS = ["unripe", "ripe"]
FRUIT_ALIAS = {"apple": "apple", "apples": "apple", "banana": "banana", "bananas": "banana",
               "orange": "orange", "oranges": "orange"}
COND_ALIAS = {"raw": "unripe", "unripe": "unripe", "green": "unripe", "ripe": "ripe"}
IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
GLCM_LEVELS, GLCM_DIST = 32, [1, 3]
SEED = 42

# ------------------------------------------------------------------ preprocessing / segmentation
def contrast_stretch(img, pct=(2, 98)):
    lo, hi = np.percentile(img, pct)
    if hi - lo < 1:
        return img.copy()
    return np.clip((img.astype(np.float32) - lo) * 255 / (hi - lo), 0, 255).astype(np.uint8)


def _clean(mask):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = ndi.binary_fill_holes(mask > 0).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n > 1:
        mask = (lab == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
    return mask * 255


def _border_white(mask):
    b = np.concatenate([mask[0], mask[-1], mask[:, 0], mask[:, -1]])
    return (b > 0).mean()


def _background_distance_mask(img):
    """Fallback: distance (Lab) from the median border colour, Otsu-thresholded."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    border = np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])
    d = np.linalg.norm(lab - np.median(border, axis=0), axis=2)
    d8 = np.clip(d * 255 / max(d.max(), 1), 0, 255).astype(np.uint8)
    _, m = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return m


def segment(bgr):
    """Otsu threshold on the S channel. Returns all intermediate images."""
    raw = cv2.resize(bgr, IMG_SIZE, interpolation=cv2.INTER_AREA)
    blur = cv2.GaussianBlur(raw, (5, 5), 1.0)
    stretched = contrast_stretch(blur)
    s = cv2.cvtColor(stretched, cv2.COLOR_BGR2HSV)[..., 1]
    thr, m = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if _border_white(m) > 0.5:          # object must be white
        m = 255 - m
    m = _clean(m)
    frac = (m > 0).mean()
    used = "S-channel Otsu"
    if frac < 0.04 or frac > 0.92:   # implausible -> fallback
        m = _clean(_background_distance_mask(stretched))
        used = "fallback: background-colour distance"
    segmented = cv2.bitwise_and(raw, raw, mask=m)
    core = cv2.erode(m, np.ones((3, 3), np.uint8), iterations=2)   # drop blended edge pixels
    if (core > 0).sum() < 100:
        core = m
    props = regionprops(label(m > 0))
    sol = max(props, key=lambda p: p.area).solidity if props else 0.0
    b = np.concatenate([m[0], m[-1], m[:, 0], m[:, -1]])
    touch = float((b > 0).mean())
    frac = float((m > 0).mean())
    reliable = bool(sol > 0.80 and touch < 0.25 and 0.03 < frac < 0.80)
    return dict(raw=raw, stretched=stretched, s=s, thr=float(thr), mask=m, core=core,
                segmented=segmented, method=used, solidity=float(sol), border_touch=touch,
                area_frac=frac, reliable=reliable)

# ------------------------------------------------------------------ features
COLOR_NAMES = ["hue_sin", "hue_cos", "hue_cstd", "s_mean", "s_std", "s_skew", "v_mean", "v_std", "v_skew",
               "lab_a_mean", "lab_b_mean", "lab_a_std", "lab_b_std",
               "frac_red", "frac_orange", "frac_yellow", "frac_green", "frac_dark"]
GLCM_PROPS = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
TEXTURE_NAMES = ([f"glcm_{p}_d{d}" for d in GLCM_DIST for p in GLCM_PROPS + ["entropy"]] +
                 [f"lbp_{i}" for i in range(10)] + ["edge_density"])
SHAPE_NAMES = ["area_frac", "perimeter_norm", "circularity", "aspect_ratio", "eccentricity",
               "solidity", "extent", "hu1", "hu2", "hu3"]
GCOLOR_NAMES = ["g_" + n for n in COLOR_NAMES] + [f"g_huehist_{i}" for i in range(12)]
GTEXTURE_NAMES = ["g_" + n for n in TEXTURE_NAMES]
FEATURE_NAMES = COLOR_NAMES + TEXTURE_NAMES + SHAPE_NAMES + GCOLOR_NAMES + GTEXTURE_NAMES
GROUPS = {"color": COLOR_NAMES, "texture": TEXTURE_NAMES, "shape": SHAPE_NAMES,
          "global_color": GCOLOR_NAMES, "global_texture": GTEXTURE_NAMES}
GROUP_OF = {n: g for g, names in GROUPS.items() for n in names}


def color_features(raw, seg_mask):
    m = seg_mask > 0
    hsv = cv2.cvtColor(raw, cv2.COLOR_BGR2HSV)
    h = hsv[..., 0][m].astype(np.float64) * 2.0           # degrees 0-358
    s = hsv[..., 1][m] / 255.0
    v = hsv[..., 2][m] / 255.0
    rad = np.deg2rad(h)
    sm, cm = np.sin(rad).mean(), np.cos(rad).mean()      # circular hue (red wraps 359 -> 0)
    R = np.clip(np.hypot(sm, cm), 1e-9, 1)
    f = [sm, cm, np.sqrt(-2 * np.log(R))]
    for ch in (s, v):
        f += [ch.mean(), ch.std(), skew(ch) if ch.std() > 1e-9 else 0.0]
    lab = cv2.cvtColor(raw, cv2.COLOR_BGR2LAB).astype(np.float64)
    a, b = lab[..., 1][m] - 128, lab[..., 2][m] - 128
    f += [a.mean(), b.mean(), a.std(), b.std()]
    ok = (s > 0.2) & (v > 0.2)                            # hue only meaningful for coloured pixels
    n = len(h)
    red = ok & ((h < 12) | (h >= 340)); orange = ok & (h >= 12) & (h < 35)
    yellow = ok & (h >= 35) & (h < 65);   green = ok & (h >= 65) & (h < 170)
    f += [red.sum() / n, orange.sum() / n, yellow.sum() / n, green.sum() / n, (v < 0.25).sum() / n]
    return f


def texture_features(raw, seg_mask, core):
    gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    m = core > 0
    # intensity normalisation INSIDE the fruit (texture = pattern, brightness is in the colour features)
    lo, hi = np.percentile(gray[m], (2, 98))
    g = np.clip((gray.astype(np.float32) - lo) / max(hi - lo, 1), 0, 1)
    ys, xs = np.where(m)
    sl = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
    mm = m[sl]
    L = GLCM_LEVELS
    q = (g[sl] * (L - 2)).astype(np.uint8) + 1
    q[~mm] = 0                                            # background -> level 0, ignored below
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
    inner = cv2.erode(core, np.ones((3, 3), np.uint8), iterations=4) > 0
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 40, 120) > 0
    f.append(edges[inner].mean() if inner.any() else 0.0)
    return f


def shape_features(mask):
    r = max(regionprops(label(mask > 0)), key=lambda p: p.area)
    H, W = mask.shape
    A, P = r.area, max(r.perimeter, 1e-6)
    cnt, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    (_, _), (w, h), _ = cv2.minAreaRect(max(cnt, key=cv2.contourArea))   # rotation-invariant aspect
    ar = max(w, h) / max(min(w, h), 1e-6)
    hu = -np.sign(r.moments_hu) * np.log10(np.abs(r.moments_hu) + 1e-30)
    return [A / (H * W), P / (H + W), min(4 * np.pi * A / P ** 2, 1.5), ar, r.eccentricity,
            r.solidity, r.extent, hu[0], hu[1], hu[2]]


def hue_hist(raw, bins=12):
    hsv = cv2.cvtColor(raw, cv2.COLOR_BGR2HSV)
    h = hsv[..., 0].ravel() * 2.0
    ok = (hsv[..., 1].ravel() > 51) & (hsv[..., 2].ravel() > 51)
    hist = np.histogram(h[ok], bins=bins, range=(0, 360))[0].astype(np.float64)
    return list(hist / h.size)


def extract(bgr):
    st = segment(bgr)
    if (st["mask"] > 0).sum() < 100:
        raise ValueError("object not found")
    full = np.full(st["mask"].shape, 255, np.uint8)       # whole frame: robust when the mask fails
    vec = (color_features(st["raw"], st["core"]) + texture_features(st["raw"], st["mask"], st["core"])
           + shape_features(st["mask"])
           + color_features(st["raw"], full) + hue_hist(st["raw"])
           + texture_features(st["raw"], full, full))
    return np.nan_to_num(np.array(vec, dtype=np.float64)), st

# ------------------------------------------------------------------ dataset
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
                rows.append(dict(path=os.path.join(root, fn), filename=fn, fruit=fruit, condition=cond))
    return pd.DataFrame(rows)


def _one(path):
    try:
        img = cv2.imread(path)
        return None if img is None else extract(img)[0]
    except Exception:
        return None


def build_features(base, use_disk_cache=True):
    df = list_images(base)
    if df.empty:
        return df
    key = hashlib.md5((os.path.abspath(base) + str(len(df)) +
                       str(max(os.path.getmtime(p) for p in df.path)) + "v3").encode()).hexdigest()[:10]
    cache = os.path.join(".fruit_cache", f"{key}.csv")
    if use_disk_cache and os.path.exists(cache):
        return pd.read_csv(cache)
    vecs = Parallel(n_jobs=-1)(delayed(_one)(p) for p in df.path)
    ok = [i for i, v in enumerate(vecs) if v is not None]
    out = pd.concat([df.iloc[ok].reset_index(drop=True),
                     pd.DataFrame([vecs[i] for i in ok], columns=FEATURE_NAMES)], axis=1)
    out["category"] = out.fruit + "_" + out.condition
    if use_disk_cache:
        os.makedirs(".fruit_cache", exist_ok=True)
        out.to_csv(cache, index=False)
    return out

# ------------------------------------------------------------------ two-stage reference-vector classifier
def _shrunk_inv(resid, alpha):
    C = np.cov(resid.T, bias=True) + 1e-9 * np.eye(resid.shape[1])
    d = C.shape[0]
    C = (1 - alpha) * C + alpha * (np.trace(C) / d) * np.eye(d)
    return np.linalg.inv(C)


def _d2(Z, m, P):
    D = Z - m
    return ((D @ P) * D).sum(1)


class Stage1:
    """Which fruit? 6 reference vectors (class means); fruit distance = Mahalanobis distance to the
    nearer of its ripe/unripe references (pooled within-class covariance, shrunk towards identity)."""

    def __init__(self, cols, alpha):
        self.cols, self.alpha = np.asarray(cols), alpha

    def fit(self, X, fruit, cond):
        X = np.asarray(X)[:, self.cols]
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        Z = (X - self.mu) / self.sd
        self.ref, res = {}, []
        for f in FRUITS:
            for c in CONDS:
                zz = Z[(fruit == f) & (cond == c)]
                self.ref[(f, c)] = zz.mean(0); res.append(zz - self.ref[(f, c)])
        self.P = _shrunk_inv(np.vstack(res), self.alpha)
        return self

    def dists(self, X):
        Z = (np.asarray(X)[:, self.cols] - self.mu) / self.sd
        return np.column_stack([np.sqrt(np.minimum(*[_d2(Z, self.ref[(f, c)], self.P) for c in CONDS]))
                                for f in FRUITS])

    def predict(self, X):
        return np.array(FRUITS)[self.dists(X).argmin(1)]


class Stage2:
    """Ripe or unripe? Compared only with the two reference vectors of the (predicted) fruit."""

    def __init__(self, cols, alpha):
        self.cols, self.alpha = np.asarray(cols), alpha

    def fit(self, X, fruit, cond):
        X = np.asarray(X)[:, self.cols]
        self.m = {}
        for f in FRUITS:
            mu, sd = X[fruit == f].mean(0), X[fruit == f].std(0) + 1e-9
            refs, res = {}, []
            for c in CONDS:
                zz = (X[(fruit == f) & (cond == c)] - mu) / sd
                refs[c] = zz.mean(0); res.append(zz - refs[c])
            self.m[f] = (mu, sd, refs, _shrunk_inv(np.vstack(res), self.alpha))
        return self

    def dists(self, X, fruits):
        X = np.asarray(X)[:, self.cols]; fruits = np.asarray(fruits)
        out = np.zeros((len(X), 2))
        for f in FRUITS:
            idx = fruits == f
            if idx.any():
                mu, sd, refs, P = self.m[f]
                Z = (X[idx] - mu) / sd
                out[idx] = np.column_stack([np.sqrt(_d2(Z, refs[c], P)) for c in CONDS])
        return out

    def predict(self, X, fruits):
        return np.array(CONDS)[self.dists(X, fruits).argmin(1)]


def group_cols(groups):
    return [i for i, n in enumerate(FEATURE_NAMES) if GROUP_OF[n] in groups]


class TwoStageClassifier:
    def __init__(self, groups1=tuple(GROUPS), groups2=tuple(GROUPS), alpha1=0.3, alpha2=0.3):
        self.groups1, self.groups2 = tuple(groups1), tuple(groups2)
        self.s1 = Stage1(group_cols(groups1), alpha1)
        self.s2 = Stage2(group_cols(groups2), alpha2)

    def fit(self, X, fruit, cond):
        X, fruit, cond = np.asarray(X), np.asarray(fruit), np.asarray(cond)
        self.s1.fit(X, fruit, cond); self.s2.fit(X, fruit, cond)
        self.ref_full = {f"{f}_{c}": X[(fruit == f) & (cond == c)].mean(0) for f in FRUITS for c in CONDS}
        return self

    def reference_table(self):
        return pd.DataFrame(self.ref_full, index=FEATURE_NAMES)

    def predict(self, X):
        pf = self.s1.predict(X)
        return pf, self.s2.predict(X, pf)

    def predict_one(self, x):
        x = np.asarray(x)[None]
        d1 = self.s1.dists(x)[0]
        fruit = FRUITS[int(d1.argmin())]
        d2 = self.s2.dists(x, [fruit])[0]
        p = np.exp(-0.5 * (d2 ** 2 - (d2 ** 2).min())); p /= p.sum()
        return dict(fruit=fruit, condition=CONDS[int(d2.argmin())], d_fruit=dict(zip(FRUITS, d1)),
                    d_cond=dict(zip(CONDS, d2)), prob=dict(zip(CONDS, p)))


def tune(X, fruit, cond, folds=5, alphas=(0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8), progress=None):
    """Pick, for EACH stage, the feature-group subset and shrinkage with the best cross-validated accuracy
    on the training data only (traditional model selection; the test set is never touched)."""
    from itertools import combinations
    from sklearn.model_selection import StratifiedKFold
    X = np.asarray(X); fruit = np.asarray(fruit); cond = np.asarray(cond)
    strat = fruit + "_" + cond
    folds = max(2, min(folds, int(pd.Series(strat).value_counts().min())))
    splits = list(StratifiedKFold(folds, shuffle=True, random_state=SEED).split(X, strat))

    def cv(stage, groups, a):
        accs = []
        for tr, te in splits:
            if stage == 1:
                m = Stage1(group_cols(groups), a).fit(X[tr], fruit[tr], cond[tr])
                accs.append((m.predict(X[te]) == fruit[te]).mean())
            else:
                m = Stage2(group_cols(groups), a).fit(X[tr], fruit[tr], cond[tr])
                accs.append((m.predict(X[te], fruit[te]) == cond[te]).mean())
        return float(np.mean(accs))

    names = list(GROUPS)
    subsets = [c for r in range(1, len(names) + 1) for c in combinations(names, r)]
    rows, best = [], {}
    for stage in (1, 2):
        sc = {g: cv(stage, g, 0.3) for g in subsets}
        for g, v in sc.items():
            rows.append(dict(stage=stage, groups="+".join(g), cv_accuracy=v))
        g_best = max(subsets, key=lambda g: sc[g] - 0.001 * len(g))     # tie -> fewer groups
        a_best = max(alphas, key=lambda a: cv(stage, g_best, a))
        best[stage] = (g_best, a_best, cv(stage, g_best, a_best))
    return best, pd.DataFrame(rows)


def evaluate(clf, df):
    from sklearn.metrics import accuracy_score
    X = df[FEATURE_NAMES].values
    pf, pc = clf.predict(X)
    s2_true = clf.s2.predict(X, df.fruit.values)
    pcat = pf + "_" + pc
    return dict(stage1_acc=accuracy_score(df.fruit, pf),
                stage2_acc_true_fruit=accuracy_score(df.condition, s2_true),
                stage2_acc=accuracy_score(df.condition, pc), overall_acc=accuracy_score(df.category, pcat),
                pred_fruit=pf, pred_cond=pc, pred_cat=pcat)


if __name__ == "__main__":      # CLI:  python core.py dataset/dataset1/train [dataset/dataset1/test]
    import sys
    from sklearn.model_selection import train_test_split
    tr = build_features(sys.argv[1])
    if len(sys.argv) > 2:
        te = build_features(sys.argv[2])
    else:
        tr, te = train_test_split(tr, test_size=0.2, stratify=tr.category, random_state=SEED)
    best, table = tune(tr[FEATURE_NAMES].values, tr.fruit.values, tr.condition.values)
    clf = TwoStageClassifier(best[1][0], best[2][0], best[1][1], best[2][1])
    clf.fit(tr[FEATURE_NAMES].values, tr.fruit.values, tr.condition.values)
    r = evaluate(clf, te)
    print("stage1 uses", best[1], "\nstage2 uses", best[2], f"\nn_train={len(tr)} n_test={len(te)}")
    print({k: round(float(v), 4) for k, v in r.items() if k.endswith("acc") or "acc_" in k})
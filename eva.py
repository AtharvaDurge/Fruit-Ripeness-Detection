import cv2
import numpy as np
import pandas as pd

import core


# ----------------------------------------------------------------------------
# Leave-one-out evaluation
# ----------------------------------------------------------------------------
def loo_predict(proto):
    Z, labels, rel = proto["Z"], proto["labels"], proto["rel"].astype(bool)
    classes = [str(c) for c in proto["classes"]]
    scaled = proto["scaled"]
    files = proto["files"]

    sel, sums, cnt = {}, {}, {}
    for k in classes:                                   # same selection rule as build_prototypes
        m = (labels == k) & rel
        m = m if m.sum() >= 3 else (labels == k)
        sel[k], sums[k], cnt[k] = m, Z[m].sum(axis=0), int(m.sum())

    rows = []
    for i in range(len(labels)):
        k = str(labels[i])
        sc = scaled
        if sel[k][i] and cnt[k] > 1:                    # drop this image from its own class mean
            sc = scaled.copy()
            sc[classes.index(k)] = (sums[k] - Z[i]) / (cnt[k] - 1)
        r = core._classify_z(Z[i], sc, classes, bool(rel[i]))
        rows.append(dict(file=str(files[i]), true=k, pred=f"{r['fruit']}_{r['condition']}",
                         fruit_margin=r["fruit_margin"], ripeness_margin=r["confidence"],
                         reliable=bool(rel[i])))
    df = pd.DataFrame(rows)
    df["true_fruit"] = df["true"].str.rsplit("_", n=1).str[0]
    df["pred_fruit"] = df["pred"].str.rsplit("_", n=1).str[0]
    df["true_cond"] = df["true"].str.rsplit("_", n=1).str[1]
    df["pred_cond"] = df["pred"].str.rsplit("_", n=1).str[1]
    df["correct"] = df["true"] == df["pred"]
    return df


def confusion(df, classes):
    return (pd.crosstab(df["true"], df["pred"])
            .reindex(index=classes, columns=classes, fill_value=0))


def class_report(cm):
    tp = np.diag(cm.values).astype(float)
    prec = tp / np.maximum(cm.values.sum(axis=0), 1)
    rec = tp / np.maximum(cm.values.sum(axis=1), 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    return pd.DataFrame({"precision": prec, "recall": rec, "f1": f1,
                         "support": cm.values.sum(axis=1)}, index=cm.index)


def summary_metrics(df):
    fruit_acc = float((df["true_fruit"] == df["pred_fruit"]).mean())
    ok = df["true_fruit"] == df["pred_fruit"]
    cond_acc = float((df.loc[ok, "true_cond"] == df.loc[ok, "pred_cond"]).mean()) if ok.any() else 0.0
    return {"Overall accuracy (6 classes)": float(df["correct"].mean()),
            "Fruit accuracy (stage 1)": fruit_acc,
            "Ripeness accuracy (given correct fruit)": cond_acc,
            "Accuracy - reliable segmentation": float(df.loc[df["reliable"], "correct"].mean()) if df["reliable"].any() else np.nan,
            "Accuracy - unreliable segmentation": float(df.loc[~df["reliable"], "correct"].mean()) if (~df["reliable"]).any() else np.nan}


# ----------------------------------------------------------------------------
# Dataset EDA
# ----------------------------------------------------------------------------
def class_overview(proto):
    labels, rel, used = proto["labels"], proto["rel"].astype(bool), proto["used"].astype(bool)
    classes = [str(c) for c in proto["classes"]]
    rows = []
    for k in classes:
        m = labels == k
        rows.append(dict(images=int(m.sum()), reliable_seg=int((m & rel).sum()),
                         reliable_pct=float((m & rel).sum() / max(m.sum(), 1)),
                         used_in_average=int((m & used).sum())))
    return pd.DataFrame(rows, index=classes)


def feature_ranking(proto):
    """One-way ANOVA style F-ratio per feature on standardised values:
       between-class variance / within-class variance (higher = separates classes better)."""
    Z, labels = proto["Z"], proto["labels"]
    classes = [str(c) for c in proto["classes"]]
    gm = Z.mean(axis=0)
    between = np.zeros(Z.shape[1])
    within = np.zeros(Z.shape[1])
    for k in classes:
        Zk = Z[labels == k]
        if len(Zk) == 0:
            continue
        between += len(Zk) * (Zk.mean(axis=0) - gm) ** 2
        within += ((Zk - Zk.mean(axis=0)) ** 2).sum(axis=0)
    f = (between / max(len(classes) - 1, 1)) / np.maximum(within / max(len(Z) - len(classes), 1), 1e-12)
    out = pd.DataFrame({"F_ratio": f, "group": [core.GROUP_OF[n] for n in core.FEATURE_NAMES]},
                       index=core.FEATURE_NAMES)
    return out.sort_values("F_ratio", ascending=False)


def feature_stats_by_class(proto):
    """Raw (un-standardised) mean per class for every feature."""
    return pd.DataFrame(proto["raw"].T, index=core.FEATURE_NAMES, columns=[str(c) for c in proto["classes"]])


def correlation(proto, names):
    idx = [core.FEATURE_NAMES.index(n) for n in names]
    return pd.DataFrame(np.corrcoef(proto["Z"][:, idx], rowvar=False), index=names, columns=names)


def group_separation(rank_df):
    return rank_df.groupby("group")["F_ratio"].agg(["mean", "max"]).sort_values("mean", ascending=False)


def image_stats(data_dir, progress=None):
    """Raw image properties (original size, aspect, brightness) per class."""
    df = core.list_images(data_dir)
    rows = []
    for i, r in enumerate(df.itertuples(index=False)):
        img = cv2.imread(r.path)
        if img is not None:
            h, w = img.shape[:2]
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            rows.append(dict(cls=f"{r.true_fruit}_{r.true_cond}", width=w, height=h,
                             aspect=w / h, megapixels=w * h / 1e6,
                             brightness=float(g.mean()), contrast=float(g.std()),
                             sharpness=float(cv2.Laplacian(g, cv2.CV_64F).var())))
        if progress:
            progress((i + 1) / max(len(df), 1))
    return pd.DataFrame(rows)
"""
evaluation.py - the "Evaluation" tab.

Give it a whole batch of labelled images (a folder, or many uploaded files). Every image goes
through the SAME pipeline as the single-image tab (core.extract + core.classify). Then the
predictions are compared with the real labels and we show accuracy, confusion matrices,
precision / recall / F1, a wrong-prediction gallery and a few EDA charts.

How the real label is found (nothing to type in):
  - from the folder names  (apples / banana / oranges  +  raw / unripe / ripe)
  - or from the file name  (e.g. Apple_Ripe_test_12.jpg)
"""
import os
import re

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

import core

THUMB = 128   # size of the small pictures kept for the galleries


# ----------------------------------------------------------------------------
# 1. finding the real label of each image
# ----------------------------------------------------------------------------
def label_from_path(path):
    """Look at the file name first, then the folders above it. Returns (fruit, condition)."""
    parts = os.path.normpath(path).split(os.sep)
    order = [os.path.splitext(parts[-1])[0]] + parts[:-1][::-1]
    fruit = cond = None
    for part in order:
        for tok in re.split(r"[^a-z]+", part.lower()):
            if fruit is None and tok in core.FRUIT_ALIAS:
                fruit = core.FRUIT_ALIAS[tok]
            if cond is None and tok in core.COND_ALIAS:
                cond = core.COND_ALIAS[tok]
    return fruit, cond


def list_folder(folder):
    items, skipped = [], 0
    for root, _, files in os.walk(folder):
        for fn in sorted(files):
            if not fn.lower().endswith(core.IMG_EXT):
                continue
            path = os.path.join(root, fn)
            fruit, cond = label_from_path(path)
            if fruit and cond:
                items.append(dict(name=fn, src=path, fruit=fruit, cond=cond))
            else:
                skipped += 1
    return items, skipped


# ----------------------------------------------------------------------------
# 2. running the pipeline on every image
# ----------------------------------------------------------------------------
def _small(bgr):
    return cv2.cvtColor(cv2.resize(bgr, (THUMB, THUMB)), cv2.COLOR_BGR2RGB)


def run_batch(items, proto, bar):
    rows, imgs = [], {}
    for i, it in enumerate(items):
        row = dict(file=it["name"], true_fruit=it["fruit"], true_cond=it["cond"],
                   true=f"{it['fruit']}_{it['cond']}", pred_fruit=None, pred_cond=None,
                   pred="failed", fruit_margin=np.nan, cond_margin=np.nan,
                   reliable=False, error="")
        img = None
        try:
            src = it["src"]
            img = cv2.imread(src) if isinstance(src, str) else cv2.imdecode(np.frombuffer(src, np.uint8), 1)
            if img is None:
                raise ValueError("could not read the image")
            feats, seg = core.extract(img)
            res = core.classify(feats, proto, seg["reliable"])
            row.update(pred_fruit=res["fruit"], pred_cond=res["condition"],
                       pred=f"{res['fruit']}_{res['condition']}",
                       fruit_margin=res["fruit_margin"], cond_margin=res["confidence"],
                       reliable=bool(seg["reliable"]))
            for k, n in enumerate(core.HUE_HIST_NAMES):
                row[f"hue_{k}"] = feats[n]
            imgs[i] = (_small(seg["raw"]), _small(seg["segmented"]))
        except Exception as e:
            row["error"] = str(e)
            if img is not None:
                imgs[i] = (_small(img), np.zeros((THUMB, THUMB, 3), np.uint8))
        rows.append(row)
        bar.progress((i + 1) / len(items))
    df = pd.DataFrame(rows)
    df["correct"] = df["true"] == df["pred"]
    df["fruit_ok"] = df["true_fruit"] == df["pred_fruit"]
    return df, imgs


# ----------------------------------------------------------------------------
# 3. showing the results
# ----------------------------------------------------------------------------
def _plot_cm(cm, labels, title):
    fig, ax = plt.subplots(figsize=(1.0 + 0.9 * len(labels), 1.0 + 0.8 * len(labels)))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    top = cm.max() if cm.size else 1
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > top / 2 else "black")
    fig.tight_layout()
    return fig


def _performance_tab(df):
    fruit_ok = df[df["fruit_ok"]]
    cond_acc = (fruit_ok["true_cond"] == fruit_ok["pred_cond"]).mean() if len(fruit_ok) else np.nan
    a, b, c, d = st.columns(4)
    a.metric("Images tested", len(df))
    b.metric("Overall accuracy", f"{df['correct'].mean():.1%}")
    c.metric("Fruit accuracy", f"{df['fruit_ok'].mean():.1%}")
    d.metric("Ripeness accuracy", "-" if np.isnan(cond_acc) else f"{cond_acc:.1%}")
    st.caption("Overall = fruit AND ripeness both right. Ripeness accuracy is measured only on "
               "images where the fruit was named correctly.")
    failed = int((df["pred"] == "failed").sum())
    if failed:
        st.caption(f"{failed} image(s) could not be processed - they are counted as wrong.")

    labels = [k for k in core.CLASSES if (df["true"] == k).any() or (df["pred"] == k).any()]
    fruits = [f for f in core.FRUITS if (df["true_fruit"] == f).any() or (df["pred_fruit"] == f).any()]

    st.markdown("#### Confusion matrices")
    st.caption("Rows = what the image really is, columns = what the system said. "
               "Numbers on the diagonal are correct answers.")
    left, right = st.columns(2)
    ok = df[df["pred"] != "failed"]
    left.pyplot(_plot_cm(confusion_matrix(ok["true_fruit"], ok["pred_fruit"], labels=fruits),
                         fruits, "Fruit (Stage 1)"))
    right.pyplot(_plot_cm(confusion_matrix(ok["true"], ok["pred"], labels=labels),
                          labels, "All classes (fruit + ripeness)"))

    st.markdown("#### Precision, recall and F1 per class")
    p, r, f, s = precision_recall_fscore_support(df["true"], df["pred"], labels=labels, zero_division=0)
    table = pd.DataFrame({"images": s, "precision": p, "recall": r, "f1": f}, index=labels)
    table.loc["average"] = [s.sum(), p.mean(), r.mean(), f.mean()]
    st.dataframe(table.round(2))
    st.caption("Precision: when the system says this class, how often is it right? "
               "Recall: of the images that really are this class, how many did it find? "
               "F1: one number that combines the two.")

    st.markdown("#### Does a bad segmentation hurt?")
    g = df.groupby("reliable").agg(images=("correct", "size"), accuracy=("correct", "mean")).round(2)
    g.index = g.index.map({True: "mask looked reliable", False: "mask unreliable / failed"})
    st.dataframe(g)

    st.markdown("#### Is the margin (confidence) meaningful?")
    m = df[df["pred"] != "failed"].groupby("correct")[["fruit_margin", "cond_margin"]].mean().round(2)
    m.index = m.index.map({True: "answer was right", False: "answer was wrong"})
    st.dataframe(m)
    st.caption("If right answers have a clearly higher margin than wrong ones, the margin can be trusted.")


def _wrong_tab(df, imgs):
    wrong = df[~df["correct"]]
    if wrong.empty:
        st.success("No wrong predictions.")
        return
    classes = ["all"] + sorted(wrong["true"].unique())
    pick = st.selectbox("Show wrong predictions whose real class is:", classes)
    if pick != "all":
        wrong = wrong[wrong["true"] == pick]
    n = st.number_input("How many to show", 1, len(wrong), min(12, len(wrong)))
    st.caption(f"{len(wrong)} wrong prediction(s). Left = the image, right = what the segmentation kept.")
    cols = st.columns(4)
    for k, (idx, row) in enumerate(wrong.head(int(n)).iterrows()):
        if idx not in imgs:
            continue
        both = np.hstack(imgs[idx])
        why = f" ({row['error']})" if row["error"] else ""
        cols[k % 4].image(both, caption=f"{row['file']}\nreal: {row['true']}\nsaid: {row['pred']}{why}")


def _eda_tab(df, imgs):
    present = [k for k in core.CLASSES if (df["true"] == k).any()]

    st.markdown("#### Images per class")
    st.bar_chart(df["true"].value_counts().reindex(present))

    st.markdown("#### Example images")
    for k in present:
        st.write(f"**{k}**")
        cols = st.columns(6)
        for col, idx in zip(cols, df[df["true"] == k].index[:6]):
            if idx in imgs:
                col.image(imgs[idx][0])

    st.markdown("#### Average hue histogram per class")
    st.caption("How much of each colour (hue angle in degrees) the fruit pixels contain. "
               "Classes whose lines overlap are the ones the system will mix up.")
    ok = df[df["pred"] != "failed"]
    hue_cols = [f"hue_{k}" for k in range(12)]
    avg = ok.groupby("true")[hue_cols].mean().T
    avg.index = [f"{i * 30:03d}-{i * 30 + 30:03d}" for i in range(12)]
    st.line_chart(avg)


def show(ev, proto):
    df, imgs = ev["df"], ev["imgs"]
    seen = df["file"].isin(set(map(str, proto["files"]))).mean()
    if seen > 0.5:
        st.warning(f"{seen:.0%} of these images were used to BUILD the reference vectors, so the "
                   f"accuracy below looks better than it really is. For an honest result, test on "
                   f"images that were not in the dataset folder (e.g. the test split).")
    if ev["skipped"]:
        st.caption(f"{ev['skipped']} image(s) skipped: could not tell fruit / ripeness from the folder or file name.")
    t1, t2, t3 = st.tabs(["📈 Performance", "❌ Wrong predictions", "🗂️ EDA"])
    with t1:
        _performance_tab(df)
    with t2:
        _wrong_tab(df, imgs)
    with t3:
        _eda_tab(df, imgs)


# ----------------------------------------------------------------------------
# 4. the tab itself (called from app.py)
# ----------------------------------------------------------------------------
def render(proto):
    st.subheader("Evaluate on a whole batch of images")
    if proto is None:
        st.info("Build the reference vectors first (sidebar).")
        return
    st.caption("Every image goes through the same pipeline. The answer is compared with the real label, "
               "which is read from the folder names (apples / raw / ripe ...) or the file name.")

    mode = st.radio("Where are the images?", ["Folder on this computer", "Upload images"], horizontal=True)
    items, skipped, go = None, 0, False

    if mode == "Folder on this computer":
        folder = st.text_input("Folder path (the whole dataset, or just one folder like apples/ripe)",
                               "dataset/dataset1")
        go = st.button("Run evaluation")
        if go:
            if not os.path.isdir(folder):
                st.error("That folder does not exist.")
                go = False
            else:
                items, skipped = list_folder(folder)
    else:
        files = st.file_uploader("Select many images at once (open the folder, press Ctrl+A)",
                                 type=["jpg", "jpeg", "png", "bmp", "webp"], accept_multiple_files=True)
        override = st.selectbox("Real label of these images", ["Auto (read from file name)"] + core.CLASSES)
        go = st.button("Run evaluation")
        if go:
            items = []
            for f in files or []:
                fruit, cond = label_from_path(f.name)
                if override != "Auto (read from file name)":
                    fruit, cond = override.split("_")
                if fruit and cond:
                    items.append(dict(name=f.name, src=f.getvalue(), fruit=fruit, cond=cond))
                else:
                    skipped += 1

    if go:
        if not items:
            st.error("No labelled images found. Folder names or file names must contain the fruit "
                     "(apple / banana / orange) and ripe / raw / unripe.")
        else:
            bar = st.progress(0.0, text=f"Running the pipeline on {len(items)} images ...")
            df, imgs = run_batch(items, proto, bar)
            bar.empty()
            st.session_state["eval"] = dict(df=df, imgs=imgs, skipped=skipped)

    if "eval" in st.session_state:
        show(st.session_state["eval"], proto)
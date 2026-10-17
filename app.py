import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

import core
from core import FEATURE_NAMES, GROUPS, GROUP_OF, FRUITS, CONDS

st.set_page_config(page_title="Fruit Ripeness Detection System", page_icon="🍎", layout="wide")
st.title("🍊 Fruit Ripeness Detection System")
st.caption("Strictly traditional image processing • HSV/Otsu segmentation • colour + texture + shape features • "
           "two-stage reference-vector (Mahalanobis) matching")

# ---------------------------------------------------------------- sidebar
train_dir = st.sidebar.text_input("Training dataset directory", "dataset/dataset1/train")
test_dir = st.sidebar.text_input("Test dataset directory (optional)", "dataset/dataset1/test")
auto_tune = st.sidebar.checkbox("Auto-select feature groups + shrinkage per stage (CV on training set)", True)
manual = st.sidebar.checkbox("Manual S-channel cutoff instead of Otsu (prediction tab)")
cutoff = st.sidebar.slider("S cutoff", 0, 255, 60) if manual else None
if st.sidebar.button("Reload dataset / clear cache"):
    st.cache_data.clear(); st.cache_resource.clear()
    import shutil; shutil.rmtree(".fruit_cache", ignore_errors=True)


@st.cache_data(show_spinner=False)
def load_features(path):
    return core.build_features(path)


@st.cache_resource(show_spinner=False)
def train_model(train_df, tune_on):
    X, f, c = train_df[FEATURE_NAMES].values, train_df.fruit.values, train_df.condition.values
    if tune_on:
        best, table = core.tune(X, f, c)
    else:
        best = {1: (tuple(GROUPS), 0.3, None), 2: (tuple(GROUPS), 0.3, None)}; table = pd.DataFrame()
    clf = core.TwoStageClassifier(best[1][0], best[2][0], best[1][1], best[2][1]).fit(X, f, c)
    return clf, best, table


with st.spinner("Extracting features from dataset (cached after first run)..."):
    df_all = load_features(train_dir)
if df_all.empty:
    st.error(f"No images found in `{train_dir}`. Expected `<fruit>/<raw|ripe>/image` folders "
             "(fruit folders: apples/banana/orange).")
    st.stop()

# train / test: use a separate test folder if present, otherwise a stratified 80/20 hold-out split
if os.path.isdir(test_dir) and not core.list_images(test_dir).empty:
    df_train, df_test = df_all, load_features(test_dir)
    split_note = f"separate test folder `{test_dir}`"
else:
    df_train, df_test = train_test_split(df_all, test_size=0.2, stratify=df_all.category, random_state=core.SEED)
    split_note = "stratified 80/20 hold-out split of the training folder (no test folder found)"

with st.spinner("Building reference vectors..."):
    clf, best, sel_table = train_model(df_train, auto_tune)

tab1, tab2, tab3, tab4 = st.tabs(["🔍 Predict", "📊 EDA", "📈 Performance", "📐 Reference vectors"])

# ---------------------------------------------------------------- predict
with tab1:
    up = st.file_uploader("Upload an apple / banana / orange image", type=["jpg", "jpeg", "png", "bmp", "webp"])
    if up:
        img = cv2.imdecode(np.frombuffer(up.read(), np.uint8), 1)
        try:
            vec, s = core.extract(img, cutoff)
        except Exception as e:
            st.error(f"Could not process the image: {e}"); st.stop()
        r = clf.predict_one(vec)
        (st.success if r["condition"] == "ripe" else st.warning)(
            f"**{r['fruit'].title()} — {r['condition'].upper()}**   "
            f"(ripeness probability {r['prob'][r['condition']]:.0%})")
        st.caption(f"Stage 1 features: {' + '.join(clf.groups1)}  |  Stage 2 features: {' + '.join(clf.groups2)}")
        if not s["reliable"]:
            st.info(f"⚠️ The S-channel mask looks unreliable for this photo (solidity {s['solidity']:.2f}, touches border "
                    f"{s['border_touch']:.0%}) — typical when fruit and leaves are both saturated. The classifier "
                    "therefore leans on whole-image features where the training data says that is more accurate.")
        st.subheader("Image-processing pipeline")
        cols = st.columns(5)
        rgb = lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        for c, (im, cap) in zip(cols, [(rgb(s["raw"]), "1. Resized"), (rgb(s["stretched"]), "2. Blur + contrast stretch"),
                                      (s["s"], "3. Saturation channel"),
                                      (s["mask"], f"4. Binary mask ({s['method']}, thr={s['thr']:.0f})"),
                                      (rgb(s["segmented"]), "5. Segmented (mask × image)")]):
            c.image(im, caption=cap, use_container_width=True, clamp=True)
        a, b = st.columns(2)
        a.subheader("Stage 1 – distance to each fruit's reference vectors")
        a.bar_chart(pd.Series(r["d_fruit"]))
        b.subheader("Stage 2 – distance to ripe / unripe reference")
        b.bar_chart(pd.Series(r["d_cond"]))
        with st.expander("Extracted feature vector"):
            st.dataframe(pd.DataFrame({"value": vec,
                                       "group": [GROUP_OF[f] for f in FEATURE_NAMES]},
                                      index=FEATURE_NAMES))

# ---------------------------------------------------------------- EDA
with tab2:
    st.write(f"Total images (training folder): **{len(df_all)}**")
    cc = df_all.category.value_counts().rename_axis("category").reset_index(name="count")
    c1, c2 = st.columns(2)
    c1.dataframe(cc); c2.bar_chart(cc.set_index("category"))
    st.subheader("Sample segmentation per class")
    fig, ax = plt.subplots(len(df_all.category.unique()), 3, figsize=(7, 2.1 * df_all.category.nunique()))
    for i, (cat, g) in enumerate(df_all.groupby("category")):
        s = core.segment(cv2.imread(g.path.iloc[0]))
        for j, (im, t) in enumerate([(cv2.cvtColor(s["raw"], cv2.COLOR_BGR2RGB), "raw"), (s["mask"], "mask"),
                                     (cv2.cvtColor(s["segmented"], cv2.COLOR_BGR2RGB), "segmented")]):
            ax[i, j].imshow(im, cmap="gray" if im.ndim == 2 else None); ax[i, j].axis("off")
            if i == 0: ax[i, j].set_title(t)
        ax[i, 0].text(-8, 128, cat, ha="right", va="center", fontsize=8)
    st.pyplot(fig)
    grp = st.selectbox("Feature group", list(GROUPS))
    fig, ax = plt.subplots(figsize=(11, 8))
    sns.heatmap(df_all[GROUPS[grp]].corr(), cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    st.pyplot(fig)
    feat = st.selectbox("Feature distribution by class", FEATURE_NAMES)
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.boxplot(x="category", y=feat, data=df_all, ax=ax); plt.xticks(rotation=45)
    st.pyplot(fig)
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    Z = PCA(2).fit_transform(StandardScaler().fit_transform(df_all[FEATURE_NAMES]))
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.scatterplot(x=Z[:, 0], y=Z[:, 1], hue=df_all.category, s=18, ax=ax); ax.set_title("PCA of feature vectors")
    st.pyplot(fig)

# ---------------------------------------------------------------- performance
with tab3:
    st.write(f"Reference vectors built from **{len(df_train)}** training images • evaluated on **{len(df_test)}** unseen "
             f"images ({split_note})")
    st.write(f"Stage 1 uses **{' + '.join(clf.groups1)}** (α={clf.s1.alpha}); "
             f"Stage 2 uses **{' + '.join(clf.groups2)}** (α={clf.s2.alpha})")
    rt = core.evaluate(clf, df_test)
    m = st.columns(4)
    m[0].metric("Stage 1 – fruit", f"{rt['stage1_acc']:.2%}")
    m[1].metric("Stage 2 – ripeness (true fruit)", f"{rt['stage2_acc_true_fruit']:.2%}")
    m[2].metric("Stage 2 – ripeness (pipeline)", f"{rt['stage2_acc']:.2%}")
    m[3].metric("Overall 6-class", f"{rt['overall_acc']:.2%}")
    for title, yt, yp, labs in [("Stage 1 – fruit", df_test.fruit, rt["pred_fruit"], FRUITS),
                                ("Stage 2 – ripe/unripe", df_test.condition, rt["pred_cond"], CONDS),
                                ("Overall 6-class", df_test.category, rt["pred_cat"],
                                 [f"{f}_{c}" for f in FRUITS for c in CONDS])]:
        st.subheader(title)
        fig, ax = plt.subplots(figsize=(1 + 0.9 * len(labs), 1 + 0.8 * len(labs)))
        sns.heatmap(confusion_matrix(yt, yp, labels=labs), annot=True, fmt="d", cmap="Blues",
                    xticklabels=labs, yticklabels=labs, ax=ax)
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        st.pyplot(fig)
        st.text(classification_report(yt, yp, zero_division=0))
    if len(sel_table):
        st.subheader("Feature-group selection (5-fold CV on the training data only)")
        piv = sel_table.pivot(index="groups", columns="stage", values="cv_accuracy")
        piv.columns = ["stage 1 (fruit)", "stage 2 (ripeness)"]
        st.dataframe(piv.sort_values("stage 1 (fruit)", ascending=False).style.format("{:.2%}"))

# ---------------------------------------------------------------- reference vectors
with tab4:
    st.write("The 6 reference vectors (class mean feature vectors):")
    st.dataframe(clf.reference_table())
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
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] { overflow-x: hidden !important; }
.block-container { max-width: 1150px !important; margin: 0 auto; padding: 2rem 2rem 3rem 2rem !important; }
h1 { text-align: center; padding-bottom: .2rem; }
[data-testid="stMetric"] { background: rgba(128,128,128,.08); border: 1px solid rgba(128,128,128,.25);
                           border-radius: 10px; padding: 12px 16px; }
[data-testid="stImage"] img { border-radius: 8px; }
[data-testid="stTabs"] button { font-size: 1rem; }
.subtitle { text-align: center; opacity: .7; margin-bottom: 1.2rem; font-size: .95rem; }
</style>
""", unsafe_allow_html=True)


def show_fig(fig):
    """Show a matplotlib figure at its natural size (not stretched to the page width)."""
    try:
        st.pyplot(fig, width="content")                    # newer Streamlit
    except TypeError:
        try:
            st.pyplot(fig, use_container_width=False)      # older Streamlit
        except TypeError:
            st.pyplot(fig)
    plt.close(fig)


st.title("🍊 Fruit Ripeness Detection System")
st.markdown('<div class="subtitle">Strictly traditional image processing • HSV/Otsu segmentation • '
            'colour + texture + shape features • two-stage reference-vector (Mahalanobis) matching</div>',
            unsafe_allow_html=True)

# ---------------------------------------------------------------- sidebar
train_dir = st.sidebar.text_input("Training dataset directory", "dataset/dataset1/train")
test_dir = st.sidebar.text_input("Test dataset directory (optional)", "dataset/dataset1/test")
auto_tune = st.sidebar.checkbox("Auto-select feature groups + shrinkage per stage (CV on training set)", True)
if st.sidebar.button("Reload dataset / clear cache"):
    st.cache_data.clear(); st.cache_resource.clear()
    import shutil; shutil.rmtree(".fruit_cache", ignore_errors=True)


@st.cache_data(show_spinner=False)
def load_features(path):
    return core.build_features(path)


@st.cache_data(show_spinner=False)
def best_sample_paths(df, n_try=40):
    """For every class, look at up to n_try images and return the one whose mask is cleanest:
    compact (high solidity), not touching the border, a sensible size, and lying on real image edges."""
    out = {}
    for cat, g in df.groupby("category"):
        best, best_score = g.path.iloc[0], -1e9
        for p in g.path.sample(min(n_try, len(g)), random_state=0):
            img = cv2.imread(p)
            if img is None:
                continue
            s = core.segment(img)
            gray = cv2.cvtColor(s["raw"], cv2.COLOR_BGR2GRAY).astype(np.float32)
            grad = np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
            edge = cv2.morphologyEx(s["mask"], cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
            align = grad[edge].mean() / (grad.mean() + 1e-6) if edge.any() else 0.0
            size_ok = 0.08 < s["area_frac"] < 0.65
            score = (2.0 if s["reliable"] else 0.0) + (0.5 if size_ok else 0.0) + s["solidity"] \
                    + min(align, 3.0) / 3.0 - s["border_touch"]
            if score > best_score:
                best, best_score = p, score
        out[cat] = best
    return out


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
            vec, s = core.extract(img)
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
    st.caption("For each class, the cleanest segmentation among up to 40 images is shown (auto-selected).")
    best_paths = best_sample_paths(df_all)
    fig, ax = plt.subplots(len(df_all.category.unique()), 3, figsize=(7, 2.1 * df_all.category.nunique()))
    for i, (cat, g) in enumerate(df_all.groupby("category")):
        s = core.segment(cv2.imread(best_paths[cat]))
        for j, (im, t) in enumerate([(cv2.cvtColor(s["raw"], cv2.COLOR_BGR2RGB), "raw"), (s["mask"], "mask"),
                                     (cv2.cvtColor(s["segmented"], cv2.COLOR_BGR2RGB), "segmented")]):
            ax[i, j].imshow(im, cmap="gray" if im.ndim == 2 else None); ax[i, j].axis("off")
            if i == 0: ax[i, j].set_title(t)
        ax[i, 0].text(-8, 128, cat, ha="right", va="center", fontsize=8)
    show_fig(fig)
    grp = st.selectbox("Feature group", list(GROUPS))
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(df_all[GROUPS[grp]].corr(), cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    show_fig(fig)
    feat = st.selectbox("Feature distribution by class", FEATURE_NAMES)
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.boxplot(x="category", y=feat, data=df_all, ax=ax); plt.xticks(rotation=45)
    show_fig(fig)
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    Z = PCA(2).fit_transform(StandardScaler().fit_transform(df_all[FEATURE_NAMES]))
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.scatterplot(x=Z[:, 0], y=Z[:, 1], hue=df_all.category, s=18, ax=ax); ax.set_title("PCA of feature vectors")
    show_fig(fig)

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
        show_fig(fig)
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
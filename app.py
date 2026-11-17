import altair as alt
import cv2
import numpy as np
import pandas as pd
import streamlit as st

import core
import eva

st.set_page_config(page_title="Fruit Ripeness Detection", page_icon="🍎", layout="wide")
st.markdown("""
<style>
.block-container { max-width: 1150px !important; margin: 0 auto; padding: 2rem 2rem 3rem 2rem !important; }
h1 { text-align: center; padding-bottom: .2rem; }
</style>
""", unsafe_allow_html=True)

st.title("🍊 Fruit Ripeness Detection System")
st.caption("Pure classical image processing - no AI / ML.")


def rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def heatmap(long_df, x, y, val, fmt=".0f", scheme="blues", size=420, text=True):
    base = alt.Chart(long_df).encode(x=alt.X(f"{x}:N", sort=None), y=alt.Y(f"{y}:N", sort=None))
    rect = base.mark_rect().encode(color=alt.Color(f"{val}:Q", scale=alt.Scale(scheme=scheme)))
    if not text:
        return rect.properties(height=size)
    txt = base.mark_text(fontSize=12).encode(text=alt.Text(f"{val}:Q", format=fmt),
                                             color=alt.value("black"))
    return (rect + txt).properties(height=size)


# ---------------------------------------------------------------- sidebar: reference images
st.sidebar.header("Dataset (reference images)")
data_dir = st.sidebar.text_input("Dataset folder (all images, no split)", "dataset/dataset1")

if "proto" not in st.session_state:
    st.session_state.proto = core.load_prototypes()

if st.sidebar.button("Build / rebuild reference vectors"):
    bar = st.sidebar.progress(0.0)
    try:
        st.session_state.proto = core.build_prototypes(data_dir, progress=bar.progress)
        st.session_state.pop("loo", None)
        st.session_state.pop("imgstats", None)
        st.sidebar.success("Reference vectors built and saved to prototypes.npz")
    except Exception as e:
        st.sidebar.error(str(e))

proto = st.session_state.proto
if proto is None:
    st.sidebar.warning("No (up-to-date) reference vectors. Click the button above first.")

tab1, tab2, tab3 = st.tabs(["🔍 Predict + pipeline", "🧮 Reference vectors", "📊 EDA & metrics"])

# ---------------------------------------------------------------- predict
with tab1:
    up = st.file_uploader("Upload an apple / banana / orange image", type=["jpg", "jpeg", "png", "bmp", "webp"])
    if up and proto is None:
        st.warning("Build the reference vectors first (sidebar).")
    if up and proto is not None:
        img = cv2.imdecode(np.frombuffer(up.read(), np.uint8), 1)
        if img is None:
            st.error("Could not read this image file.")
            st.stop()
        try:
            feats, seg = core.extract(img)
        except Exception as e:
            st.error(f"Could not process the image: {e}")
            st.stop()
        res = core.classify(feats, proto, seg["reliable"])

        (st.success if res["condition"] == "ripe" else st.warning)(
            f"**{res['fruit'].title()} - {res['condition'].upper()}**   "
            f"(fruit margin {res['fruit_margin']:.0%}, ripeness margin {res['confidence']:.0%})")
        if not seg["reliable"]:
            st.info(f"⚠️ Segmentation looks unreliable (solidity {seg['solidity']:.2f}, "
                    f"touches border {seg['border_touch']:.0%}). Shape features were ignored; "
                    f"results may still be wrong.")

        st.subheader("Pipeline")
        stages = [("1. Resized 256x256", rgb(seg["raw"])),
                  ("2. Gaussian blur", rgb(seg["blur"])),
                  ("3. Contrast stretch", rgb(seg["stretched"])),
                  ("4. S-channel (HSV)", seg["s"]),
                  (f"5. Best candidate mask: {seg['method']}", seg["cand_raw"]),
                  ("6. Cleaned + hull-filled mask", seg["mask"]),
                  ("7. Core mask (eroded 2px)", seg["core"]),
                  ("8. Segmented fruit", rgb(seg["segmented"]))]
        for row in range(2):
            cols = st.columns(4)
            for col, (title, im) in zip(cols, stages[row * 4:(row + 1) * 4]):
                col.image(im, caption=title)
        st.caption(f"Background detected as **{'plain' if seg['uniform_bg'] else 'cluttered'}**.")

        a, b, c = st.columns(3)
        a.subheader("Stage 1: distance to class vectors (fruit)")
        a.bar_chart(pd.Series(res["distances"]))
        b.subheader(f"Stage 2: ripeness of {res['fruit']}")
        b.bar_chart(pd.Series(res["cond_distances"]))
        c.subheader("Hue histogram (fruit pixels only)")
        c.bar_chart(pd.Series([feats[n] for n in core.HUE_HIST_NAMES],
                              index=[f"{i * 30:03d}-{i * 30 + 30:03d}" for i in range(12)]))

        with st.expander("All 108 extracted features"):
            st.dataframe(pd.DataFrame({"value": [feats[n] for n in core.FEATURE_NAMES],
                                       "group": [core.GROUP_OF[n] for n in core.FEATURE_NAMES]},
                                      index=core.FEATURE_NAMES))

# ---------------------------------------------------------------- reference vectors
with tab2:
    if proto is None:
        st.info("Not built yet.")
    else:
        st.subheader("Images averaged per class")
        st.dataframe(pd.Series(proto["counts"], index=proto["classes"], name="images"))
        st.subheader("Standardised reference vectors (6 x 108)")
        st.dataframe(pd.DataFrame(proto["scaled"], index=proto["classes"], columns=core.FEATURE_NAMES))

# ---------------------------------------------------------------- EDA & metrics
with tab3:
    if proto is None:
        st.info("Build the reference vectors first (sidebar).")
    else:
        classes = [str(c) for c in proto["classes"]]

        # ---- 1. dataset overview
        st.header("1. Dataset overview")
        ov = eva.class_overview(proto)
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total images used", int(len(proto["labels"])))
        k2.metric("Skipped (unreadable / failed)", int(proto["skipped"]))
        k3.metric("Reliable segmentation", f"{ov['reliable_seg'].sum() / max(ov['images'].sum(), 1):.0%}")
        k4.metric("Classes", len(classes))
        c1, c2 = st.columns(2)
        c1.subheader("Class balance")
        c1.bar_chart(ov["images"])
        c2.subheader("Segmentation reliability per class")
        c2.bar_chart(ov["reliable_pct"])
        st.dataframe(ov.style.format({"reliable_pct": "{:.1%}"}), use_container_width=True)

        st.subheader("Raw image properties")
        if st.button("Scan dataset images (size / brightness / sharpness)"):
            pb = st.progress(0.0)
            st.session_state.imgstats = eva.image_stats(data_dir, progress=pb.progress)
        ist = st.session_state.get("imgstats")
        if ist is not None and len(ist):
            d1, d2 = st.columns(2)
            d1.caption("Mean brightness per class")
            d1.bar_chart(ist.groupby("cls")["brightness"].mean())
            d2.caption("Mean sharpness (Laplacian variance) per class")
            d2.bar_chart(ist.groupby("cls")["sharpness"].mean())
            st.dataframe(ist.groupby("cls")[["width", "height", "aspect", "megapixels",
                                              "brightness", "contrast", "sharpness"]].describe().T,
                         use_container_width=True)

        # ---- 2. feature analysis
        st.header("2. Feature analysis")
        rank = eva.feature_ranking(proto)
        g1, g2 = st.columns(2)
        g1.subheader("Top 20 class-separating features (F-ratio)")
        g1.bar_chart(rank["F_ratio"].head(20))
        g2.subheader("Mean separation per feature group")
        g2.bar_chart(eva.group_separation(rank)["mean"])
        with st.expander("Full feature ranking (108)"):
            st.dataframe(rank, use_container_width=True)

        st.subheader("Class means of any feature")
        fs = eva.feature_stats_by_class(proto)
        pick = st.multiselect("Features", core.FEATURE_NAMES, default=list(rank.index[:5]))
        if pick:
            st.bar_chart(fs.loc[pick].T)
            st.dataframe(fs.loc[pick], use_container_width=True)

        st.subheader("Correlation of the top features")
        n_top = st.slider("Number of top features", 8, 40, 20)
        cor = eva.correlation(proto, list(rank.index[:n_top]))
        cl = cor.stack().rename("r").reset_index()
        cl.columns = ["f1", "f2", "r"]
        st.altair_chart(heatmap(cl, "f2", "f1", "r", fmt=".1f", scheme="redblue", size=520, text=False),
                        use_container_width=True)

        st.subheader("Hue histogram per class (mean)")
        hue_cols = [f"{i * 30:03d}-{i * 30 + 30:03d}" for i in range(12)]
        hh = pd.DataFrame(fs.loc[core.HUE_HIST_NAMES].values, index=hue_cols, columns=classes)
        st.bar_chart(hh)

        # ---- 3. evaluation
        st.header("3. Performance (leave-one-out, no train/test split)")
        st.caption("Each reference image is classified against class vectors rebuilt without it. "
                   "Nothing is trained - this is purely template matching.")
        if "loo" not in st.session_state:
            with st.spinner("Evaluating..."):
                st.session_state.loo = eva.loo_predict(proto)
        loo = st.session_state.loo
        sm = eva.summary_metrics(loo)
        cols = st.columns(len(sm))
        for col, (name, v) in zip(cols, sm.items()):
            col.metric(name, "n/a" if pd.isna(v) else f"{v:.1%}")

        cm = eva.confusion(loo, classes)
        rep = eva.class_report(cm)
        m1, m2 = st.columns(2)
        m1.subheader("Confusion matrix (6 classes)")
        cml = cm.stack().rename("count").reset_index()
        cml.columns = ["true", "pred", "count"]
        m1.altair_chart(heatmap(cml, "pred", "true", "count"), use_container_width=True)
        m2.subheader("Per-class precision / recall / F1")
        m2.dataframe(rep.style.format({"precision": "{:.2%}", "recall": "{:.2%}", "f1": "{:.2%}"}),
                     use_container_width=True)
        m2.bar_chart(rep[["precision", "recall", "f1"]])

        fr = [f for f in core.FRUITS]
        cmf = (pd.crosstab(loo["true_fruit"], loo["pred_fruit"])
               .reindex(index=fr, columns=fr, fill_value=0))
        n1, n2 = st.columns(2)
        n1.subheader("Stage 1 confusion (fruit)")
        fl = cmf.stack().rename("count").reset_index()
        fl.columns = ["true", "pred", "count"]
        n1.altair_chart(heatmap(fl, "pred", "true", "count", size=300), use_container_width=True)
        n2.subheader("Accuracy by class")
        n2.bar_chart(loo.groupby("true")["correct"].mean())

        st.subheader("Margin (confidence) distribution")
        q1, q2 = st.columns(2)
        q1.caption("Fruit margin: correct vs wrong")
        q1.bar_chart(loo.groupby("correct")["fruit_margin"].mean())
        q2.caption("Ripeness margin: correct vs wrong")
        q2.bar_chart(loo.groupby("correct")["ripeness_margin"].mean())

        st.subheader("Misclassified images")
        wrong = loo[~loo["correct"]][["file", "true", "pred", "fruit_margin", "ripeness_margin", "reliable"]]
        st.dataframe(wrong, use_container_width=True)
        st.download_button("Download all LOO predictions (CSV)", loo.to_csv(index=False).encode(),
                           "loo_predictions.csv", "text/csv")
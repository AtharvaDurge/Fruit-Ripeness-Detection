import cv2
import numpy as np
import pandas as pd
import streamlit as st

import core

st.set_page_config(page_title="Fruit Ripeness Detection", page_icon="🍎", layout="wide")
st.markdown("""
<style>
.block-container { max-width: 1150px !important; margin: 0 auto; padding: 2rem 2rem 3rem 2rem !important; }
h1 { text-align: center; padding-bottom: .2rem; }
</style>
""", unsafe_allow_html=True)

st.title("🍊 Fruit Ripeness Detection System")


def rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------- sidebar: reference images
st.sidebar.header("Reference (train) images")
train = st.sidebar.text_input("Train folder", "dataset/dataset1/train")

if "proto" not in st.session_state:
    st.session_state.proto = core.load_prototypes()

if st.sidebar.button("Build / rebuild reference vectors"):
    bar = st.sidebar.progress(0.0)
    try:
        st.session_state.proto = core.build_prototypes(train, progress=bar.progress)
        st.sidebar.success("Reference vectors built and saved to prototypes.npz")
    except Exception as e:
        st.sidebar.error(str(e))

proto = st.session_state.proto
if proto is None:
    st.sidebar.warning("No reference vectors yet. Click the button above first.")

tab1, tab2 = st.tabs(["🔍 Predict + pipeline", "🧮 Reference vectors"])

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
        res = core.classify(feats, proto)

        (st.success if res["condition"] == "ripe" else st.warning)(
            f"**{res['fruit'].title()} — {res['condition'].upper()}**   (margin {res['confidence']:.0%})")
        if not seg["reliable"]:
            st.info(f"⚠️ Segmentation looks unreliable (solidity {seg['solidity']:.2f}, "
                    f"touches border {seg['border_touch']:.0%}). Results may be wrong.")

        st.subheader("Pipeline")
        stages = [("1. Resized 256x256", rgb(seg["raw"])),
                  ("2. Gaussian blur", rgb(seg["blur"])),
                  ("3. Contrast stretch", rgb(seg["stretched"])),
                  ("4. S-channel (HSV)", seg["s"]),
                  (f"5. Otsu (T={seg['thr']:.0f})", seg["otsu"]),
                  ("6. Cleaned mask", seg["mask"]),
                  ("7. Core mask (eroded 2px)", seg["core"]),
                  ("8. Segmented fruit", rgb(seg["segmented"]))]
        for row in range(2):
            cols = st.columns(4)
            for col, (title, im) in zip(cols, stages[row * 4:(row + 1) * 4]):
                col.image(im, caption=title)

        a, b = st.columns(2)
        a.subheader("Distance to each class vector")
        a.bar_chart(pd.Series(res["distances"]))
        b.subheader("Hue histogram (12 bins)")
        b.bar_chart(pd.Series([feats[n] for n in core.HUE_HIST_NAMES],
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
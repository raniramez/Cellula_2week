import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime
from classifier import ToxicClassifier
from imagecaption import caption_image, preload
import json

st.set_page_config(page_title="Caption + Classify", layout="wide")

DB_PATH = Path("db.csv")

if "caption_model_ready" not in st.session_state:
    with st.sidebar:
        with st.spinner("Loading captioning model..."):
            preload()
    st.session_state["caption_model_ready"] = True

@st.cache_resource
def load_clf():
    return ToxicClassifier()

def append_row(kind, text, label, score):
    row = pd.DataFrame([{
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "type": kind,                 # "text" or "caption"
        "content": text,
        "label": label,
        "score": score
    }])
    if DB_PATH.exists():
        row.to_csv(DB_PATH, mode="a", header=False, index=False)
    else:
        row.to_csv(DB_PATH, index=False)

st.title("🖼️→📝 Image Captioning + 🧪 Toxic Classifier")
tab1, tab2, tab3 = st.tabs(["Run", "Database", "Metrics"])

with tab1:
    mode = st.radio("Input type", ["Text", "Image → Caption"], horizontal=True)
    clf = load_clf()

    if mode == "Text":
        with st.form("text_form", clear_on_submit=False):
            text = st.text_area("Enter text:", height=120)
            submitted = st.form_submit_button("Submit")
            if submitted:
                if text.strip():
                    label, score = clf.classify(text.strip())
                    st.success(f"Predicted label: **{label}**  |  score: {score:.3f}")
                    append_row("text", text.strip(), label, score)
                    st.toast("Saved to db.csv", icon="💾")
                else:
                    st.warning("Please enter some text.")
    else:
        with st.form("image_form", clear_on_submit=False):
            up = st.file_uploader("Upload image", type=["png","jpg","jpeg","webp"])
            submitted = st.form_submit_button("Submit")
            if submitted:
                if up is None:
                    st.warning("Please upload an image.")
                else:
                    suffix = up.name.split(".")[-1].lower()
                    tmp = Path(f"tmp_upload.{suffix}")
                    tmp.write_bytes(up.read())
                    cap = caption_image(str(tmp), "Describe this image in detail.")
                    st.info(f"Caption: {cap}")
                    label, score = clf.classify(cap)
                    st.success(f"Predicted label: **{label}**  |  score: {score:.3f}")
                    append_row("caption", cap, label, score)
                    st.toast("Saved to db.csv", icon="💾")
                    try:
                        tmp.unlink(missing_ok=True)
                    except Exception:
                        pass

with tab2:
    st.subheader("Database")
    if DB_PATH.exists():
        df = pd.read_csv(DB_PATH)
        st.dataframe(df, use_container_width=True)
        st.download_button("Download CSV",
                           df.to_csv(index=False).encode("utf-8"),
                           "db.csv", "text/csv")
    else:
        st.warning("Database is empty.")

with tab3:
    mpath = Path("./outputs_distilbert_lora/metrics.json")
    if mpath.exists():
        m = json.load(open(mpath))
        st.metric("Val Accuracy", f"{m['val'].get('eval_accuracy', 0):.3f}")
        st.metric("Val F1",       f"{m['val'].get('eval_f1', 0):.3f}")
        st.metric("Test Accuracy",f"{m['test'].get('eval_accuracy', 0):.3f}")
        st.metric("Test F1",      f"{m['test'].get('eval_f1', 0):.3f}")
    else:
        st.info("Train first to generate metrics.json")

"""Streamlit demo: Hinglish sentiment analyzer with word-level highlighting."""
import html
import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from data import LABELS  # noqa: E402
from predict import available_models, explain, predict  # noqa: E402

EMOJI = {"positive": "😊", "neutral": "😐", "negative": "😞"}
EXAMPLES = [
    "Movie mast thi yaar, but ending bekar thi",
    "Aaj ka match toh ekdum zabardast tha, maza aa gaya!",
    "Ye phone ka battery bahut kharab hai, paisa barbaad",
    "Kal office jaana hai, meeting 10 baje hai",
    "Bhai tune toh dil jeet liya, thank you so much",
]

st.set_page_config(page_title="Hinglish Sentiment Analyzer", page_icon="💬", layout="centered")
st.title("💬 Hinglish Sentiment Analyzer")
st.caption("Sentiment of Hindi-English code-mixed text written in Roman script: positive, neutral or negative.")


@st.cache_resource(show_spinner="Loading model... (the first load can take a minute)")
def load(name):
    return available_models()[name]()


def read_json(path):
    p = ROOT / "models" / path
    return json.loads(p.read_text()) if p.exists() else None


models = available_models()
if not models:
    st.error("No trained model found in `models/`. Run `python src/train_baseline.py` first.")
    st.stop()

test = read_json("test_metrics.json")

with st.sidebar:
    st.header("Settings")
    choice = st.selectbox("Model", list(models), index=len(models) - 1)
    if test:
        st.header("Test-set scores")
        for label, key in [("TF-IDF + Logistic Regression", "baseline"), ("Fine-tuned MiniLM", "transformer")]:
            m = test[key]
            st.metric(label, f"{m['accuracy']*100:.1f}% accuracy", f"macro-F1 {m['macro_f1']:.3f}", delta_color="off")
    st.caption("Dataset: SentiMix (SemEval-2020 Task 9, Hindi-English). 14k train / 3k dev / 3k test tweets.")

if "text" not in st.session_state:
    st.session_state.text = EXAMPLES[0]

st.write("**Try an example:**")
cols = st.columns(len(EXAMPLES))
for i, (c, ex) in enumerate(zip(cols, EXAMPLES)):
    if c.button(f"#{i+1}", help=ex, use_container_width=True):
        st.session_state.text = ex

text = st.text_area("Type a Hinglish sentence", key="text", height=100)

if st.button("Analyze", type="primary") and text.strip():
    model = load(choice)
    probs, _ = predict(model, text)
    top = max(probs, key=probs.get)

    st.subheader(f"{EMOJI[top]} {top.capitalize()}  ({probs[top]*100:.1f}%)")
    for l in LABELS:
        st.progress(probs[l], text=f"{l}: {probs[l]*100:.1f}%")

    pairs, _ = explain(model, text)
    if pairs:
        peak = max(abs(s) for _, s in pairs) or 1.0
        spans = []
        for w, s in pairs:
            a = min(abs(s) / peak, 1.0) * 0.75
            rgb = "46,158,91" if s > 0 else "214,69,69"
            bg = f"rgba({rgb},{a:.2f})" if abs(s) > 0.005 else "transparent"
            spans.append(
                f'<span title="{s:+.3f}" style="background:{bg};padding:2px 5px;margin:1px;'
                f'border-radius:4px;display:inline-block">{html.escape(w)}</span>'
            )
        st.markdown("**Which words influenced the prediction**")
        st.markdown(" ".join(spans), unsafe_allow_html=True)
        st.caption(f"Green = pushes towards '{top}', red = pushes against it. Hover a word to see its score.")

    if len(models) > 1:
        with st.expander("Compare both models"):
            for name in models:
                p, _ = predict(load(name), text)
                best = max(p, key=p.get)
                st.write(f"**{name}** → {EMOJI[best]} {best} ({p[best]*100:.1f}%)")

"""
app.py — Disease Risk Predictor (Streamlit)

Runs the *actual* trained scikit-learn models (StandardScaler + LogisticRegression,
pickled in models/*.pkl) — no reimplementation in JS. Run:

    streamlit run app.py
"""
import pickle
import json
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from content import DISEASE_CONTENT, DISCLAIMER, risk_label, risk_color

BASE = Path(__file__).parent
MODELS_DIR = BASE / "models"

st.set_page_config(page_title="Disease Risk Predictor", layout="wide")

INK = "#152238"
SLATE = "#6B7280"
TEAL = "#3A6B6B"
PAPER = "#FAFAF8"
LINE = "#DEDCD3"

st.markdown(
    f"""
    <style>
    .stApp {{ background-color: {PAPER}; }}
    h1, h2, h3 {{ font-family: Georgia, 'Times New Roman', serif; color: {INK}; }}
    .block-container {{ padding-top: 2rem; max-width: 1100px; }}
    div[data-testid="stMetricValue"] {{ color: {INK}; }}
    .precaution-item {{ font-size: 0.9rem; color: {INK}; margin-bottom: 0.35rem; }}
    .model-note {{ font-size: 0.8rem; color: {SLATE}; }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_models():
    registry = json.load(open(MODELS_DIR / "registry.json"))
    models = {}
    for name in registry:
        with open(MODELS_DIR / f"{name}.pkl", "rb") as f:
            models[name] = pickle.load(f)
    return registry, models


registry, models = load_models()

st.title("Risk Assessment")
st.caption("Trained on real clinical data — not a diagnosis or medical advice.")

disease = st.radio(
    "Disease",
    options=list(registry.keys()),
    format_func=lambda k: DISEASE_CONTENT[k]["title"],
    horizontal=True,
    label_visibility="collapsed",
)

meta = registry[disease]
content = DISEASE_CONTENT[disease]
bundle = models[disease]
pipeline = bundle["pipeline"]
feature_order = bundle["feature_order"]

st.markdown(f"*{meta['description']}*")

left, right = st.columns([1, 2], gap="large")

inputs = {}
with left:
    st.subheader("Patient values")
    for key in feature_order:
        f = meta["features"][key]
        if f["kind"] == "toggle":
            inputs[key] = int(st.toggle(f["label"], value=False, key=f"{disease}_{key}"))
        else:
            inputs[key] = st.slider(
                f"{f['label']}" + (f" ({f['unit']})" if f["unit"] else ""),
                min_value=float(f["min"]),
                max_value=float(f["max"]),
                value=float(f["min"] + (f["max"] - f["min"]) / 3),
                step=float(f["step"]),
                key=f"{disease}_{key}",
            )

# Real model inference — not a JS re-implementation
X = pd.DataFrame([[inputs[k] for k in feature_order]], columns=feature_order)
proba = float(pipeline.predict_proba(X)[0, 1])
pct = min(98, max(1, proba * 100))
color = risk_color(pct)
label = risk_label(pct)

# contribution breakdown using the underlying logistic regression coefficients
scaler = pipeline.named_steps["scaler"]
clf = pipeline.named_steps["clf"]
z_contribs = clf.coef_[0] * scaler.transform(X)[0]
contrib_df = pd.DataFrame({
    "factor": [meta["features"][k]["label"] for k in feature_order],
    "contribution": z_contribs,
}).sort_values("contribution", ascending=False)
contrib_df = contrib_df[contrib_df["contribution"] > 0.05].head(6)

with right:
    st.subheader("Result")
    gauge_col, text_col = st.columns([1, 1.4])

    with gauge_col:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pct,
            number={"suffix": "%", "font": {"size": 40, "color": INK}},
            gauge={
                "axis": {"range": [0, 100], "visible": False},
                "bar": {"color": color, "thickness": 0.3},
                "bgcolor": "#EFEDE4",
                "borderwidth": 0,
            },
        ))
        fig.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor=PAPER)
        st.plotly_chart(fig, use_container_width=True)

    with text_col:
        st.markdown(
            f"<span style='background:{color};color:{PAPER};padding:4px 10px;font-size:0.8rem;'>"
            f"{label} risk</span>",
            unsafe_allow_html=True,
        )
        st.write(meta["description"])
        st.markdown(
            f"<div class='model-note'>Model: logistic regression · "
            f"{meta['n_samples']:,} patient records · {meta['source']} · "
            f"test AUC {meta['auc']} · accuracy {meta['accuracy']*100:.0f}%</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("**Contributing factors**")
    if len(contrib_df) == 0:
        st.write("No significant risk-increasing factors detected.")
    else:
        bar_fig = go.Figure(go.Bar(
            x=contrib_df["contribution"],
            y=contrib_df["factor"],
            orientation="h",
            marker_color=color,
        ))
        bar_fig.update_layout(
            height=max(120, 34 * len(contrib_df)),
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor=PAPER,
            plot_bgcolor=PAPER,
            xaxis=dict(visible=False),
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(bar_fig, use_container_width=True)

    if disease == "stroke" and "emergency" in content:
        st.error(f"**Know the emergency signs — act FAST.** {content['emergency']}")

    adv_col1, adv_col2 = st.columns(2)
    with adv_col1:
        st.markdown("**Precautions**")
        for item in content["precautions"]:
            st.markdown(f"<div class='precaution-item'>· {item}</div>", unsafe_allow_html=True)
    with adv_col2:
        st.markdown("**General advice**")
        for item in content["advice"]:
            st.markdown(f"<div class='precaution-item'>· {item}</div>", unsafe_allow_html=True)

st.markdown("---")
st.caption(DISCLAIMER)

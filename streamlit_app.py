import json, io
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from catboost import CatBoostRegressor

from src.features.featurize import featurize_trip
from src.models.pricing import price_from_risk

MODEL_PATH = "models/gbm_risk.cbm"
FEATS_PATH = "models/gbm_risk_features.json"

@st.cache_resource
def load_model():
    model = CatBoostRegressor(); model.load_model(MODEL_PATH)
    feat_names = json.loads(Path(FEATS_PATH).read_text())
    return model, feat_names

st.set_page_config(page_title="Telematics UBI Demo", layout="wide")
st.title("Telematics UBI  Risk & Pricing Demo")

model, feat_names = load_model()

uploaded = st.file_uploader("Upload trip JSONL (one JSON per line)", type=["jsonl","txt"])
colL, colR = st.columns([1,1])
with colL:
    base_premium = st.number_input("Base premium", 50.0, 10000.0, 100.0, 10.0)
    floor = st.slider("Min factor (discount cap)", 0.5, 1.0, 0.75, 0.01)
    cap = st.slider("Max factor (surcharge cap)", 1.0, 2.5, 1.50, 0.01)
    slope = st.slider("Pricing elasticity (slope)", 0.5, 3.0, 1.75, 0.05)

if uploaded and st.button("Score & Price"):
    # Parse JSONL to temp file (reusing existing featurizer)
    text = uploaded.getvalue().decode("utf-8")
    tmp = Path("data/samples/ui_tmp.jsonl"); tmp.parent.mkdir(parents=True, exist_ok=True); tmp.write_text(text, encoding="utf-8")

    feats = featurize_trip(tmp)
    row = np.array([feats[n] for n in feat_names], dtype=float).reshape(1,-1)
    risk = float(model.predict(row)[0]); risk = max(0, min(1, risk))
    priced = price_from_risk(risk, base_premium=base_premium, floor=floor, cap=cap, slope=slope)

    st.subheader("Risk score")
    st.metric("Risk (01, higher = riskier)", f"{risk:.3f}")
    st.subheader("Pricing")
    st.write(pd.DataFrame([{
        "premium": priced["premium"],
        "premium_factor": priced["premium_factor"],
        **priced["params"]
    }]))

    st.subheader("Key features (from this trip)")
    keep = ["hard_brake_rate_100km","p95_speed","std_speed","corner_rate_100km",
            "jerk_p95","jerk_mean","speeding_exposure","night_mile_share","rain_mile_share"]
    view = {k: feats.get(k) for k in keep if k in feats}
    st.write(pd.DataFrame([view]))

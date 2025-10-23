import json
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt

from src.features.featurize import featurize_trip
from src.models.pricing import price_from_risk
from src.gamification.badges import make_badges
import math

MODEL_PATH = "models/gbm_risk.cbm"
FEATS_PATH = "models/gbm_risk_features.json"

@st.cache_resource
def load_model():
    model = CatBoostRegressor()
    model.load_model(MODEL_PATH)
    feat_names = json.loads(Path(FEATS_PATH).read_text())
    return model, feat_names

def compute_top_contributors(model, x, feat_names, topk=10):
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(x)[0]
        pairs = sorted(
            [{"feature": n, "value": float(v)} for n, v in zip(feat_names, sv)],
            key=lambda d: abs(d["value"]), reverse=True
        )[:topk]
        return pairs
    except Exception as e:
        return [{"feature":"(explainability unavailable)", "value":0.0, "error":str(e)}]

st.set_page_config(page_title="Telematics UBI Demo", layout="wide")
st.title("Telematics UBI – Risk & Pricing Demo")

model, feat_names = load_model()

with st.sidebar:
    st.header("Pricing controls")
    base_premium = st.number_input("Base premium", 50.0, 10000.0, 100.0, 10.0)
    floor = st.slider("Min factor (discount cap)", 0.5, 1.0, 0.75, 0.01)
    cap = st.slider("Max factor (surcharge cap)", 1.0, 2.5, 1.50, 0.01)
    slope = st.slider("Pricing elasticity (slope)", 0.5, 3.0, 1.75, 0.05)
    pivot = st.slider("Pivot risk (factor≈1 at this risk)", 0.10, 0.90, 0.50, 0.01)

    

uploaded = st.file_uploader("Upload trip JSONL (one JSON object per line)", type=["jsonl","txt"])

if uploaded:
    tmp = Path("data/samples/ui_tmp.jsonl")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_bytes(uploaded.getvalue())

    feats = featurize_trip(tmp)
    row = np.array([feats[n] for n in feat_names], dtype=float).reshape(1, -1)
    risk = float(model.predict(row)[0])
    risk = max(0.0, min(1.0, risk))

    # Compute unclamped factor to show clamping reason
    from math import exp, log
    def logit(x, eps=1e-6): x = min(1 - eps, max(eps, x)); return log(x/(1-x))
    intercept = - slope * logit(pivot)
    unclamped = math.exp(intercept + slope * logit(risk))  
    
    # Then price with guardrails
    from src.models.pricing import price_from_risk
    
    priced = price_from_risk(risk, base_premium=base_premium, floor=floor, cap=cap, slope=slope, intercept=intercept)

    prev_prem = st.session_state.get("last_premium")
    delta_prem = None if prev_prem is None else round(priced["premium"] - prev_prem, 2)
    st.session_state["last_premium"] = priced["premium"]

    clamp_note = ("CLAMPED at floor" if abs(priced["premium_factor"]-floor) < 1e-6
              else "CLAMPED at cap" if abs(priced["premium_factor"]-cap) < 1e-6
              else "Not clamped") 
    

    badges = make_badges(feats)
    topk = compute_top_contributors(model, row, feat_names, topk=10)

    col1, col2, col3 = st.columns([1,1,1])

    # --- Risk block ---
    with col1:
        st.subheader("Risk")
        st.metric("Risk score (0–1)", round(risk, 3))   # numeric, not f-string
        st.progress(min(max(risk, 0.0), 1.0))

# --- Pricing block ---
    with col2:
        st.subheader("Pricing")
        st.metric(label="Premium ($)", value=round(priced["premium"], 2), delta=delta_prem)
        st.caption(
            f"Factor: {priced['premium_factor']:.3f} | base={base_premium:.2f} | "
            f"unclamped={unclamped:.3f} | "
            f"{'CLAMPED at floor' if abs(priced['premium_factor']-floor)<1e-6 else ('CLAMPED at cap' if abs(priced['premium_factor']-cap)<1e-6 else 'Not clamped')} | "
            f"slope={slope:.2f}, pivot={pivot:.2f}, floor={floor:.2f}, cap={cap:.2f}"
        )

    with col3:
        st.subheader("Badges")
        if badges:
            for b in badges:
                medal = "🥇" if b["tier"]=="gold" else "🥈"
                st.write(f"{medal} **{b['name']}**  {b['reason']}")
        else:
            st.write("No badges earned this trip. Keep it smooth!")


    export = {
    "risk_score": round(risk, 6),
    "premium": round(priced["premium"], 2),
    "premium_factor": round(priced["premium_factor"], 6),
    "unclamped_factor": round(unclamped, 6),
    "params": {
        "base": base_premium, "floor": floor, "cap": cap,
        "slope": slope, "pivot": pivot},
    "badges": badges}

    st.download_button(
        "Download pricing JSON",
        data=json.dumps(export, indent=2).encode("utf-8"),
        file_name="pricing_result.json",
        mime="application/json")
        

    st.divider()
    st.subheader("Top contributors (SHAP)")
    dfc = pd.DataFrame(topk)
    if "error" in dfc.columns:
        st.write(dfc)
    else:
        # bar chart (horizontal)
        fig, ax = plt.subplots(figsize=(7,4))
        names = list(dfc["feature"])[::-1]
        vals  = list(dfc["value"])[::-1]
        ax.barh(range(len(vals)), vals)
        ax.set_yticks(range(len(vals)))
        ax.set_yticklabels(names)
        ax.set_xlabel("SHAP contribution to risk")
        ax.set_title("Top contributors")
        st.pyplot(fig)

    
    st.divider()
    st.subheader("Trip feature snapshot")
    keep = [
        "dist_km","duration_sec","avg_speed","p50_speed","p95_speed","std_speed",
        "hard_brakes","hard_brake_rate_100km","harsh_accels","harsh_accel_rate_100km",
        "cornering_events","corner_rate_100km","speeding_exposure","idle_share",
        "night_mile_share","rain_mile_share","jerk_mean","jerk_p95"
    ]
    view = {k: feats.get(k) for k in keep if k in feats}
    st.dataframe(pd.DataFrame([view]))
else:
    st.info("Upload a JSONL trip file to compute risk & price.")

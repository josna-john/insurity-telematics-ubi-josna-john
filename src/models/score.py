from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from catboost import CatBoostRegressor

from src.features.featurize import featurize_trip

"""
Score a telematics trip with the trained GBM model.

Reads a JSONL trip, computes engineered features, assembles the model
input vector using a persisted feature order, and outputs a bounded
risk score in [0, 1]. Optionally emits per-feature SHAP contributions.
"""


def load_feature_order(path: str | Path):
    """
    Load the persisted feature name order used at training time.

    Args:
        path: Path to a JSON file containing a list of feature names.

    Returns:
        List of feature names in the expected model order.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def assemble_vector(feats: dict, feat_names):
    """
    Assemble a numeric row vector X from a feature dict and an ordered name list.

    Args:
        feats: Mapping of feature name -> value (from featurize_trip).
        feat_names: Ordered list of feature names.

    Returns:
        NumPy array shaped (1, n_features) suitable for model.predict().
    """
    row = [feats[name] for name in feat_names]
    return np.array(row, dtype=float).reshape(1, -1)


def main():
    """
    CLI entry point.

    Usage:
        python -m src.models.score --input data/samples/trip.jsonl \
            --model models/gbm_risk.cbm \
            --featnames models/gbm_risk_features.json \
            --out data/derived/risk_score.json \
            [--with-shap --topk 8]
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to trip JSONL")
    ap.add_argument("--model", default="models/gbm_risk.cbm")
    ap.add_argument("--featnames", default="models/gbm_risk_features.json")
    ap.add_argument("--out", default="data/derived/risk_score.json")
    ap.add_argument("--with-shap", action="store_true", help="Also output top SHAP contributions")
    ap.add_argument("--topk", type=int, default=8)
    args = ap.parse_args()

    feat_names = load_feature_order(args.featnames)
    feats = featurize_trip(Path(args.input))
    X = assemble_vector(feats, feat_names)

    model = CatBoostRegressor()
    model.load_model(args.model)
    risk = float(model.predict(X)[0])
    risk = max(0.0, min(1.0, risk))

    out = {
        "trip_id": feats["trip_id"],
        "driver_id": feats["driver_id"],
        "risk_score": risk,
    }

    if args.with_shap:
        import shap  # optional dependency; used only when requested
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X)
        pairs = sorted(
            [{"feature": n, "value": float(v)} for n, v in zip(feat_names, sv[0])],
            key=lambda d: abs(d["value"]),
            reverse=True
        )[: args.topk]
        out["top_contributors"] = pairs

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor

from src.features.featurize import featurize_trip

def plot_top_shap(pairs, title: str, out_png: Path):
    """
    pairs: list of {"feature": str, "value": float} ranked by |value|
    Saves a horizontal bar chart with sign-aware colors and numeric labels.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    names = [p["feature"] for p in pairs][::-1]
    vals  = np.array([p["value"] for p in pairs], dtype=float)[::-1]
    colors = np.where(vals >= 0, "#C23B22", "#1F77B4")  # red up-risk, blue down-risk

    plt.figure(figsize=(9, 6))
    y = np.arange(len(vals))
    plt.barh(y, vals, color=colors)
    plt.axvline(0, color="k", linewidth=0.8)
    plt.yticks(y, names)
    plt.xlabel("SHAP contribution to risk (positive = increases risk)")
    plt.title(title)
    # numeric labels at bar ends
    for i, v in enumerate(vals):
        plt.text(v + (0.003 if v >= 0 else -0.003), i, f"{v:+.3f}",
                 va="center", ha="left" if v >= 0 else "right", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-data", default="data/training/features.csv")
    ap.add_argument("--model", default="models/gbm_risk.cbm")
    ap.add_argument("--featnames", default="models/gbm_risk_features.json")
    ap.add_argument("--trip", required=True, help="JSONL path to explain")
    ap.add_argument("--outdir", default="docs/explain")
    ap.add_argument("--topk", type=int, default=12)
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # Load training data to build a background for SHAP (TreeExplainer is model-based, but background helps plots)
    df = pd.read_csv(args.train_data)
    with open(args.featnames, "r", encoding="utf-8") as f:
        feat_names = json.load(f)
    X_train = df[feat_names].values

    model = CatBoostRegressor()
    model.load_model(args.model)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    # Global summary (PNG)
    plt.figure(figsize=(7, 6))
    shap.summary_plot(shap_values, X_train, feature_names=feat_names, show=False)
    plt.title("Global SHAP summary")
    plt.tight_layout()
    plt.savefig(outdir / "global_shap_summary.png", dpi=150)
    plt.close()

    # Explain a single trip
    feats = featurize_trip(Path(args.trip))
    x = np.array([feats[n] for n in feat_names], dtype=float).reshape(1, -1)
    sv = explainer.shap_values(x)[0]
    pairs = sorted(
        [{"feature": n, "value": float(v)} for n, v in zip(feat_names, sv)],
        key=lambda d: abs(d["value"]),
        reverse=True
    )[: args.topk]

    # Simple horizontal bar plot
    # plt.figure(figsize=(8, 5))
    # names = [p["feature"] for p in pairs][::-1]
    # vals = [p["value"] for p in pairs][::-1]
    # plt.barh(range(len(vals)), vals)
    # plt.yticks(range(len(vals)), names)
    # plt.xlabel("SHAP contribution to risk")
    # plt.title(f"Top {args.topk} contributors  {feats['trip_id']}")
    # plt.tight_layout()
    # plt.savefig(outdir / f"{feats['trip_id']}_top_shap.png", dpi=150)
    # plt.close()

    # Per-trip top contributors (sign-aware bar)
    plot_top_shap(
        pairs,
        title=f"Top {args.topk} contributors  {feats['trip_id']}",
        out_png=outdir / f"{feats['trip_id']}_top_shap.png"
    )




    with open(outdir / f"{feats['trip_id']}_top_shap.json", "w", encoding="utf-8") as f:
        json.dump({"trip_id": feats["trip_id"], "top_contributors": pairs}, f, indent=2)

    print(f"Wrote global_shap_summary.png and {feats['trip_id']}_top_shap.png to {outdir}")

if __name__ == "__main__":
    main()

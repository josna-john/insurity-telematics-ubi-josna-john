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

"""
Explainability utilities for the telematics risk model.

This module:
  • Builds a SHAP background from the training feature matrix.
  • Produces a global SHAP summary plot for the fitted CatBoost model.
  • Explains a single trip by computing per-feature SHAP contributions and
    saving both a sign-aware bar chart and a compact JSON artifact.
"""


def plot_top_shap(pairs, title: str, out_png: Path):
    """
    Render a sign-aware horizontal bar chart of top SHAP contributors.

    Args:
        pairs (list[dict]): Items like {"feature": str, "value": float}, sorted by |value|.
        title (str): Plot title.
        out_png (Path): Destination path for the PNG output.

    Side Effects:
        Saves a PNG file at `out_png`.
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
    for i, v in enumerate(vals):
        plt.text(
            v + (0.003 if v >= 0 else -0.003),
            i,
            f"{v:+.3f}",
            va="center",
            ha="left" if v >= 0 else "right",
            fontsize=9,
        )
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def main():
    """
    CLI entry point.

    Arguments (via argparse):
        --train-data (str): Path to training feature CSV (for SHAP background).
        --model (str): Path to CatBoost .cbm file.
        --featnames (str): JSON file with ordered feature names used by the model.
        --trip (str, required): Path to the JSONL trip to explain.
        --outdir (str): Output directory for plots/artifacts (default: docs/explain).
        --topk (int): Number of top contributors to display (default: 12).

    Behavior:
        1) Loads training matrix to compute background for SHAP.
        2) Loads the CatBoost model and generates a global SHAP summary plot.
        3) Featurizes the provided trip, computes SHAP for that instance,
           and writes both a PNG bar chart and a compact JSON with top contributors.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-data", default="data/training/features.csv")
    ap.add_argument("--model", default="models/gbm_risk.cbm")
    ap.add_argument("--featnames", default="models/gbm_risk_features.json")
    ap.add_argument("--trip", required=True, help="JSONL path to explain")
    ap.add_argument("--outdir", default="docs/explain")
    ap.add_argument("--topk", type=int, default=12)
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # Build SHAP background from training features.
    df = pd.read_csv(args.train_data)
    with open(args.featnames, "r", encoding="utf-8") as f:
        feat_names = json.load(f)
    X_train = df[feat_names].values

    model = CatBoostRegressor()
    model.load_model(args.model)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    # Global SHAP summary (PNG).
    plt.figure(figsize=(7, 6))
    shap.summary_plot(shap_values, X_train, feature_names=feat_names, show=False)
    plt.title("Global SHAP summary")
    plt.tight_layout()
    plt.savefig(outdir / "global_shap_summary.png", dpi=150)
    plt.close()

    # Per-trip explanation.
    feats = featurize_trip(Path(args.trip))
    x = np.array([feats[n] for n in feat_names], dtype=float).reshape(1, -1)
    sv = explainer.shap_values(x)[0]
    pairs = sorted(
        [{"feature": n, "value": float(v)} for n, v in zip(feat_names, sv)],
        key=lambda d: abs(d["value"]),
        reverse=True
    )[: args.topk]

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

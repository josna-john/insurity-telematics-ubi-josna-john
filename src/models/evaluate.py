from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from catboost import CatBoostRegressor, Pool

from numpy.typing import NDArray
from typing import Tuple

"""
Model evaluation utilities for the telematics risk scorer.

This module:
  - Loads a training CSV and constructs design matrices.
  - Trains and evaluates CatBoost (primary) and optionally LightGBM baselines.
  - Reports RMSE and R² on a validation split.
  - Emits diagnostic plots: residuals, score distribution, calibration curve,
    and CatBoost global feature importance (gain) with a companion CSV.
"""


def safe_rmse(y_true, y_pred) -> float:
    """
    Compute RMSE, compatible with older scikit-learn versions that lack
    the `squared` argument in `mean_squared_error`.

    Args:
        y_true: Ground-truth array-like.
        y_pred: Predicted array-like.

    Returns:
        float: Root mean squared error.
    """
    try:
        return mean_squared_error(y_true, y_pred, squared=False)
    except TypeError:
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))


# Optional LightGBM baseline
try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False


EXCLUDE = {"trip_id", "driver_id", "mode", "target"}


def load_xy(csv_path: str):
    """
    Load features and target from a training CSV.

    Args:
        csv_path: Path to CSV produced by the dataset generator.

    Returns:
        Tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
            (full DataFrame, X matrix, y vector, feature names in order)
    """
    df = pd.read_csv(csv_path)
    y = df["target"].values.astype(float)
    feat_names = [c for c in df.columns if c not in EXCLUDE]
    X = df[feat_names].values.astype(float)
    return df, X, y, feat_names


def catboost_fit(
    Xtr: NDArray[np.floating],
    ytr: NDArray[np.floating],
    Xva: NDArray[np.floating],
    yva: NDArray[np.floating],
    feat_names: list[str],
) -> Tuple[CatBoostRegressor, NDArray[np.floating]]:
    """
    Fit a CatBoostRegressor and return predictions on the validation set.

    Args:
        Xtr, ytr: Training features and target.
        Xva, yva: Validation features and target.
        feat_names: Ordered feature names (unused by CatBoost here, but kept for symmetry).

    Returns:
        (model, preds_on_val)
    """
    model = CatBoostRegressor(
        loss_function="RMSE",
        depth=6,
        learning_rate=0.1,
        iterations=500,
        random_seed=42,
        verbose=False,
    )
    model.fit(Pool(Xtr, label=ytr), eval_set=Pool(Xva, label=yva))
    pred = np.asarray(model.predict(Xva), dtype=float).reshape(-1)
    return model, pred


def lightgbm_fit(
    Xtr: NDArray[np.floating],
    ytr: NDArray[np.floating],
    Xva: NDArray[np.floating],
    yva: NDArray[np.floating],
    feat_names: list[str],
) -> Tuple["LGBMRegressor", NDArray[np.floating]]:
    """
    Fit an LGBMRegressor baseline and return predictions on the validation set.

    Args:
        Xtr, ytr: Training features and target.
        Xva, yva: Validation features and target.
        feat_names: Ordered feature names (unused directly by this baseline).

    Returns:
        (model, preds_on_val)
    """
    model = LGBMRegressor(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=-1,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
    )
    model.fit(Xtr, ytr)
    pred = np.asarray(model.predict(Xva), dtype=float).reshape(-1)
    return model, pred


def calibration_plot(y, pred, out_png):
    """
    Plot a regression-style calibration curve by binning predictions and
    comparing mean predicted vs. mean actual.

    Args:
        y: Ground-truth values (validation).
        pred: Predicted values (validation).
        out_png: Output PNG path.
    """
    df = pd.DataFrame({"y": y, "p": np.clip(pred, 0, 1)})
    df["bin"] = pd.qcut(df["p"], q=10, duplicates="drop")
    agg = df.groupby("bin", observed=False).agg(y_mean=("y", "mean"), p_mean=("p", "mean")).reset_index()
    plt.figure(figsize=(5, 4))
    plt.plot(agg["p_mean"], agg["y_mean"], marker="o")
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("Predicted (binned mean)")
    plt.ylabel("Actual (binned mean)")
    plt.title("Calibration (regression reliability)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def residuals_plot(y, pred, out_png):
    """
    Plot residuals (y - pred) versus predictions to diagnose bias/heteroscedasticity.

    Args:
        y: Ground-truth values (validation).
        pred: Predicted values (validation).
        out_png: Output PNG path.
    """
    res = y - pred
    plt.figure(figsize=(5, 4))
    plt.scatter(pred, res, s=8, alpha=0.5)
    plt.axhline(0, color="k", linestyle="--")
    plt.xlabel("Predicted")
    plt.ylabel("Residual (y - pred)")
    plt.title("Residuals vs Prediction")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def dist_plot(pred, out_png):
    """
    Plot the distribution of validation predictions (clipped to [0,1]).

    Args:
        pred: Predicted values (validation).
        out_png: Output PNG path.
    """
    plt.figure(figsize=(5, 4))
    plt.hist(np.clip(pred, 0, 1), bins=30)
    plt.xlabel("Predicted risk")
    plt.ylabel("Count")
    plt.title("Risk score distribution (val)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def catboost_importance_plot(model, feat_names, out_png, top=20):
    """
    Render CatBoost global feature importance (gain) and write a companion CSV.

    Args:
        model: Fitted CatBoostRegressor.
        feat_names: Ordered list of feature names.
        out_png: Output PNG path for the bar plot.
        top: Number of top features to display.

    Side Effects:
        - Saves PNG (barh gain plot).
        - Saves CSV with feature, gain, and percentage share.
    """
    try:
        imps = np.asarray(model.get_feature_importance(), dtype=float)
        total = imps.sum() if imps.sum() > 0 else 1.0
        pct = 100.0 * imps / total

        order = np.argsort(imps)[::-1][:top]
        imps_o = imps[order]
        pct_o = pct[order]
        names = [feat_names[i] for i in order]

        plt.figure(figsize=(7.5, 6))
        y = np.arange(len(order))
        plt.barh(y, imps_o[::-1])
        plt.yticks(y, names[::-1])
        for i, (v, p) in enumerate(zip(imps_o[::-1], pct_o[::-1])):
            plt.text(v * 1.01, i, f"{p:,.1f}%", va="center", fontsize=9)

        plt.xlabel("Gain (higher = more important)")
        plt.title("CatBoost Global Feature Importance (Gain)")
        plt.tight_layout()
        plt.savefig(out_png, dpi=150)
        plt.close()

        pd.DataFrame({"feature": names, "gain": imps_o, "share_pct": pct_o}).to_csv(
            Path(out_png).with_suffix(".csv"), index=False
        )
    except Exception as e:
        print("importance plot failed:", e)


def main():
    """
    CLI to train/evaluate models and emit diagnostics.

    Args (via argparse):
        --data: Path to features CSV (default: data/training/features.csv).
        --outdir: Directory to write plots and comparison CSV (default: docs/metrics).

    Behavior:
        - Train/val split (25% val).
        - Fit CatBoost and (if available) LightGBM.
        - Print RMSE and R².
        - Write diagnostic plots and a small leaderboard CSV.
    """
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/training/features.csv")
    ap.add_argument("--outdir", default="docs/metrics")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df, X, y, fe = load_xy(args.data)
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.25, random_state=42)

    cb, p_cb = catboost_fit(Xtr, ytr, Xva, yva, fe)
    rmse_cb = safe_rmse(yva, p_cb)
    r2_cb = r2_score(yva, p_cb)

    rmse_lgb = r2_lgb = None
    if HAS_LGBM:
        lgb, p_lgb = lightgbm_fit(Xtr, ytr, Xva, yva, fe)
        rmse_lgb = safe_rmse(yva, p_lgb)
        r2_lgb = r2_score(yva, p_lgb)

    residuals_plot(yva, p_cb, outdir / "residuals.png")
    dist_plot(p_cb, outdir / "score_distribution.png")
    calibration_plot(yva, p_cb, outdir / "calibration.png")
    catboost_importance_plot(cb, fe, outdir / "feature_importance.png")

    rows = [{"model": "CatBoost", "rmse": rmse_cb, "r2": r2_cb}]
    if rmse_lgb is not None:
        rows.append({"model": "LightGBM", "rmse": rmse_lgb, "r2": r2_lgb})
    pd.DataFrame(rows).to_csv(outdir / "model_comparison.csv", index=False)

    print(f"CatBoost  RMSE={rmse_cb:.4f}  R2={r2_cb:.4f}")
    if rmse_lgb is not None:
        print(f"LightGBM RMSE={rmse_lgb:.4f}  R2={r2_lgb:.4f}")
    print(f"Wrote plots to {outdir}")


if __name__ == "__main__":
    main()

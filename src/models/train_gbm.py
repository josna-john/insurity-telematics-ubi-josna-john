from __future__ import annotations
import json
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

EXCLUDE_COLS = {
    "trip_id","driver_id","mode","target"  # meta/labels
}

def build_features(df: pd.DataFrame):
    cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    X = df[cols].values
    return X, cols

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/training/features.csv")
    ap.add_argument("--model-out", default="models/gbm_risk.cbm")
    ap.add_argument("--featnames-out", default="models/gbm_risk_features.json")
    args = ap.parse_args()

    Path("models").mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data)
    y = df["target"].values
    X, feat_names = build_features(df)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)

    model = CatBoostRegressor(
        loss_function="RMSE",
        depth=6,
        learning_rate=0.1,
        iterations=500,
        random_seed=42,
        verbose=False
    )

    model.fit(Pool(X_train, label=y_train), eval_set=Pool(X_val, label=y_val))

    pred = model.predict(X_val)
    mse  = mean_squared_error(y_val, pred)
    rmse = float(np.sqrt(mse))
    r2   = r2_score(y_val, pred)
    print(f"Validation RMSE: {rmse:.4f} | R^2: {r2:.4f}")

    model.save_model(args.model_out)
    with open(args.featnames_out, "w", encoding="utf-8") as f:
        json.dump(feat_names, f, indent=2)
    print(f"Saved model to {args.model_out}")
    print(f"Saved feature names to {args.featnames_out}")

if __name__ == "__main__":
    main()

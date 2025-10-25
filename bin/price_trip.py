import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse, json
import numpy as np
from pathlib import Path as P
from catboost import CatBoostRegressor

from src.models.pricing import price_from_risk
from src.features.featurize import featurize_trip
from src.models.score import load_feature_order

"""
Command-line tool: score a trip JSONL file and compute the corresponding price.

Steps:
  1) Load the trained CatBoost model and the expected feature ordering.
  2) Featurize the input trip (JSONL) and build a model-ready vector.
  3) Predict a risk score (clamped to [0, 1]).
  4) Convert risk to premium via `price_from_risk` (GLM-style mapping with
     floor/cap/slope and optional territory/vehicle multipliers).
  5) Write a JSON report with pricing details to --out and echo it to stdout.

Typical usage:
  python bin/price_trip.py --trip data/samples/trip_eval.jsonl --base-premium 120
"""


def main():
    """
    Parse CLI args, score the trip, price it, and emit a JSON report.

    CLI Arguments:
        --trip (str, required): Path to the input trip JSONL.
        --model (str): Path to CatBoost model file (.cbm).
        --featnames (str): Path to JSON list of feature names (order used by model).
        --base-premium (float): Base premium prior to behavior-based factor. Default: 100.0
        --floor (float): Minimum premium factor (discount cap). Default: 0.75
        --cap (float): Maximum premium factor (surcharge cap). Default: 1.50
        --slope (float): Pricing elasticity controlling sensitivity to risk. Default: 1.75
        --territory-mult (float): Multiplicative factor for territory rating. Default: 1.0
        --vehicle-mult (float): Multiplicative factor for vehicle rating. Default: 1.0
        --out (str): Output JSON path for priced result. Default: data/derived/trip_price.json

    Output JSON fields (subset):
        - risk_score: Predicted risk in [0, 1]
        - premium_factor: Final factor after clamping and multipliers
        - premium: Premium in currency units
        - trip_id / driver_id: Copied from featurized trip
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--trip", required=True)
    ap.add_argument("--model", default="models/gbm_risk.cbm")
    ap.add_argument("--featnames", default="models/gbm_risk_features.json")
    ap.add_argument("--base-premium", type=float, default=100.0)
    ap.add_argument("--floor", type=float, default=1.50)
    ap.add_argument("--cap", type=float, default=1.50)
    ap.add_argument("--slope", type=float, default=1.75)
    ap.add_argument("--territory-mult", type=float, default=1.0)
    ap.add_argument("--vehicle-mult", type=float, default=1.0)
    ap.add_argument("--out", default="data/derived/trip_price.json")
    args = ap.parse_args()

    feat_names = load_feature_order(args.featnames)
    feats = featurize_trip(P(args.trip))
    row = [feats[n] for n in feat_names]
    X = np.array(row, dtype=float).reshape(1, -1)

    model = CatBoostRegressor(); model.load_model(args.model)
    risk = float(model.predict(X)[0]); risk = max(0.0, min(1.0, risk))

    priced = price_from_risk(
        risk,
        base_premium=args.base_premium,
        floor=args.floor,
        cap=args.cap,
        slope=args.slope,
        territory_mult=args.territory_mult,
        vehicle_mult=args.vehicle_mult,
    )

    priced.update({"trip_id": feats["trip_id"], "driver_id": feats["driver_id"]})
    P(args.out).parent.mkdir(parents=True, exist_ok=True)
    P(args.out).write_text(json.dumps(priced, indent=2), encoding="utf-8")
    print(json.dumps(priced, indent=2))


if __name__ == "__main__":
    main()

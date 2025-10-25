import sys, json, subprocess, math
from pathlib import Path

"""
Utility to synthesize a *mid-risk* trip by blending simulated smooth and aggressive
JSONL trips until the model’s predicted risk is near a target value.

Workflow:
  1) Generate paired trips (smooth, aggressive) of equal length via simulate_stream.py.
  2) Binary-search the fraction of aggressive records to mix into the smooth trip.
  3) Score each blend using the trained GBM; stop when within tolerance of target.
  4) Write the blended trip to --out with normalized trip/driver identifiers.

Intended use:
  python bin/make_midrisk_blend.py --target 0.50 --tol 0.02 --duration 60 --hz 10 \
    --out data/samples/trip_mid_blend.jsonl
"""

# enable project imports when run from /bin
from pathlib import Path as _P; sys.path.append(str(_P(__file__).resolve().parents[1]))

from src.features.featurize import featurize_trip
from catboost import CatBoostRegressor
import numpy as np

MODEL_PATH = "models/gbm_risk.cbm"
FEATS_PATH = "models/gbm_risk_features.json"


def load_model():
    """
    Load the trained CatBoost model and the feature name order.

    Returns:
        tuple[CatBoostRegressor, list[str]]: Loaded model and ordered feature names.
    """
    model = CatBoostRegressor(); model.load_model(MODEL_PATH)
    feat_names = json.loads(Path(FEATS_PATH).read_text(encoding="utf-8"))
    return model, feat_names


def score(jsonl: Path, model, feat_names):
    """
    Compute a clamped risk score for a single trip JSONL file.

    Args:
        jsonl (Path): Path to trip records in JSONL format.
        model:       Trained model with a .predict API.
        feat_names:  Ordered feature names expected by the model.

    Returns:
        float: Risk score in [0, 1].
    """
    feats = featurize_trip(jsonl)
    row = np.array([feats[n] for n in feat_names], dtype=float).reshape(1, -1)
    risk = float(model.predict(row)[0]); return max(0.0, min(1.0, risk))


def mix_jsonl(smooth: Path, aggressive: Path, out: Path, frac_aggr: float, trip_id="mid_blend", driver_id="drv_mid"):
    """
    Blend two same-length trips by taking the first k=f*N lines from the aggressive
    trip and the remaining lines from the smooth trip, then rewrite identifiers.

    Args:
        smooth (Path): Path to the smooth trip JSONL.
        aggressive (Path): Path to the aggressive trip JSONL.
        out (Path): Output path for the blended JSONL.
        frac_aggr (float): Fraction of aggressive samples to place at the front [0,1].
        trip_id (str): Trip identifier to set on all output records.
        driver_id (str): Driver identifier to set on all output records.
    """
    s_lines = smooth.read_text(encoding="utf-8").strip().splitlines()
    a_lines = aggressive.read_text(encoding="utf-8").strip().splitlines()
    n = min(len(s_lines), len(a_lines))
    k = int(max(0, min(1, frac_aggr)) * n)
    chosen = a_lines[:k] + s_lines[k:n]
    with out.open("w", encoding="utf-8") as f:
        for ln in chosen:
            rec = json.loads(ln)
            rec["trip_id"] = trip_id
            rec["driver_id"] = driver_id
            f.write(json.dumps(rec) + "\n")


def main():
    """
    CLI entry point: produce a mid-risk blended trip near a target score.

    Args (via argparse):
        --target (float): Desired risk score (default 0.50).
        --tol (float): Acceptable absolute error from target (default 0.02).
        --duration (int): Simulator duration in seconds (default 60).
        --hz (int): Simulator sampling frequency (records/sec, default 10).
        --out (str): Output JSONL path for the blended trip.

    Side effects:
        - Generates temporary smooth/aggressive trips alongside the output.
        - Writes the best or within-tolerance blended trip to --out.
        - Prints iteration diagnostics and final selection.
    """
    import argparse, tempfile
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, default=0.50)
    ap.add_argument("--tol", type=float, default=0.02)
    ap.add_argument("--duration", type=int, default=60)
    ap.add_argument("--hz", type=int, default=10)
    ap.add_argument("--out", default="data/samples/trip_mid_blend.jsonl")
    args = ap.parse_args()

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)

    s_path = out.parent / "_tmp_smooth.jsonl"
    a_path = out.parent / "_tmp_aggressive.jsonl"
    for mode, path in [("smooth", s_path), ("aggressive", a_path)]:
        cmd = ["python","bin/simulate_stream.py","--mode",mode,
               "--duration",str(args.duration),"--hz",str(args.hz),
               "--out",str(path),"--trip-id",f"{mode}_base","--driver-id","drv_mid"]
        subprocess.run(cmd, check=True)

    model, feat_names = load_model()
    r_s = score(s_path, model, feat_names)
    r_a = score(a_path, model, feat_names)
    print(f"smooth risk  {r_s:.3f} | aggressive risk  {r_a:.3f}")

    lo, hi = 0.0, 1.0
    best = (10.0, None, None)  # diff, f, risk
    for it in range(16):  # ~binary search
        f = 0.5 if it==0 else 0.5*(lo+hi)
        tmp = out.parent / "_tmp_mix.jsonl"
        mix_jsonl(s_path, a_path, tmp, f, trip_id="mid_blend", driver_id="drv_mid")
        r = score(tmp, model, feat_names)
        diff = abs(r - args.target)
        print(f"[{it:02d}] frac_aggr={f:.3f} -> risk={r:.3f} (diff={diff:.3f})")
        if diff < best[0]:
            best = (diff, f, r)
        if diff <= args.tol:
            tmp.replace(out)
            print(f" Saved mid-risk blended trip to {out} (risk={r:.3f}, frac_aggr={f:.3f})")
            return
        if r < args.target:
            lo = f
        else:
            hi = f

    _, f, r = best
    mix_jsonl(s_path, a_path, out, f, trip_id="mid_blend", driver_id="drv_mid")
    print(f" Saved closest blend to {out} (risk={r:.3f}, frac_aggr={f:.3f})")


if __name__ == "__main__":
    main()

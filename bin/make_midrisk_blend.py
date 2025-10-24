import sys, json, subprocess, math
from pathlib import Path
# ensure project imports work when run from /bin
from pathlib import Path as _P; sys.path.append(str(_P(__file__).resolve().parents[1]))

from src.features.featurize import featurize_trip
from catboost import CatBoostRegressor
import numpy as np

MODEL_PATH = "models/gbm_risk.cbm"
FEATS_PATH = "models/gbm_risk_features.json"

def load_model():
    model = CatBoostRegressor(); model.load_model(MODEL_PATH)
    feat_names = json.loads(Path(FEATS_PATH).read_text(encoding="utf-8"))
    return model, feat_names

def score(jsonl: Path, model, feat_names):
    feats = featurize_trip(jsonl)
    row = np.array([feats[n] for n in feat_names], dtype=float).reshape(1, -1)
    risk = float(model.predict(row)[0]); return max(0.0, min(1.0, risk))

def mix_jsonl(smooth: Path, aggressive: Path, out: Path, frac_aggr: float, trip_id="mid_blend", driver_id="drv_mid"):
    """Create mixed trip: first f*N records from aggressive, rest from smooth (same length)."""
    s_lines = smooth.read_text(encoding="utf-8").strip().splitlines()
    a_lines = aggressive.read_text(encoding="utf-8").strip().splitlines()
    n = min(len(s_lines), len(a_lines))
    k = int(max(0, min(1, frac_aggr)) * n)
    chosen = a_lines[:k] + s_lines[k:n]
    # rewrite trip_id/driver_id consistently
    with out.open("w", encoding="utf-8") as f:
        for ln in chosen:
            rec = json.loads(ln)
            rec["trip_id"] = trip_id
            rec["driver_id"] = driver_id
            f.write(json.dumps(rec) + "\n")

def main():
    import argparse, tempfile
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, default=0.50)
    ap.add_argument("--tol", type=float, default=0.02)
    ap.add_argument("--duration", type=int, default=60)
    ap.add_argument("--hz", type=int, default=10)
    ap.add_argument("--out", default="data/samples/trip_mid_blend.jsonl")
    args = ap.parse_args()

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)

    # generate paired smooth/aggressive trips of equal length
    s_path = out.parent / "_tmp_smooth.jsonl"
    a_path = out.parent / "_tmp_aggressive.jsonl"
    for mode, path in [("smooth", s_path), ("aggressive", a_path)]:
        cmd = ["python","bin/simulate_stream.py","--mode",mode,
               "--duration",str(args.duration),"--hz",str(args.hz),
               "--out",str(path),"--trip-id",f"{mode}_base","--driver-id","drv_mid"]
        subprocess.run(cmd, check=True)

    model, feat_names = load_model()
    # quick endpoints to understand the range
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
        # decide direction
        if r < args.target:
            lo = f
        else:
            hi = f

    # save closest if we didn't hit tolerance
    _, f, r = best
    mix_jsonl(s_path, a_path, out, f, trip_id="mid_blend", driver_id="drv_mid")
    print(f" Saved closest blend to {out} (risk={r:.3f}, frac_aggr={f:.3f})")

if __name__ == "__main__":
    main()

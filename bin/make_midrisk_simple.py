import sys, subprocess, json, random
from pathlib import Path
from time import time
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

def score_trip(jsonl_path: Path, model, feat_names):
    feats = featurize_trip(jsonl_path)
    row = np.array([feats[n] for n in feat_names], dtype=float).reshape(1, -1)
    risk = float(model.predict(row)[0])
    return max(0.0, min(1.0, risk)), feats

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, default=0.50)
    ap.add_argument("--tol", type=float, default=0.03)
    ap.add_argument("--attempts", type=int, default=12)
    ap.add_argument("--duration", type=int, default=60)
    ap.add_argument("--hz", type=int, default=10)
    ap.add_argument("--out", default="data/samples/trip_mid.jsonl")
    args = ap.parse_args()

    model, feat_names = load_model()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)

    best = None
    for k in range(1, args.attempts+1):
        tmp = out.parent / f"_tmp_normal_{int(time())}_{k}.jsonl"
        trip_id = f"mid_normal_{k:02d}"
        cmd = ["python","bin/simulate_stream.py","--mode","normal",
               "--duration",str(args.duration),"--hz",str(args.hz),
               "--out",str(tmp),"--trip-id",trip_id,"--driver-id","drv_mid"]
        subprocess.run(cmd, check=True)
        risk, feats = score_trip(tmp, model, feat_names)
        diff = abs(risk - args.target)
        print(f"[{k:02d}] risk={risk:.3f} diff={diff:.3f}  -> {tmp.name}")
        if best is None or diff < best[0]:
            best = (diff, risk, tmp, feats)
        if diff <= args.tol:
            tmp.replace(out)
            print(f" Saved mid-risk trip to {out} (risk={risk:.3f})")
            return

    # if we didn't hit the window, keep the closest
    _, risk, tmp, _ = best
    tmp.replace(out)
    print(f" Could not hit target within tol; saved closest to {out} (risk={risk:.3f})")

if __name__ == "__main__":
    main()

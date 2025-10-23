import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import os
import subprocess
from pathlib import Path
import pandas as pd

from src.features.featurize import featurize_trip

MODE_TO_TARGET = {"smooth": 0.1, "normal": 0.4, "aggressive": 0.9}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trips-per-mode", type=int, default=20)
    ap.add_argument("--duration", type=int, default=60)
    ap.add_argument("--hz", type=int, default=10)
    ap.add_argument("--out", default="data/training/features.csv")
    args = ap.parse_args()

    out_csv = Path(args.out)
    tmp_dir = Path("data/samples/train")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for mode, target in MODE_TO_TARGET.items():
        for i in range(args.trips_per_mode):
            trip_id = f"{mode}_{i:03d}"
            jsonl_path = tmp_dir / f"{trip_id}.jsonl"

            # call the simulator script
            cmd = [
                "python", "bin/simulate_stream.py",
                "--mode", mode, "--duration", str(args.duration),
                "--hz", str(args.hz), "--out", str(jsonl_path),
                "--driver-id", "drv_train", "--trip-id", trip_id
            ]
            subprocess.run(cmd, check=True)

            feats = featurize_trip(jsonl_path)
            feats["target"] = target
            feats["mode"] = mode
            rows.append(feats)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"Wrote training dataset: {out_csv} (rows={len(df)})")

if __name__ == "__main__":
    main()

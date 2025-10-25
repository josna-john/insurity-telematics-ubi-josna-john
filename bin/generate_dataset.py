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
"""
Mapping from simulated driving mode to the target risk value used for training.

Values:
    - "smooth"     -> 0.1  (low risk)
    - "normal"     -> 0.4  (moderate risk)
    - "aggressive" -> 0.9  (high risk)
"""


def main():
    """
    Generate a supervised training dataset from simulated trips.

    This script:
      1) Calls the telematics simulator (`bin/simulate_stream.py`) to produce JSONL
         trip records for each driving mode (smooth/normal/aggressive).
      2) Featurizes each trip via `src.features.featurize.featurize_trip`.
      3) Assigns a mode-specific target risk (see `MODE_TO_TARGET`).
      4) Writes a single CSV of feature rows to `--out`.

    CLI Arguments:
        --trips-per-mode (int): Number of trips to simulate per driving mode. Default: 20.
        --duration       (int): Trip duration in seconds for each simulation. Default: 60.
        --hz             (int): Sampling frequency (records/sec). Default: 10.
        --out           (path): Output CSV path for aggregated features.
                                Default: data/training/features.csv

    Side Effects:
        - Creates `data/samples/train/` with per-trip JSONL files.
        - Ensures parent directories for `--out` exist.

    Outputs:
        CSV with one row per simulated trip, including engineered features, `target`,
        and `mode`. Prints a summary line with the final row count.
    """
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

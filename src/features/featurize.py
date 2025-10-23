from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.data.loader import load_jsonl, infer_hz

# Thresholds (m/s^2, m/s)  tweakable
HARD_BRAKE_THR = -3.0
HARSH_ACCEL_THR =  2.5
CORNER_THR = 2.5
IDLE_SPEED_THR = 0.5  # ~1.1 mph

# Proxy speed limits by road type (m/s)
SPEED_LIMITS = {
    "highway": 30.0,  # ~67 mph
    "city":    15.0,  # ~34 mph
    "rural":   22.0,  # ~49 mph
}

def _speed_limit(row: Dict) -> float:
    return SPEED_LIMITS.get(row["road_type"], 22.0)

def featurize_trip(jsonl_path: Path) -> Dict:
    rows = load_jsonl(jsonl_path)
    df = pd.DataFrame(rows)
    trip_id = str(df["trip_id"].iloc[0])
    driver_id = str(df["driver_id"].iloc[0])

    # infer Hz & dt
    hz = infer_hz(df["timestamp"].tolist())
    dt = 1.0 / max(hz, 1e-6)

    # distance approximation (sum speed*dt)
    dist_m = float((df["speed_mps"] * dt).sum())
    dist_km = dist_m / 1000.0

    # events
    hard_brakes = int((df["accel_long_mps2"] <= HARD_BRAKE_THR).sum())
    harsh_accels = int((df["accel_long_mps2"] >= HARSH_ACCEL_THR).sum())
    cornering = int((df["accel_lat_mps2"].abs() >= CORNER_THR).sum())

    per_100km = (100.0 / max(dist_km, 1e-6))
    hard_brake_rate_100km = hard_brakes * per_100km
    harsh_accel_rate_100km = harsh_accels * per_100km
    corner_rate_100km = cornering * per_100km

    # speeding exposure proxy
    sp_lim = df.apply(_speed_limit, axis=1)
    speeding_exposure = float((df["speed_mps"] > sp_lim).mean())  # fraction of time

    # shares
    night_mile_share = float((df["time_of_day"] == "night").mean())
    rain_mile_share = float((df["weather"] == "rain").mean())
    idle_share = float((df["speed_mps"] <= IDLE_SPEED_THR).mean())

    # speed stats
    speed = df["speed_mps"].to_numpy()
    avg_speed = float(speed.mean())
    p50_speed = float(np.percentile(speed, 50))
    p95_speed = float(np.percentile(speed, 95))
    std_speed = float(speed.std(ddof=0))

    # jerk stats (magnitude)
    jerk = np.abs(df["jerk_mps3"].to_numpy())
    jerk_mean = float(jerk.mean())
    jerk_p95 = float(np.percentile(jerk, 95))

    out = dict(
        trip_id=trip_id,
        driver_id=driver_id,
        n_records=int(len(df)),
        hz=float(hz),
        duration_sec=float(len(df) * dt),
        dist_km=float(dist_km),

        hard_brakes=hard_brakes,
        harsh_accels=harsh_accels,
        cornering_events=cornering,

        hard_brake_rate_100km=float(hard_brake_rate_100km),
        harsh_accel_rate_100km=float(harsh_accel_rate_100km),
        corner_rate_100km=float(corner_rate_100km),

        speeding_exposure=float(speeding_exposure),
        night_mile_share=float(night_mile_share),
        rain_mile_share=float(rain_mile_share),
        idle_share=float(idle_share),

        avg_speed=float(avg_speed),
        p50_speed=float(p50_speed),
        p95_speed=float(p95_speed),
        std_speed=float(std_speed),

        jerk_mean=float(jerk_mean),
        jerk_p95=float(jerk_p95),
    )
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to trip JSONL (from simulate_stream.py)")
    ap.add_argument("--outdir", default="data/derived", help="Output directory for CSV/JSON")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    features = featurize_trip(in_path)

    # Write single-row CSV and JSON
    csv_path = out_dir / f"{features['trip_id']}_features.csv"
    json_path = out_dir / f"{features['trip_id']}_features.json"

    pd.DataFrame([features]).to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(features, f, indent=2)

    print(f"Wrote:\n  {csv_path}\n  {json_path}")

if __name__ == "__main__":
    main()

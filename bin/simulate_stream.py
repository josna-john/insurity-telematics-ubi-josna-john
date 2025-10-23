import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import json
import math
import os
import random
import time
from datetime import datetime, timezone


# Simple, self-contained simulator (no external deps)
TIME_OF_DAY_BUCKETS = [("night", 0, 6), ("morning", 6, 12), ("midday", 12, 17), ("evening", 17, 21), ("night", 21, 24)]
ROAD_TYPES = ["highway", "city", "rural"]
WEATHER = ["clear", "rain", "snow", "fog"]

def bucket_time_of_day(dt: datetime) -> str:
    hour = dt.hour
    for name, start, end in TIME_OF_DAY_BUCKETS:
        if start <= hour < end:
            return name
    return "night"

def random_walk(val, step, lo, hi):
    val += random.uniform(-step, step)
    return max(lo, min(hi, val))

def simulate_record(t0, i, hz, mode, driver_id, trip_id):
    dt = datetime.fromtimestamp(t0 + i / hz, tz=timezone.utc)
    tod = bucket_time_of_day(dt)
    road = random.choice(ROAD_TYPES)
    weather = random.choices(WEATHER, weights=[0.8, 0.15, 0.03, 0.02])[0]

    # base dynamics per mode
    if mode == "smooth":
        speed = random.uniform(12, 28)  # m/s ~ 2763 mph
        accel_long = random.uniform(-0.5, 0.5)
        accel_lat = random.uniform(-0.2, 0.2)
        jerk = random.uniform(-0.5, 0.5)
    elif mode == "aggressive":
        speed = random.uniform(5, 33)
        # occasional harsh events
        accel_long = random.uniform(-3.5, 2.5) if random.random() < 0.07 else random.uniform(-1.0, 1.0)
        accel_lat = random.uniform(-3.0, 3.0) if random.random() < 0.05 else random.uniform(-0.8, 0.8)
        jerk = random.uniform(-5.0, 5.0)
    else:  # normal
        speed = random.uniform(8, 30)
        accel_long = random.uniform(-1.0, 1.0)
        accel_lat = random.uniform(-0.5, 0.5)
        jerk = random.uniform(-1.5, 1.5)

    # simple heading/gps drift (not geographically accurate)
    heading = (i * (random.uniform(0.05, 0.2))) % 360
    # seed a pseudo gpx around a center
    lat0, lon0 = 32.7157, -117.1611  # San Diego-ish
    gps_lat = lat0 + math.sin(i / 300.0) * 0.01 + random.uniform(-1e-4, 1e-4)
    gps_lon = lon0 + math.cos(i / 300.0) * 0.01 + random.uniform(-1e-4, 1e-4)

    rec = {
        "timestamp": dt.isoformat(),
        "driver_id": driver_id,
        "trip_id": trip_id,
        "speed_mps": round(speed, 3),
        "accel_long_mps2": round(accel_long, 3),
        "accel_lat_mps2": round(accel_lat, 3),
        "jerk_mps3": round(jerk, 3),
        "heading_deg": round(heading, 2),
        "gps_lat": round(gps_lat, 6),
        "gps_lon": round(gps_lon, 6),
        "time_of_day": tod,
        "road_type": road,
        "weather": weather,
    }
    return rec

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="normal", choices=["normal", "smooth", "aggressive"])
    parser.add_argument("--duration", type=int, default=60, help="Duration (seconds)")
    parser.add_argument("--hz", type=int, default=10, help="Samples per second")
    parser.add_argument("--out", default="data/samples/trip.jsonl", help="Output file; use '-' for stdout")
    parser.add_argument("--realtime", action="store_true", help="Sleep to emulate realtime emission")
    parser.add_argument("--driver-id", default="drv_001")
    parser.add_argument("--trip-id", default="trip_001")
    args = parser.parse_args()

    n = args.duration * args.hz
    t0 = time.time()

    # ensure directory
    if args.out != "-":
        Path(os.path.dirname(args.out)).mkdir(parents=True, exist_ok=True)

    sink = sys.stdout if args.out == "-" else open(args.out, "w", encoding="utf-8")

    try:
        for i in range(n):
            rec = simulate_record(t0, i, args.hz, args.mode, args.driver_id, args.trip_id)
            sink.write(json.dumps(rec) + "\n")
            if args.realtime:
                time.sleep(1.0 / args.hz)
    finally:
        if sink is not sys.stdout:
            sink.close()

    print(f"wrote {n} records to {args.out}", file=sys.stderr)

if __name__ == "__main__":
    main()

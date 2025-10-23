from __future__ import annotations
from typing import Dict, Iterable, List, Tuple
import json
from datetime import datetime
from pathlib import Path

REQUIRED_FIELDS = [
    "timestamp","driver_id","trip_id","speed_mps","accel_long_mps2","accel_lat_mps2",
    "jerk_mps3","heading_deg","gps_lat","gps_lon","time_of_day","road_type","weather",
]

# Basic enums (POC)
TIME_OF_DAY = {"morning","midday","evening","night"}
ROAD_TYPES = {"highway","city","rural"}
WEATHER = {"clear","rain","snow","fog"}

def _parse_ts(ts: str) -> datetime:
    # Accept ISO 8601 with or without timezone suffix
    return datetime.fromisoformat(ts.replace("Z","+00:00"))  # tolerate 'Z'

def validate_record(rec: Dict) -> Tuple[bool, str]:
    for k in REQUIRED_FIELDS:
        if k not in rec:
            return False, f"missing field: {k}"
    # Type checks (lightweight)
    try:
        _ = _parse_ts(rec["timestamp"])
        float(rec["speed_mps"]); float(rec["accel_long_mps2"]); float(rec["accel_lat_mps2"])
        float(rec["jerk_mps3"]); float(rec["heading_deg"]); float(rec["gps_lat"]); float(rec["gps_lon"])
    except Exception as e:
        return False, f"type conversion error: {e}"
    if rec["time_of_day"] not in TIME_OF_DAY:
        return False, f"bad time_of_day: {rec['time_of_day']}"
    if rec["road_type"] not in ROAD_TYPES:
        return False, f"bad road_type: {rec['road_type']}"
    if rec["weather"] not in WEATHER:
        return False, f"bad weather: {rec['weather']}"
    return True, ""

def load_jsonl(path: str | Path, validate: bool = True) -> List[Dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            rec = json.loads(line)
            if validate:
                ok, msg = validate_record(rec)
                if not ok:
                    raise ValueError(f"validation failed at line {i}: {msg}")
            rows.append(rec)
    if not rows:
        raise ValueError("no records loaded")
    return rows

def infer_hz(timestamps: Iterable[str]) -> float:
    ts = [_parse_ts(x).timestamp() for x in timestamps]
    if len(ts) < 3:
        return 1.0
    diffs = [ts[i+1]-ts[i] for i in range(len(ts)-1)]
    diffs = [d for d in diffs if d > 0]
    if not diffs:
        return 1.0
    diffs_sorted = sorted(diffs)
    mid = diffs_sorted[len(diffs_sorted)//2]  # median
    if mid <= 0:
        return 1.0
    return round(1.0/mid, 3)

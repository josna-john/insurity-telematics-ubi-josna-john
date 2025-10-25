from __future__ import annotations
from typing import Dict, Iterable, List, Tuple
import json
from datetime import datetime
from pathlib import Path

"""
Utilities for loading and validating telematics JSONL data.

Capabilities:
- Lightweight schema validation for per-record telematics dictionaries.
- Robust JSON Lines loading with optional validation.
- Sample rate (Hz) inference from timestamp sequences.
"""

REQUIRED_FIELDS = [
    "timestamp", "driver_id", "trip_id", "speed_mps", "accel_long_mps2", "accel_lat_mps2",
    "jerk_mps3", "heading_deg", "gps_lat", "gps_lon", "time_of_day", "road_type", "weather",
]

TIME_OF_DAY = {"morning", "midday", "evening", "night"}
ROAD_TYPES = {"highway", "city", "rural"}
WEATHER = {"clear", "rain", "snow", "fog"}


def _parse_ts(ts: str) -> datetime:
    """
    Parse an ISO-8601 timestamp, accepting a trailing 'Z' as UTC.

    Args:
        ts: Timestamp string (e.g., "2025-01-01T12:34:56Z" or with explicit offset).

    Returns:
        datetime: A timezone-aware datetime.
    """
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def validate_record(rec: Dict) -> Tuple[bool, str]:
    """
    Validate a single telematics record against required fields and types.

    Checks:
      - Presence of all REQUIRED_FIELDS.
      - Parsable timestamp.
      - Numeric types for core kinematic/GPS fields.
      - Membership of categorical fields in allowed sets.

    Args:
        rec: Record dictionary.

    Returns:
        (ok, message): ok=True if valid; otherwise False and a brief reason.
    """
    for k in REQUIRED_FIELDS:
        if k not in rec:
            return False, f"missing field: {k}"
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
    """
    Load a JSON Lines file of telematics records, optionally validating each row.

    Args:
        path: Filesystem path to the JSONL file.
        validate: If True, enforce schema/type checks on each record.

    Returns:
        List[Dict]: Parsed records.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If validation fails at any line or no records are found.
    """
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
    """
    Infer sampling frequency (Hz) from a sequence of ISO-8601 timestamps.

    Uses the median positive inter-arrival time and returns its reciprocal.
    Falls back to 1.0 Hz when there is insufficient or non-increasing data.

    Args:
        timestamps: Iterable of timestamp strings.

    Returns:
        float: Estimated sampling frequency, rounded to 3 decimals.
    """
    ts = [_parse_ts(x).timestamp() for x in timestamps]
    if len(ts) < 3:
        return 1.0
    diffs = [ts[i + 1] - ts[i] for i in range(len(ts) - 1)]
    diffs = [d for d in diffs if d > 0]
    if not diffs:
        return 1.0
    diffs_sorted = sorted(diffs)
    mid = diffs_sorted[len(diffs_sorted) // 2]
    if mid <= 0:
        return 1.0
    return round(1.0 / mid, 3)

from __future__ import annotations
from typing import List

# Map feature name -> constraint: +1 (increasing), -1 (decreasing), 0 (no constraint)
RULES = {
    # stricly risk-increasing
    "hard_brake_rate_100km": +1,
    "harsh_accel_rate_100km": +1,
    "corner_rate_100km": +1,
    "p95_speed": +1,
    "std_speed": +1,
    "speeding_exposure": +1,
    "jerk_mean": +1,
    "jerk_p95": +1,
    # typically risk-decreasing
    "idle_share": -1,
    # leave others unconstrained unless you have policy guidance
}

def make_constraints(feat_names: List[str]) -> List[int]:
    return [RULES.get(name, 0) for name in feat_names]

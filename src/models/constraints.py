from __future__ import annotations
from typing import List

"""
Monotonicity constraints for risk model features.

RULES maps feature names to monotone directions:
    +1 => prediction must be non-decreasing as the feature increases
    -1 => prediction must be non-increasing as the feature increases
     0 => no constraint

These constraints can be passed to tree-based learners that support
monotonicity (e.g., CatBoost, LightGBM) to enforce domain-aligned behavior.
"""

# Map feature name -> constraint: +1 (increasing), -1 (decreasing), 0 (no constraint)
RULES = {
    "hard_brake_rate_100km": +1,
    "harsh_accel_rate_100km": +1,
    "corner_rate_100km": +1,
    "p95_speed": +1,
    "std_speed": +1,
    "speeding_exposure": +1,
    "jerk_mean": +1,
    "jerk_p95": +1,
    "idle_share": -1,
    # Unlisted features default to 0 (no constraint)
}


def make_constraints(feat_names: List[str]) -> List[int]:
    """
    Produce a constraint vector aligned to a given feature order.

    Args:
        feat_names: Ordered list of feature names used by the model.

    Returns:
        List[int]: Constraint values (+1, -1, or 0) in the same order as feat_names.
    """
    return [RULES.get(name, 0) for name in feat_names]

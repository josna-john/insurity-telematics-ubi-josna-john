from __future__ import annotations
from typing import List, Dict

"""
Gamification badge assignment from trip-level features.

Badges are awarded using simple threshold rules on safety-related features:
  - Smooth Operator: low hard_brake_rate_100km
  - Speed Limit Hero: low speeding_exposure
  - Gentle Handling: low jerk_p95
  - Daylight Driver: low night_mile_share

Each badge can be awarded at "gold" or "silver" tiers.
"""

THR = {
    "hard_brake_rate_100km": {"gold": 2, "silver": 5},
    "speeding_exposure": {"gold": 0.02, "silver": 0.05},
    "jerk_p95": {"gold": 1.2, "silver": 1.8},
    "night_mile_share": {"gold": 0.05, "silver": 0.10},  # lower is safer
}


def make_badges(feats: Dict) -> List[Dict]:
    """
    Derive gamification badges based on trip-level features.

    Args:
        feats: Feature mapping produced by featurize_trip().

    Returns:
        List[Dict]: Zero or more badges with fields:
            - name: Badge display name.
            - tier: "gold" | "silver".
            - reason: Short rationale tied to threshold(s).
    """
    badges: List[Dict] = []

    # Smooth Operator (few hard brakes)
    hbr = feats.get("hard_brake_rate_100km", 999.0)
    if hbr <= THR["hard_brake_rate_100km"]["gold"]:
        badges.append({"name": "Smooth Operator", "tier": "gold", "reason": f"hard_brake_rate_100km  {THR['hard_brake_rate_100km']['gold']}"})
    elif hbr <= THR["hard_brake_rate_100km"]["silver"]:
        badges.append({"name": "Smooth Operator", "tier": "silver", "reason": f"hard_brake_rate_100km  {THR['hard_brake_rate_100km']['silver']}"})

    # Speed Limit Hero (low speeding exposure)
    sp = feats.get("speeding_exposure", 1.0)
    if sp <= THR["speeding_exposure"]["gold"]:
        badges.append({"name": "Speed Limit Hero", "tier": "gold", "reason": f"speeding_exposure  {THR['speeding_exposure']['gold']}"})
    elif sp <= THR["speeding_exposure"]["silver"]:
        badges.append({"name": "Speed Limit Hero", "tier": "silver", "reason": f"speeding_exposure  {THR['speeding_exposure']['silver']}"})

    # Gentle Handling (low jerk)
    jp = feats.get("jerk_p95", 9e9)
    if jp <= THR["jerk_p95"]["gold"]:
        badges.append({"name": "Gentle Handling", "tier": "gold", "reason": f"jerk_p95 = {THR['jerk_p95']['gold']}"})
    elif jp <= THR["jerk_p95"]["silver"]:
        badges.append({"name": "Gentle Handling", "tier": "silver", "reason": f"jerk_p95 = {THR['jerk_p95']['silver']}"})

    # Daylight Driver (low night share)
    night = feats.get("night_mile_share", 1.0)
    if night <= THR["night_mile_share"]["gold"]:
        badges.append({"name": "Daylight Driver", "tier": "gold", "reason": f"night_mile_share = {THR['night_mile_share']['gold']}"})
    elif night <= THR["night_mile_share"]["silver"]:
        badges.append({"name": "Daylight Driver", "tier": "silver", "reason": f"night_mile_share = {THR['night_mile_share']['silver']}"})

    return badges

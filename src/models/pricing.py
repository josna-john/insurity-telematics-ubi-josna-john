from __future__ import annotations
from math import exp, log

"""
Pricing utilities.

Implements a GLM-style mapping from a bounded risk score in [0,1] to a
premium factor using a logit link, with guardrails (floor/cap) and
multipliers for traditional rating factors (e.g., territory, vehicle).
"""


def _logit(x: float, eps: float = 1e-6) -> float:
    """
    Numerically safe logit transform.

    Args:
        x: Value expected in [0, 1].
        eps: Small epsilon to avoid log(0) at the boundaries.

    Returns:
        The log-odds: log(x / (1 - x)).
    """
    x = min(1 - eps, max(eps, x))
    return log(x / (1 - x))


def price_from_risk(
    risk_score: float,
    base_premium: float = 100.0,
    floor: float = 0.75,
    cap: float = 1.50,
    slope: float = 1.75,
    intercept: float = 0.0,
    territory_mult: float = 1.0,
    vehicle_mult: float = 1.0,
) -> dict:
    """
    Map a risk score to a premium factor and premium via a logit-linked curve.

    The transformation follows:
        factor_unclamped = exp(intercept + slope * logit(risk_score))
        premium_factor   = clamp(factor_unclamped, [floor, cap]) * territory_mult * vehicle_mult
        premium          = base_premium * premium_factor

    Args:
        risk_score: Model-predicted risk in [0, 1].
        base_premium: Starting premium before behavior-based adjustment.
        floor: Minimum allowed premium factor (max discount).
        cap: Maximum allowed premium factor (max surcharge).
        slope: Elasticity; larger values increase curvature/sensitivity.
        intercept: Horizontal shift in logit space (sets pivot where factor≈1).
        territory_mult: Traditional rating multiplier (territory/region).
        vehicle_mult: Traditional rating multiplier (vehicle characteristics).

    Returns:
        dict with:
            - risk_score: Echo of the input risk.
            - premium_factor: Final factor after clamping and multipliers.
            - premium: base_premium * premium_factor.
            - params: Echo of pricing parameters used.
    """
    z = intercept + slope * _logit(risk_score)
    factor = exp(z)
    factor = min(cap, max(floor, factor))
    factor *= territory_mult * vehicle_mult
    premium = base_premium * factor
    return {
        "risk_score": float(risk_score),
        "premium_factor": float(factor),
        "premium": float(premium),
        "params": {
            "base_premium": base_premium,
            "floor": floor,
            "cap": cap,
            "slope": slope,
            "intercept": intercept,
            "territory_mult": territory_mult,
            "vehicle_mult": vehicle_mult,
        },
    }

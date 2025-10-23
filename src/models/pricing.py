from __future__ import annotations
from math import exp, log

def _logit(x: float, eps: float = 1e-6) -> float:
    x = min(1 - eps, max(eps, x))
    return log(x / (1 - x))

def price_from_risk(
    risk_score: float,
    base_premium: float = 100.0,
    floor: float = 0.75,   # min factor (25% discount cap)
    cap: float = 1.50,     # max factor (50% surcharge cap)
    slope: float = 1.75,   # elasticity of pricing to risk
    intercept: float = 0.0,
    territory_mult: float = 1.0,
    vehicle_mult: float = 1.0,
) -> dict:
    """Map risk_score -> premium_factor via a logit link, clamp to [floor, cap], then apply multipliers."""
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
            "base_premium": base_premium, "floor": floor, "cap": cap,
            "slope": slope, "intercept": intercept,
            "territory_mult": territory_mult, "vehicle_mult": vehicle_mult,
        },
    }

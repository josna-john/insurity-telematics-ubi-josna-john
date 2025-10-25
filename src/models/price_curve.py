from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.models.pricing import price_from_risk

"""
Price curve generator for the GLM-style mapping.

This CLI samples risk scores in [0.01, 0.99], applies `price_from_risk`
to compute (premium_factor, premium) under the chosen guardrails, writes a
CSV table, and saves two reference plots:
  • price_curve_factor.png  – premium factor vs. risk with floor/cap guides
  • price_curve_premium.png – premium vs. risk with an annotation where factor≈1
"""


def main():
    """
    Generate CSV and plots showing premium factor/premium as a function of risk.

    Arguments (via argparse):
        --base-premium (float): Starting premium before behavior (default: 120).
        --floor (float): Minimum factor (max discount), e.g. 0.80.
        --cap (float): Maximum factor (max surcharge), e.g. 1.40.
        --slope (float): Pricing elasticity controlling curvature.
        --outdir (str): Output directory for CSV/plots (default: docs/pricing).

    Side Effects:
        Writes:
          • <outdir>/price_curve.csv
          • <outdir>/price_curve_factor.png
          • <outdir>/price_curve_premium.png
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-premium", type=float, default=120.0)
    ap.add_argument("--floor", type=float, default=0.80)
    ap.add_argument("--cap", type=float, default=1.40)
    ap.add_argument("--slope", type=float, default=1.20)
    ap.add_argument("--outdir", default="docs/pricing")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # Build curve
    risks = np.linspace(0.01, 0.99, 99)
    rows = []
    for r in risks:
        p = price_from_risk(
            r,
            base_premium=args.base_premium,
            floor=args.floor,
            cap=args.cap,
            slope=args.slope,
        )
        rows.append({"risk": r, "premium_factor": p["premium_factor"], "premium": p["premium"]})
    df = pd.DataFrame(rows)

    # Find risk where premium ≈ base (factor ≈ 1)
    idx = (df["premium"] - args.base_premium).abs().idxmin()
    x0  = float(df.loc[idx, "risk"])
    y0  = float(df.loc[idx, "premium"])

    df.to_csv(outdir / "price_curve.csv", index=False)

    # Plot 1: premium factor vs risk with guardrail labels
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df["risk"], df["premium_factor"])
    ax.set_xlabel("Risk score"); ax.set_ylabel("Premium factor")
    ax.set_title("Pricing Sensitivity (factor vs. risk)")
    ax.axhline(args.floor, linestyle="--")
    ax.axhline(args.cap, linestyle="--")
    ax.text(0.01, args.floor + 0.005, f"floor={args.floor:.2f}", fontsize=9)
    ax.text(0.01, args.cap   + 0.005, f"cap={args.cap:.2f}", fontsize=9)
    fig.tight_layout()
    fig.savefig(outdir / "price_curve_factor.png", dpi=150)
    plt.close(fig)

    # Plot 2: premium vs risk with min/max annotations
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df["risk"], df["premium"])
    ax.scatter([x0], [y0], s=30)
    ax.annotate(
        f"factor≈1 @ risk≈{x0:.2f}",
        (x0, y0), xytext=(x0 + 0.03, y0 + 5),
        arrowprops=dict(arrowstyle="->"), fontsize=9,
    )

    ax.set_xlabel("Risk score"); ax.set_ylabel("Premium ($)")
    ax.set_title(f"Premium vs. risk (base={args.base_premium:.1f})")
    min_idx = int(df["premium"].idxmin()); max_idx = int(df["premium"].idxmax())
    min_x, min_y = df.loc[min_idx, ["risk", "premium"]]
    max_x, max_y = df.loc[max_idx, ["risk", "premium"]]
    ax.text(min_x + 0.02, min_y + 1.0, f"min=${min_y:.2f}", fontsize=9)
    ax.text(max_x - 0.20, max_y - 1.0, f"max=${max_y:.2f}", fontsize=9)
    fig.tight_layout()
    fig.savefig(outdir / "price_curve_premium.png", dpi=150)
    plt.close(fig)

    print(f"Wrote {outdir/'price_curve.csv'}, price_curve_factor.png, price_curve_premium.png")


if __name__ == "__main__":
    main()

from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.models.pricing import price_from_risk

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-premium", type=float, default=100.0)
    ap.add_argument("--floor", type=float, default=0.75)
    ap.add_argument("--cap", type=float, default=1.50)
    ap.add_argument("--slope", type=float, default=1.75)
    ap.add_argument("--outdir", default="docs/pricing")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    risks = np.linspace(0.01, 0.99, 99)
    rows = []
    for r in risks:
        p = price_from_risk(r, base_premium=args.base_premium, floor=args.floor, cap=args.cap, slope=args.slope)
        rows.append({"risk": r, "premium_factor": p["premium_factor"], "premium": p["premium"]})
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "price_curve.csv", index=False)

    # Plot premium factor vs risk
    plt.figure(figsize=(6,4))
    plt.plot(df["risk"], df["premium_factor"])
    plt.axhline(args.floor, linestyle="--")
    plt.axhline(args.cap, linestyle="--")
    plt.xlabel("Risk score")
    plt.ylabel("Premium factor")
    plt.title("Pricing Sensitivity (factor vs. risk)")
    plt.tight_layout()
    plt.savefig(outdir / "price_curve_factor.png", dpi=150)

    # Plot premium vs risk
    plt.figure(figsize=(6,4))
    plt.plot(df["risk"], df["premium"])
    plt.xlabel("Risk score")
    plt.ylabel("Premium ($)")
    plt.title(f"Premium vs. risk (base={args.base_premium})")
    plt.tight_layout()
    plt.savefig(outdir / "price_curve_premium.png", dpi=150)

    print(f"Wrote {outdir/'price_curve.csv'}, factor.png, premium.png")

if __name__ == "__main__":
    main()

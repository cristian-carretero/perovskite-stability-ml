"""
Module: src/audit_pff_distribution.py
Description: Diagnostic for the pseudo-FF (pFF) distribution produced by the
clustering module.

Background
----------
`pseudo_FF` is computed in module 03 on shape-normalized curves. It is NOT
a physical fill factor: it measures the "squareness" of a J-V curve after
both axes have been scaled to [0,1]. Degenerate curves (essentially step
functions) can produce pFF close to 1.0, which is not physically meaningful.

This script reports the empirical distribution of pFF across all curves,
per cell, to identify whether a clean separation exists between the bulk
of curves and the outlier tail. If a clear gap exists, the T80 tracker
(module 06) uses a cap to exclude the outliers before daily aggregation.

Usage
-----
    python -m src.audit_pff_distribution
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.config import FILE_JV_LABELED

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("Audit-pFF")


def main() -> None:
    logger.info(f"Loading {FILE_JV_LABELED.name}...")
    df = pd.read_parquet(FILE_JV_LABELED, columns=["cell_name", "curve", "pseudo_FF"])
    df = df.drop_duplicates(["cell_name", "curve"])

    print("\n" + "=" * 70)
    print(" GLOBAL pFF DISTRIBUTION")
    print("=" * 70)
    print(df["pseudo_FF"].describe(percentiles=[0.5, 0.9, 0.95, 0.99, 0.999]).round(4))

    print("\nFraction of curves above thresholds:")
    for k in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98]:
        n = int((df["pseudo_FF"] > k).sum())
        print(f"  pFF > {k}: {n:>5,}  ({100 * n / len(df):>7.3f}%)")

    print("\n" + "=" * 70)
    print(" pFF DISTRIBUTION PER CELL")
    print("=" * 70)
    per_cell = df.groupby("cell_name")["pseudo_FF"].agg(
        n="count",
        median="median",
        p99=lambda x: x.quantile(0.99),
        p999=lambda x: x.quantile(0.999),
        max="max",
    )
    print(per_cell.round(4).to_string())

    # Gap analysis: look for a natural cutoff between bulk and outliers.
    print("\n" + "=" * 70)
    print(" GAP ANALYSIS (natural cutoff detection)")
    print("=" * 70)
    values = df["pseudo_FF"].dropna().to_numpy()
    sorted_vals = np.sort(values)

    # Largest gap in the upper tail (above p95)
    p95 = np.quantile(values, 0.95)
    upper = sorted_vals[sorted_vals >= p95]
    if len(upper) > 1:
        diffs = np.diff(upper)
        idx_max_gap = int(np.argmax(diffs))
        gap_lo = upper[idx_max_gap]
        gap_hi = upper[idx_max_gap + 1]
        print(f"  Largest gap in [p95, max]: {gap_lo:.4f} -> {gap_hi:.4f} "
              f"(width {gap_hi - gap_lo:.4f})")
        print(f"  Curves above the gap: {int((values > gap_hi).sum())}")
        print(f"  Suggested cap: {(gap_lo + gap_hi) / 2:.4f}")

    print("=" * 70)


if __name__ == "__main__":
    main()
"""
Module: src/rul_temporal_baseline.py
Description: Naive temporal baseline for the RUL engine.

             Evaluates the simplest possible prognostic model:

                 RUL_pred(t) = max(0, t_ref - t)

             i.e. a fixed "expected lifetime" minus the current exposure day.
             No XGBoost, no kinematics, no weather. Just the clock.

             Purpose: establish a lower bound against which any damage-aware
             model can be compared. If a sophisticated engine does not beat
             this trivial baseline, its predictive value is questionable.

             Outputs:
               - outputs/diagnostics/rul_temporal_baseline.parquet
               - outputs/diagnostics/rul_temporal_baseline.txt
"""

from __future__ import annotations

import importlib
import logging
import sys
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from src.config import (
    BURN_IN_DAYS,
    FILE_HEALTHY_COHORT,
    FILE_T80_TRUTH,
    FILE_SCREENING_ARTIFACTS,
    DIAGNOSTICS_RUL_DIR,
)

# ------------------------------------------------------------------------------
# Dynamic import of the RUL module (its filename starts with a digit).
# ------------------------------------------------------------------------------
rul = importlib.import_module("src.08_mppt_rul_forecasting")
build_rul_matrix = rul.build_rul_matrix
ANCHOR_SPACING = rul.ANCHOR_SPACING


# ==============================================================================
# Logging + report paths
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("RUL-Temporal-Baseline")

REPORT_PATH = DIAGNOSTICS_RUL_DIR / "rul_temporal_baseline.txt"
RESULTS_PATH = DIAGNOSTICS_RUL_DIR / "rul_temporal_baseline.parquet"


class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)
        return len(data)

    def flush(self):
        for s in self._streams:
            s.flush()


def _coerce_optional_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(f):
        return None
    return f


# ==============================================================================
# Anchor reconstruction (mirrors the production backtesting protocol)
# ==============================================================================
def build_anchor_table(
    df_daily: pd.DataFrame,
    t80_metrics: pd.DataFrame,
    healthy_cohort: list,
) -> pd.DataFrame:
    """
    Reconstruct the (cell, anchor_day, true_survival_days) tuples exactly as
    the production engine sees them during backtesting, but WITHOUT training
    XGBoost or running any simulation.

    Anchors are spaced ANCHOR_SPACING days apart, starting at BURN_IN_DAYS.
    The final anchor is always the last observed exposure day for the cell.
    """
    rows = []
    for cell in healthy_cohort:
        cell_data = df_daily[df_daily["cell_name"] == cell].sort_values("Exposure_Days")
        if cell_data.empty:
            continue

        true_survival = (
            _coerce_optional_float(t80_metrics.loc[cell, "survival_days_pce"])
            if cell in t80_metrics.index
            else None
        )

        max_days = cell_data["Exposure_Days"].max()
        anchors = list(range(int(BURN_IN_DAYS), int(max_days) + 1, ANCHOR_SPACING))
        if int(max_days) not in anchors:
            anchors.append(int(max_days))

        for anchor in anchors:
            hist_cutoff = cell_data[cell_data["Exposure_Days"] <= anchor]
            if hist_cutoff.empty:
                continue
            rows.append({
                "cell_name": cell,
                "Anchor_Day": float(hist_cutoff.iloc[-1]["Exposure_Days"]),
                "True_Survival_Days": true_survival,
            })

    return pd.DataFrame(rows)


# ==============================================================================
# Temporal baseline evaluation
# ==============================================================================
def evaluate_temporal_baseline(
    anchors: pd.DataFrame,
    t_ref_grid: list,
) -> pd.DataFrame:
    print("\n" + "=" * 90)
    print(" TEMPORAL BASELINE — RUL_pred(t) = max(0, t_ref - t)")
    print("=" * 90)
    print(" No XGBoost, no physics, no weather. Just the clock.")
    print(f" Evaluating {len(t_ref_grid)} candidate t_ref values on "
          f"{len(anchors)} (cell, anchor) points...\n")

    # Filter to valid points (RUL_real > 0)
    df = anchors.copy()
    df["RUL_Real"] = df["True_Survival_Days"] - df["Anchor_Day"]
    valid = df[df["RUL_Real"] > 0].reset_index(drop=True)

    results = []
    for t_ref in t_ref_grid:
        preds = np.maximum(0.0, t_ref - valid["Anchor_Day"].to_numpy())
        mae_global = float(np.mean(np.abs(preds - valid["RUL_Real"].to_numpy())))

        # Per-cell MAE (simple loop, robust across pandas versions)
        per_cell = {}
        for cell in valid["cell_name"].unique():
            mask = valid["cell_name"] == cell
            anchor_vals = np.asarray(valid.loc[mask, "Anchor_Day"], dtype=float)
            real_vals = np.asarray(valid.loc[mask, "RUL_Real"], dtype=float)
            sub_preds = np.maximum(0.0, t_ref - anchor_vals)
            sub_real = real_vals
            per_cell[str(cell)] = float(np.mean(np.abs(sub_preds - sub_real)))

        results.append({
            "t_ref": float(t_ref),
            "mae_global": mae_global,
            "mae_by_cell": per_cell,
            "n": int(len(valid)),
        })
        print(f"  t_ref = {t_ref:>5.1f} d | MAE_global = {mae_global:6.3f} d")

    df_results = pd.DataFrame(results).sort_values("mae_global").reset_index(drop=True)
    best = df_results.iloc[0]

    print("\n" + "=" * 90)
    print(" TEMPORAL BASELINE — BEST")
    print("=" * 90)
    print(f"  Best t_ref  = {best['t_ref']:.1f} days")
    print(f"  MAE         = {best['mae_global']:.3f} days")
    print(f"  N           = {int(best['n'])}")
    print("\n  Per-cell MAE at best t_ref:")
    for c, v in best["mae_by_cell"].items():
        print(f"    {c:<15s} : {v:6.3f} d")
    print("=" * 90)

    # Comparison against the calibrated engine
    print("\n  Reference: calibrated RUL engine (W=1) sensor MAE = 2.63 d")
    delta = best["mae_global"] - 2.63
    if delta < 0:
        print(f"  → Temporal baseline BEATS the engine by {-delta:.2f} d  (⚠ engine underperforms a clock)")
    else:
        print(f"  → Temporal baseline is {delta:.2f} d WORSE than the engine")
    print("=" * 90)
    return df_results


# ==============================================================================
# Entry point
# ==============================================================================
def _run() -> None:
    logger.info("Loading inputs...")
    df_twin = pd.read_parquet(FILE_HEALTHY_COHORT)
    t80_metrics = pd.read_parquet(FILE_T80_TRUTH)
    healthy_cohort = joblib.load(FILE_SCREENING_ARTIFACTS)["healthy_cohort"]

    if "PCE_initial" not in df_twin.columns:
        df_twin = df_twin.merge(
            t80_metrics[["PCE_initial"]],
            left_on="cell_name", right_index=True, how="left",
        )

    df_daily = build_rul_matrix(df_twin, healthy_cohort)

    anchors = build_anchor_table(df_daily, t80_metrics, healthy_cohort)
    print(f"\n  Anchor points reconstructed: {len(anchors)}")
    print(f"  Cells: {sorted(anchors['cell_name'].unique().tolist())}")

    t_ref_grid = [0, 10, 20, 30, 35, 40, 45, 50, 52, 54, 55, 56, 58, 60, 65, 70, 80, 90]
    df_results = evaluate_temporal_baseline(anchors, t_ref_grid)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_results.drop(columns=["mae_by_cell"]).to_parquet(RESULTS_PATH, index=False)
    print(f"\n  Results table saved -> {RESULTS_PATH}")


def main() -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    original_stdout = sys.stdout
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        sys.stdout = _Tee(original_stdout, f)
        try:
            _run()
        finally:
            sys.stdout = original_stdout
    logger.info(f"Full report saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
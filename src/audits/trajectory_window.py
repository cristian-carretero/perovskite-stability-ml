"""
Module: src/audit_trajectory_window.py
Description: Windowing audit for module 09 (trajectory forecasting).

             Two hardcoded parameters control how much history the engine
             consumes before forecasting:

               - ANCHOR_DAY (default 14.0): the first day from which a
                 forecast is produced. Smaller = less history but earlier
                 prediction.

               - ROLLING_WINDOW (default 7): the window of the rolling
                 median used for Rolling_Irradiance, Rolling_Thermal_load,
                 and the smoothed state of each target parameter.

             The audit answers two operational questions:

               1. What is the minimum anchor day at which the engine still
                  performs within 1 SE of its best result? (mentor's query)

               2. Does the rolling window need to be 7, or would a shorter
                  window suffice / perform better?

             Design (pre-registered):

               Experiment A — Anchor sweep
                 ANCHOR_DAY      ∈ {3, 5, 7, 10, 14, 21, 28}
                 ROLLING_WINDOW  = 7 (production default, fixed)

               Experiment B — Rolling window sweep
                 ROLLING_WINDOW  ∈ {1, 2, 3, 5, 7, 10, 14}
                 ANCHOR_DAY      = 14 (production default, fixed)

             Metric: per-cell MAE on the first EVALUATION_HORIZON days
             of the forecast, averaged per parameter (mean of per-cell MAEs).
             Cells whose evaluation window is truncated (fewer than
             EVALUATION_HORIZON sensor days after the anchor) are excluded,
             matching the exclusion policy of module 09.

             Uncertainty: SE = std(per-cell MAE) / sqrt(N_cells). With N=4,
             effects smaller than 1 SE are indistinguishable from noise.

             Acceptability criterion (anchor sweep):
               An anchor A* is 'usable' for parameter p if:
                   N(A*) >= MIN_CELLS_FOR_VERDICT  AND
                   MAE(A*, p) <= MAE_best(p) + SE(p)
               where MAE_best(p) is also computed only over anchors with
               N >= MIN_CELLS_FOR_VERDICT.

             Why the N >= MIN_CELLS_FOR_VERDICT requirement:
               When a cell dies shortly after the anchor, module 09 excludes
               its truncated evaluation from the aggregate. If enough cells
               die, the surviving aggregate may be computed on N=2 or even
               N=1 cells, giving an artificially small SE and a spuriously
               low MAE. Anchors with such a small effective N are not
               reliable evidence for a 'best' or 'usable' decision, so they
               are filtered out of the verdict while still being reported in
               the raw sweep table for transparency.

             Read-only. Does not modify production artifacts.
"""

from __future__ import annotations

import contextlib
import importlib
import io
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import (
    DIAGNOSTICS_TRAJECTORY_DIR,
    EVALUATION_HORIZON,
    FILE_HEALTHY_COHORT,
    FILE_SCREENING_ARTIFACTS,
)

# Dynamic import (module filename starts with a digit).
traj = importlib.import_module("src.09_jv_mppt_trajectory_forecasting")

build_trajectory_matrix = traj.build_trajectory_matrix
run_loocv_evaluation = traj.run_loocv_evaluation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("AuditTrajectoryWindow")


# ==============================================================================
# CONFIGURATION
# ==============================================================================
ANCHOR_GRID: List[float] = [3.0, 5.0, 7.0, 10.0, 14.0, 21.0, 28.0]
ROLLING_WINDOW_GRID: List[int] = [1, 2, 3, 5, 7, 10, 14]
PRODUCTION_ANCHOR: float = 14.0
PRODUCTION_ROLLING_WINDOW: int = 7
TARGET_PARAMS: List[str] = ["PCE", "FF", "Jsc", "Voc"]

# Minimum number of cells required for an anchor (or a rolling window) to be
# considered in the 'best' / 'usable' verdict. Anchors whose effective N falls
# below this threshold are still reported in the sweep table but are excluded
# from the decision because their SE is unreliable.
MIN_CELLS_FOR_VERDICT: int = 3


# ==============================================================================
# SAFE NUMERIC COERCION
# ==============================================================================
def _f(value: Any) -> float:
    """Coerce any pandas/numpy scalar (or 0-d/1-d array) to a Python float."""
    if value is None:
        return float("nan")
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float, np.floating, np.integer)):
        return float(value)
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
        if arr.size == 0:
            return float("nan")
        return float(arr[0])
    except (TypeError, ValueError):
        return float("nan")


def _i(value: Any) -> int:
    """Coerce to int; returns 0 on failure."""
    f = _f(value)
    if np.isnan(f):
        return 0
    return int(f)


@contextlib.contextmanager
def _quiet_stdout():
    old = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = old


def _set_module_attr(module: Any, name: str, value: Any) -> None:
    """
    Assign a module-level attribute from outside the module.

    Python allows this at runtime, but static type checkers refuse to
    acknowledge assignments to imported names as if they were module
    attributes. setattr() bypasses the checker without changing behaviour.
    """
    setattr(module, name, value)


# ==============================================================================
# MAE EXTRACTION
# ==============================================================================
def _extract_mae_per_cell(
    records: List[dict],
    targets: List[str],
    horizon: int = EVALUATION_HORIZON,
    exclude_truncated: bool = True,
) -> Dict[str, Dict[str, float]]:
    """
    From the raw records returned by run_loocv_evaluation, compute per-cell
    MAE per parameter over the first `horizon` forecast days.

    When exclude_truncated is True (default), cells whose Forecast_Truncated
    flag is True are dropped entirely, matching the aggregate policy of
    module 09. This prevents artificially small effective N from entering
    the SE calculation.
    """
    df = pd.DataFrame(records)
    forecast = df[df["Phase"] == "Forecast"].copy()

    per_cell: Dict[str, Dict[str, float]] = {}
    for cell, sub in forecast.groupby("cell_name"):
        if exclude_truncated and bool(sub["Forecast_Truncated"].iloc[0]):
            continue
        sub_sorted = sub.sort_values("Exposure_Days").head(horizon)
        cell_str = str(cell)
        per_cell[cell_str] = {}
        for p in targets:
            a = np.asarray(sub_sorted[f"Actual_{p}"].to_numpy(), dtype=float)
            b = np.asarray(sub_sorted[f"Pred_{p}"].to_numpy(), dtype=float)
            mask = np.isfinite(a) & np.isfinite(b)
            if mask.sum() > 0:
                per_cell[cell_str][p] = float(np.mean(np.abs(a[mask] - b[mask])))
    return per_cell


def _aggregate_per_param(
    per_cell: Dict[str, Dict[str, float]],
    targets: List[str],
    config_value: float,
    config_name: str,
    healthy_cohort: List[str],
) -> List[Dict]:
    """Aggregate per-cell MAEs into per-parameter mean ± SE rows."""
    rows: List[Dict] = []
    for p in targets:
        vals = [per_cell[c][p] for c in per_cell if p in per_cell[c]]
        if not vals:
            continue
        mean_mae = float(np.mean(vals))
        se = (
            float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
            if len(vals) > 1 else float("nan")
        )
        row: Dict = {
            config_name: config_value,
            "parameter": p,
            "n_cells": len(vals),
            "mae_mean": mean_mae,
            "mae_se": se,
        }
        for c in healthy_cohort:
            row[f"mae_{c}"] = per_cell.get(c, {}).get(p, float("nan"))
        rows.append(row)
    return rows


# ==============================================================================
# EXPERIMENT A — ANCHOR SWEEP
# ==============================================================================
def run_anchor_sweep(
    df_healthy: pd.DataFrame,
    healthy_cohort: List[str],
    targets: List[str],
) -> pd.DataFrame:
    print("\n" + "=" * 88)
    print(" EXPERIMENT A — ANCHOR_DAY sweep (rolling window fixed at production)")
    print("=" * 88)
    print(f"  ANCHOR_DAY grid        : {ANCHOR_GRID}")
    print(f"  ROLLING_WINDOW (fixed) : {PRODUCTION_ROLLING_WINDOW}")
    print(f"  Targets                : {targets}")
    print(f"  Metric                 : per-cell MAE on first {EVALUATION_HORIZON} forecast days")
    print(f"  Truncated cells        : EXCLUDED (matches module 09 aggregate policy)")

    _set_module_attr(traj, "ROLLING_WINDOW", int(PRODUCTION_ROLLING_WINDOW))
    df_daily, _ = build_trajectory_matrix(df_healthy)

    old_anchor = _f(getattr(traj, "ANCHOR_DAY"))

    rows: List[Dict] = []
    try:
        for anchor in ANCHOR_GRID:
            print(f"\n  Evaluating anchor = {anchor:.1f} d ...")
            _set_module_attr(traj, "ANCHOR_DAY", float(anchor))
            with _quiet_stdout():
                records = run_loocv_evaluation(df_daily, healthy_cohort, targets)
            per_cell = _extract_mae_per_cell(records, targets)
            rows.extend(_aggregate_per_param(
                per_cell, targets, float(anchor), "anchor", healthy_cohort,
            ))
    finally:
        _set_module_attr(traj, "ANCHOR_DAY", old_anchor)

    return pd.DataFrame(rows)


# ==============================================================================
# EXPERIMENT B — ROLLING WINDOW SWEEP
# ==============================================================================
def run_rolling_sweep(
    df_healthy: pd.DataFrame,
    healthy_cohort: List[str],
    targets: List[str],
) -> pd.DataFrame:
    print("\n" + "=" * 88)
    print(" EXPERIMENT B — ROLLING_WINDOW sweep (anchor fixed at production)")
    print("=" * 88)
    print(f"  ROLLING_WINDOW grid    : {ROLLING_WINDOW_GRID}")
    print(f"  ANCHOR_DAY (fixed)     : {PRODUCTION_ANCHOR}")
    print(f"  Targets                : {targets}")
    print(f"  Metric                 : per-cell MAE on first {EVALUATION_HORIZON} forecast days")
    print(f"  Truncated cells        : EXCLUDED (matches module 09 aggregate policy)")

    old_rw = _i(getattr(traj, "ROLLING_WINDOW"))
    old_anchor = _f(getattr(traj, "ANCHOR_DAY"))

    rows: List[Dict] = []
    try:
        _set_module_attr(traj, "ANCHOR_DAY", float(PRODUCTION_ANCHOR))
        for rw in ROLLING_WINDOW_GRID:
            print(f"\n  Evaluating rolling_window = {rw} ...")
            _set_module_attr(traj, "ROLLING_WINDOW", int(rw))
            df_daily, _ = build_trajectory_matrix(df_healthy)
            with _quiet_stdout():
                records = run_loocv_evaluation(df_daily, healthy_cohort, targets)
            per_cell = _extract_mae_per_cell(records, targets)
            rows.extend(_aggregate_per_param(
                per_cell, targets, float(rw), "rolling_window", healthy_cohort,
            ))
    finally:
        _set_module_attr(traj, "ROLLING_WINDOW", old_rw)
        _set_module_attr(traj, "ANCHOR_DAY", old_anchor)

    return pd.DataFrame(rows)


# ==============================================================================
# REPORTING
# ==============================================================================
def print_sweep_table(
    df: pd.DataFrame,
    config_col: str,
    title: str,
    healthy_cohort: List[str],
) -> None:
    print("\n" + "=" * 88)
    print(f" {title}")
    print("=" * 88)
    header = (
        f"  {config_col:<15} {'param':<5} {'N':>3} {'MAE (d)':>10} "
        f"{'SE (d)':>9}  per-cell MAE"
    )
    print(header)
    print("  " + "-" * (len(header.strip()) + 20))
    for _, row in df.sort_values([config_col, "parameter"]).iterrows():
        parts: List[str] = []
        for c in healthy_cohort:
            col = f"mae_{c}"
            if col in df.columns:
                v = _f(row[col])
                if not np.isnan(v):
                    parts.append(f"{c[:8]}={v:.3f}")
        per_cell = " ".join(parts)
        print(
            f"  {_f(row[config_col]):<15.1f} {str(row['parameter']):<5} "
            f"{_i(row['n_cells']):>3} "
            f"{_f(row['mae_mean']):>10.4f} {_f(row['mae_se']):>9.4f}  {per_cell}"
        )


def print_anchor_verdict(df_anchor: pd.DataFrame) -> None:
    print("\n" + "=" * 88)
    print(" VERDICT — Minimum usable anchor day (mentor's question)")
    print("=" * 88)
    print("  Criterion: anchor A* is 'usable' for parameter p if")
    print(f"             N(A*) >= {MIN_CELLS_FOR_VERDICT}  AND")
    print("             MAE(A*, p) <= MAE_best(p) + 1 SE(p)")
    print(f"  where MAE_best(p) is computed only over anchors with "
          f"N >= {MIN_CELLS_FOR_VERDICT}.")
    print()
    print(f"  Anchors excluded from the verdict (N < {MIN_CELLS_FOR_VERDICT}):")
    excluded_rows = df_anchor[df_anchor["n_cells"] < MIN_CELLS_FOR_VERDICT]
    if excluded_rows.empty:
        print("    (none)")
    else:
        for _, r in excluded_rows.iterrows():
            print(
                f"    anchor = {_f(r['anchor']):>5.1f} d, param = {str(r['parameter']):<4} "
                f"(N = {_i(r['n_cells'])})"
            )

    min_anchors: List[float] = []
    print()
    for p in TARGET_PARAMS:
        sub = df_anchor[df_anchor["parameter"] == p].sort_values("anchor")
        if sub.empty:
            continue

        anchors_arr = np.asarray(sub["anchor"].to_numpy(), dtype=float)
        mae_arr = np.asarray(sub["mae_mean"].to_numpy(), dtype=float)
        se_arr = np.asarray(sub["mae_se"].to_numpy(), dtype=float)
        n_arr = np.asarray(sub["n_cells"].to_numpy(), dtype=float)

        # Only anchors with sufficient N are eligible to define 'best' and
        # to qualify as 'usable'.
        eligible = (n_arr >= float(MIN_CELLS_FOR_VERDICT)) & np.isfinite(mae_arr)
        if not eligible.any():
            print(f"  {p:<5}: no anchor with N >= {MIN_CELLS_FOR_VERDICT}. Skipped.")
            continue

        masked_mae = np.where(eligible, mae_arr, np.inf)
        best_idx = int(np.argmin(masked_mae))
        best_mae = float(mae_arr[best_idx])
        best_anchor = float(anchors_arr[best_idx])
        best_se_raw = float(se_arr[best_idx])
        best_se = best_se_raw if np.isfinite(best_se_raw) else 0.0
        threshold = best_mae + best_se

        usable_mask = eligible & (mae_arr <= threshold)
        if not usable_mask.any():
            print(f"  {p:<5}: no usable anchor found. Skipped.")
            continue

        a_min = float(np.min(anchors_arr[usable_mask]))
        min_anchors.append(a_min)
        print(
            f"  {p:<5}: best MAE = {best_mae:.4f} d @ anchor = {best_anchor:.0f} d "
            f"(N={_i(n_arr[best_idx])}); threshold = {threshold:.4f} d; "
            f"minimum usable = {a_min:.0f} d"
        )

    if min_anchors:
        overall = float(np.max(min_anchors))
        print(
            f"\n  Overall minimum usable anchor across all parameters: "
            f"{overall:.0f} days"
        )
        print(
            "  (The overall value is the max across parameters because every "
            "parameter must remain usable at the chosen anchor.)"
        )
    print("=" * 88)


def print_rolling_verdict(df_roll: pd.DataFrame) -> None:
    print("\n" + "=" * 88)
    print(" VERDICT — Rolling window")
    print("=" * 88)
    print(f"  Same N >= {MIN_CELLS_FOR_VERDICT} requirement applies.")
    print()
    for p in TARGET_PARAMS:
        sub = df_roll[df_roll["parameter"] == p].sort_values("rolling_window")
        if sub.empty:
            continue

        w_arr = np.asarray(sub["rolling_window"].to_numpy(), dtype=float)
        mae_arr = np.asarray(sub["mae_mean"].to_numpy(), dtype=float)
        se_arr = np.asarray(sub["mae_se"].to_numpy(), dtype=float)
        n_arr = np.asarray(sub["n_cells"].to_numpy(), dtype=float)

        eligible = (n_arr >= float(MIN_CELLS_FOR_VERDICT)) & np.isfinite(mae_arr)
        prod_mask = (w_arr == float(PRODUCTION_ROLLING_WINDOW)) & eligible
        if not prod_mask.any():
            print(f"  {p:<5}: production W has insufficient N. Skipped.")
            continue

        prod_mae = float(mae_arr[prod_mask][0])
        prod_se_raw = float(se_arr[prod_mask][0])
        prod_se = prod_se_raw if np.isfinite(prod_se_raw) else 0.0

        masked_mae = np.where(eligible, mae_arr, np.inf)
        best_idx = int(np.argmin(masked_mae))
        best_mae = float(mae_arr[best_idx])
        best_w = float(w_arr[best_idx])
        delta = best_mae - prod_mae

        if abs(delta) < prod_se:
            verdict = (
                f"production W={PRODUCTION_ROLLING_WINDOW} indistinguishable from best"
            )
        elif delta < 0:
            verdict = (
                f"best W={best_w:.0f} beats production by {abs(delta):.4f} d (> 1 SE)"
            )
        else:
            verdict = "production is already optimal"

        print(
            f"  {p:<5}: best MAE = {best_mae:.4f} d @ W = {best_w:.0f}; "
            f"production W = {PRODUCTION_ROLLING_WINDOW} -> {prod_mae:.4f} d; "
            f"{verdict}"
        )
    print("=" * 88)


def make_figure(df_anchor: pd.DataFrame, df_roll: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {"PCE": "#1f77b4", "FF": "#2ca02c", "Jsc": "#9467bd", "Voc": "#d62728"}

    # --- Anchor sweep ---
    ax = axes[0]
    for p in TARGET_PARAMS:
        sub = df_anchor[df_anchor["parameter"] == p].sort_values("anchor")
        if sub.empty:
            continue
        x = np.asarray(sub["anchor"].to_numpy(), dtype=float)
        y = np.asarray(sub["mae_mean"].to_numpy(), dtype=float)
        e = np.asarray(sub["mae_se"].to_numpy(), dtype=float)
        n = np.asarray(sub["n_cells"].to_numpy(), dtype=float)
        # Marker size scaled by N: bigger = more cells in the aggregate.
        sizes = np.clip(n * 3.0, 5.0, 15.0)
        for xi, yi, ei, ni, si in zip(x, y, e, n, sizes):
            style = "o" if ni >= MIN_CELLS_FOR_VERDICT else "x"
            ax.errorbar(
                xi, yi, yerr=ei, label=None, marker=style, capsize=3,
                color=colors.get(p, "#666666"), markersize=si,
            )
        # Add one entry to the legend per parameter.
        ax.plot([], [], marker="o", linestyle="-", color=colors.get(p, "#666666"),
                label=p)
    ax.axvline(PRODUCTION_ANCHOR, color="gray", linestyle=":", alpha=0.6,
               label=f"production anchor = {PRODUCTION_ANCHOR:.0f} d")
    ax.set_xlabel("ANCHOR_DAY (d)")
    ax.set_ylabel(f"MAE on first {EVALUATION_HORIZON} forecast days")
    ax.set_title(
        "Experiment A — Anchor sweep\n"
        f"(rolling window fixed at production; "
        f"x = N < {MIN_CELLS_FOR_VERDICT}, o = N >= {MIN_CELLS_FOR_VERDICT})"
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    # --- Rolling window sweep ---
    ax = axes[1]
    for p in TARGET_PARAMS:
        sub = df_roll[df_roll["parameter"] == p].sort_values("rolling_window")
        if sub.empty:
            continue
        x = np.asarray(sub["rolling_window"].to_numpy(), dtype=float)
        y = np.asarray(sub["mae_mean"].to_numpy(), dtype=float)
        e = np.asarray(sub["mae_se"].to_numpy(), dtype=float)
        n = np.asarray(sub["n_cells"].to_numpy(), dtype=float)
        sizes = np.clip(n * 3.0, 5.0, 15.0)
        for xi, yi, ei, ni, si in zip(x, y, e, n, sizes):
            style = "s" if ni >= MIN_CELLS_FOR_VERDICT else "x"
            ax.errorbar(
                xi, yi, yerr=ei, label=None, marker=style, capsize=3,
                color=colors.get(p, "#666666"), markersize=si,
            )
        ax.plot([], [], marker="s", linestyle="-", color=colors.get(p, "#666666"),
                label=p)
    ax.axvline(PRODUCTION_ROLLING_WINDOW, color="gray", linestyle=":", alpha=0.6,
               label=f"production W = {PRODUCTION_ROLLING_WINDOW}")
    ax.set_xlabel("ROLLING_WINDOW (days)")
    ax.set_ylabel(f"MAE on first {EVALUATION_HORIZON} forecast days")
    ax.set_title(
        "Experiment B — Rolling window sweep\n"
        f"(anchor fixed at production; "
        f"x = N < {MIN_CELLS_FOR_VERDICT}, s = N >= {MIN_CELLS_FOR_VERDICT})"
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ==============================================================================
# MAIN
# ==============================================================================
def main() -> None:
    DIAGNOSTICS_TRAJECTORY_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading inputs...")
    df_healthy = pd.read_parquet(FILE_HEALTHY_COHORT)
    artifact = joblib.load(FILE_SCREENING_ARTIFACTS)
    healthy_cohort = list(artifact["healthy_cohort"])
    logger.info(f"Production cohort: {healthy_cohort}")

    # --- Experiment A ---
    df_anchor = run_anchor_sweep(df_healthy, healthy_cohort, TARGET_PARAMS)
    print_sweep_table(
        df_anchor, "anchor", "ANCHOR SWEEP — per-parameter MAE", healthy_cohort,
    )
    print_anchor_verdict(df_anchor)

    # --- Experiment B ---
    df_roll = run_rolling_sweep(df_healthy, healthy_cohort, TARGET_PARAMS)
    print_sweep_table(
        df_roll, "rolling_window",
        "ROLLING WINDOW SWEEP — per-parameter MAE", healthy_cohort,
    )
    print_rolling_verdict(df_roll)

    # --- Persist ---
    out_anchor = DIAGNOSTICS_TRAJECTORY_DIR / "audit_trajectory_anchor_sweep.parquet"
    out_roll = DIAGNOSTICS_TRAJECTORY_DIR / "audit_trajectory_rolling_sweep.parquet"
    df_anchor.to_parquet(out_anchor, index=False)
    df_roll.to_parquet(out_roll, index=False)
    logger.info(f"Anchor sweep -> {out_anchor}")
    logger.info(f"Rolling sweep -> {out_roll}")

    # --- Figure ---
    out_fig = DIAGNOSTICS_TRAJECTORY_DIR / "audit_trajectory_window_sweep.png"
    make_figure(df_anchor, df_roll, out_fig)
    logger.info(f"Figure -> {out_fig}")

    print("\n" + "=" * 88)
    print(" TRAJECTORY WINDOW AUDIT COMPLETE")
    print("=" * 88)


if __name__ == "__main__":
    main()
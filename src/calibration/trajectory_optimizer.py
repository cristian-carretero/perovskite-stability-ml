"""
Module: src/trajectory_calibration_optimizer.py
Description: Empirical calibration of the per-parameter persistence-blend
             coefficient k_blend of 09_jv_mppt_trajectory_forecasting.py.

             Key insight: k_blend does NOT affect the autoregressive state
             (the blend is applied only to the reported value). The raw motor
             trajectory is therefore k-independent, so we can:

               1. Pre-compute one raw motor trajectory per (cell, param)
                  using blind LOOCV models (k=0).
               2. Sweep k in [0, 1] by recombining the cached arrays.
               3. Persist the per-parameter optimal k to JSON.

             A minimum relative improvement threshold (MIN_REL_IMPROVEMENT)
             filters out parameters where the blend does not earn its extra
             degree of freedom: if the best k does not beat k=0 by at least
             that fraction, the parameter is pinned to k=0.

             Horizon alignment: the sweep evaluates MAE only over the first
             EVALUATION_HORIZON days (the same window used to produce the
             MAE figures in the technical report). Cells whose evaluation
             window is shorter are excluded to keep the comparison fair.

             Outputs:
               - outputs/diagnostics/trajectory_k_sensitivity.parquet
               - outputs/diagnostics/trajectory_coeffs_calibrated.json
               - outputs/diagnostics/trajectory_coeffs_optimization.txt
"""

from __future__ import annotations

import importlib
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, TextIO

import joblib
import numpy as np
import pandas as pd

from src.config import (
    FILE_HEALTHY_COHORT,
    FILE_SCREENING_ARTIFACTS,
    FILE_TRAJECTORY_COEFFS_CALIBRATED,
    DIAGNOSTICS_TRAJECTORY_DIR,
    ANCHOR_DAY,
    EVALUATION_HORIZON,
)


# ==============================================================================
# Dynamic import of the 09 module (filename starts with a digit)
# ==============================================================================
traj = importlib.import_module("src.09_jv_mppt_trajectory_forecasting")


# ==============================================================================
# Logging + report paths
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("Trajectory-Calib-Optimizer")

REPORT_PATH = DIAGNOSTICS_TRAJECTORY_DIR / "trajectory_coeffs_optimization.txt"
SENSITIVITY_PATH = DIAGNOSTICS_TRAJECTORY_DIR / "trajectory_k_sensitivity.parquet"


# ==============================================================================
# Configuration
# ==============================================================================
K_GRID: tuple[float, ...] = (
    0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30,
    0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00,
)
# <2% relative improvement over k=0 -> do not activate the blend.
MIN_REL_IMPROVEMENT = 0.02


# ==============================================================================
# Stdout tee
# ==============================================================================
class _Tee:
    def __init__(self, *streams: TextIO) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for s in self._streams:
            s.write(data)
        return len(data)

    def flush(self) -> None:
        for s in self._streams:
            s.flush()


# ==============================================================================
# Raw trajectory cache
# ==============================================================================
@dataclass
class RawTrajectory:
    """Per (cell, param): the k-independent motor trajectory + persistence."""
    cell: str
    param: str
    days: np.ndarray
    raw_motor: np.ndarray
    persistence: float
    actual: np.ndarray


def precompute_raw_trajectories(
    df_daily: pd.DataFrame,
    healthy_cohort: List[str],
    targets: List[str],
) -> List[RawTrajectory]:
    """
    One LOOCV pass with k=0 -> cache raw motor + persistence + actual.

    Cells whose evaluation window is shorter than EVALUATION_HORIZON are
    excluded from the cache and therefore from the k_blend sweep. Rationale:
    the sweep compares MAE across cells, so all cells must share the same
    evaluation window. Truncated cells (e.g. M83AB302) are still handled by
    the 09 engine for dashboard display, but they do not participate in
    calibration.
    """
    units: List[RawTrajectory] = []

    for test_cell in healthy_cohort:
        df_train = df_daily[df_daily["cell_name"] != test_cell]
        df_test = df_daily[df_daily["cell_name"] == test_cell]
        if df_test.empty:
            continue

        hist = df_test[df_test["Exposure_Days"] <= ANCHOR_DAY]
        future = df_test[df_test["Exposure_Days"] > ANCHOR_DAY].head(EVALUATION_HORIZON)

        if hist.empty:
            logger.warning(f"[{test_cell}] No history before anchor. Skipping.")
            continue

        if len(future) < EVALUATION_HORIZON:
            logger.info(
                f"[{test_cell}] Truncated cell ({len(future)} of {EVALUATION_HORIZON} future days) "
                f"— excluded from k_blend sweep to keep MAE comparable across cells."
            )
            continue

        blind = traj.train_multivariate_engine(df_train, targets)

        init_norm = {p: float(hist.iloc[-1][f"{p}_Smooth"]) for p in targets}
        init_raw = {p: float(hist.iloc[-1][f"{p}_Initial"]) for p in targets}

        future_weather = future[[
            "Date_Day", "Exposure_Days",
            "Daily_Irradiance_Dose", "Daily_Max_Temp_C", "Daily_Median_Humidity",
        ]]

        df_sim = traj.simulate_trajectory(
            init_norm, init_raw, future_weather, blind, hist, targets,
            k_blend_per_param={p: 0.0 for p in targets},
        )

        for p in targets:
            units.append(RawTrajectory(
                cell=test_cell,
                param=p,
                days=future["Exposure_Days"].to_numpy(dtype=float),
                raw_motor=df_sim[f"Pred_{p}"].to_numpy(dtype=float),
                persistence=init_raw[p] * init_norm[p],
                actual=(future[f"{p}_Smooth"] * future[f"{p}_Initial"]).to_numpy(dtype=float),
            ))
        logger.info(f"[{test_cell}] cached {len(targets)} parameter trajectories.")

    return units


# ==============================================================================
# Vectorized k-sweep
# ==============================================================================
def sweep_k_blend(units: List[RawTrajectory]) -> pd.DataFrame:
    """
    Recombine cached arrays to compute MAE(k) per parameter. The math is a
    closed-form recombination:
        pred(k) = (1 - k) * raw_motor + k * persistence
    """
    rows = []
    for k in K_GRID:
        for u in units:
            pred = (1.0 - k) * u.raw_motor + k * u.persistence
            mae = float(np.mean(np.abs(pred - u.actual)))
            rows.append({"k": k, "Parameter": u.param, "cell": u.cell, "MAE": mae})

    df = pd.DataFrame(rows)
    # Mean MAE across cells == LOOCV MAE for that parameter (since each cell
    # contributes exactly EVALUATION_HORIZON points).
    return df.groupby(["k", "Parameter"])["MAE"].mean().unstack("Parameter")


# ==============================================================================
# Optimal-k selection (with the 2% adoption filter)
# ==============================================================================
def select_optimal_k(sweep_df: pd.DataFrame) -> Dict[str, float]:
    optimal: Dict[str, float] = {}
    for p in sweep_df.columns:
        series: pd.Series = sweep_df[p]
        mae_0 = float(series.loc[0.0])
        k_star = float(series.idxmin())
        mae_star = float(series.loc[k_star])
        rel = (mae_0 - mae_star) / mae_0 if mae_0 > 0 else 0.0
        if rel >= MIN_REL_IMPROVEMENT:
            optimal[p] = k_star
        else:
            optimal[p] = 0.0
    return optimal


# ==============================================================================
# Reporting
# ==============================================================================
def print_sweep_table(sweep_df: pd.DataFrame, optimal: Dict[str, float]) -> None:
    print("\n" + "=" * 90)
    print(" PERSISTENCE-BLEND SWEEP (MAE vs. k, LOOCV)")
    print("=" * 90)
    header = "  " + "k".ljust(7) + "".join(p.rjust(12) for p in sweep_df.columns)
    print(header)
    for k in sweep_df.index:
        row: pd.Series = sweep_df.loc[k]
        line = f"  {k:<7.2f}" + "".join(
            f"{float(row[p]):>12.4f}" for p in sweep_df.columns
        )
        print(line)

    print("\n  Optimal k per parameter (with %d%% relative-improvement filter):"
          % int(MIN_REL_IMPROVEMENT * 100))
    for p in sweep_df.columns:
        series: pd.Series = sweep_df[p]
        mae_0 = float(series.loc[0.0])
        k_star = optimal[p]
        mae_star = float(series.loc[k_star])
        rel = 100.0 * (mae_0 - mae_star) / mae_0 if mae_0 > 0 else 0.0
        flag = " [filtered]" if k_star == 0.0 and series.idxmin() != 0.0 else ""
        print(f"    {p:<5s} k* = {k_star:<5.2f}  "
              f"MAE(k*) = {mae_star:.4f}  (MAE(0) = {mae_0:.4f}, Δ = {rel:+.1f}%){flag}")
    print("=" * 90)


# ==============================================================================
# Persist calibrated coefficients
# ==============================================================================
def persist_calibrated_coefficients(
    optimal: Dict[str, float],
    sweep_df: pd.DataFrame,
) -> None:
    payload = {
        "k_blend_per_param": {p: float(k) for p, k in optimal.items()},
        "metadata": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "generated_by": "src/trajectory_calibration_optimizer.py",
            "min_rel_improvement": MIN_REL_IMPROVEMENT,
            "evaluation_horizon": int(EVALUATION_HORIZON),
            "k_grid": list(K_GRID),
            "mae_at_k0": {
                p: float(sweep_df[p].loc[0.0]) for p in sweep_df.columns
            },
            "mae_at_kstar": {
                p: float(sweep_df[p].loc[optimal[p]]) for p in sweep_df.columns
            },
        },
    }
    FILE_TRAJECTORY_COEFFS_CALIBRATED.parent.mkdir(parents=True, exist_ok=True)
    with FILE_TRAJECTORY_COEFFS_CALIBRATED.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\n  Calibrated k_blend persisted -> {FILE_TRAJECTORY_COEFFS_CALIBRATED}")
    print("  The 09 module will pick these up on its next run.")


# ==============================================================================
# Entry point
# ==============================================================================
def _run() -> None:
    logger.info("Loading Digital Twin healthy cohort...")
    df_healthy = pd.read_parquet(FILE_HEALTHY_COHORT)
    artifacts = joblib.load(FILE_SCREENING_ARTIFACTS)
    healthy_cohort = artifacts["healthy_cohort"]

    df_daily, available_targets = traj.build_trajectory_matrix(df_healthy)
    logger.info(f"Parameters to calibrate: {available_targets}")
    logger.info(f"Evaluation horizon for the sweep: {EVALUATION_HORIZON} days")

    print("\n" + "=" * 90)
    print(" PRE-COMPUTING RAW MOTOR TRAJECTORIES (k=0, one LOOCV pass)")
    print("=" * 90)
    units = precompute_raw_trajectories(df_daily, healthy_cohort, available_targets)
    print(f"  Cached {len(units)} (cell, param) trajectories.")

    sweep_df = sweep_k_blend(units)
    SENSITIVITY_PATH.parent.mkdir(parents=True, exist_ok=True)
    sweep_df.to_parquet(SENSITIVITY_PATH)
    print(f"\n  Sweep table saved -> {SENSITIVITY_PATH}")

    optimal = select_optimal_k(sweep_df)
    print_sweep_table(sweep_df, optimal)
    persist_calibrated_coefficients(optimal, sweep_df)


def main() -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    original_stdout = sys.stdout
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        sys.stdout = _Tee(original_stdout, f)
        try:
            _run()
        finally:
            sys.stdout = original_stdout
    logger.info(f"Full optimization report saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
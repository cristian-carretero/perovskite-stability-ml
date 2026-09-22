"""
Module: src/audit_trajectory_window_all_cells.py
Description: Exploratory variant of src/audit_trajectory_window.py that uses
             ALL cells in the dataset instead of the 4-cell healthy cohort.

             Purpose: document whether the healthy-only conclusions
             (minimum anchor = 14 days, rolling window 7 suboptimal for PCE
             and Jsc) change with a larger, noisier cohort.

             WARNING — NOT a production configuration.
             Module 09 is designed to train and evaluate only on the healthy
             cohort. Including short-lived cells (A162ALXP01, A170AB302, M0,
             all with lifespan < 10 days) and a cell excluded by the LOOCV
             gate (P12, 95.8% alert frequency) contaminates both training
             and the aggregate MAE. Results are a sensitivity analysis,
             not a recommendation.

             Reuses every helper from the healthy-only audit:
               - _extract_mae_per_cell (via run_*_sweep)
               - _aggregate_per_param  (via run_*_sweep)
               - run_anchor_sweep / run_rolling_sweep
               - print_sweep_table / print_anchor_verdict / print_rolling_verdict
               - make_figure

             Outputs:
               outputs/diagnostics/trajectory/
                 audit_trajectory_anchor_sweep_all_cells.parquet
                 audit_trajectory_rolling_sweep_all_cells.parquet
                 audit_trajectory_window_sweep_all_cells.png
"""

from __future__ import annotations

import logging

import joblib
import pandas as pd

from src.audit_trajectory_window import (
    make_figure,
    print_anchor_verdict,
    print_rolling_verdict,
    print_sweep_table,
    run_anchor_sweep,
    run_rolling_sweep,
)
from src.config import (
    DAYLIGHT_IRRADIANCE_MIN_W_M2,
    DIAGNOSTICS_TRAJECTORY_DIR,
    FILE_MERGED_FEATURES,
    FILE_SCREENING_ARTIFACTS,
    FILE_T80_TRUTH,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("AuditTrajectoryWindowAllCells")


# ==============================================================================
# DATA LOADING — ALL CELLS
# ==============================================================================
def load_all_cells_data() -> pd.DataFrame:
    """
    Load the merged feature matrix (all cells), apply the same preprocessing
    module 07 applies before its own aggregation, and return a DataFrame
    suitable for module 09's build_trajectory_matrix.

    Critical detail: Day_Zero must be computed on the UNFILTERED telemetry
    so that Exposure_Days matches the reference definition used by module 07
    (and therefore by everything downstream). Computing Day_Zero after the
    daylight filter would shift Exposure_Days by a few hours for cells whose
    first observation was at night.
    """
    df = pd.read_parquet(FILE_MERGED_FEATURES)

    # Defensive: some parquet writers store Timestamp as the index.
    if df.index.name == "Timestamp":
        df = df.reset_index()

    # PCE_initial for format parity with FILE_HEALTHY_COHORT. Module 09's
    # build_trajectory_matrix does not consume it (it derives per-parameter
    # initial values from the first three days), but keeping the column
    # avoids surprises if downstream code assumes the production schema.
    if "PCE_initial" not in df.columns:
        t80 = pd.read_parquet(FILE_T80_TRUTH)
        df = df.merge(
            t80[["PCE_initial"]],
            left_on="cell_name", right_index=True, how="left",
        )

    # Datetime + Day_Zero BEFORE the daylight filter (module 07 reference).
    if "Datetime" not in df.columns:
        df["Datetime"] = pd.to_datetime(df["Timestamp"], utc=True)
    df["Day_Zero"] = df.groupby("cell_name")["Datetime"].transform("min")

    # Daylight filter.
    df = df[df["POA_Irradiance_W_m2"] > DAYLIGHT_IRRADIANCE_MIN_W_M2].copy()

    # Exposure_Days uses the pre-filter Day_Zero.
    df["Exposure_Days"] = (
        (df["Datetime"] - df["Day_Zero"]).dt.total_seconds() / 86400.0
    )
    return df


# ==============================================================================
# MAIN
# ==============================================================================
def main() -> None:
    DIAGNOSTICS_TRAJECTORY_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading merged feature matrix (all cells)...")
    df_all = load_all_cells_data()

    all_cells = sorted(str(c) for c in df_all["cell_name"].unique())
    healthy_artifact = joblib.load(FILE_SCREENING_ARTIFACTS)
    healthy_cohort = list(healthy_artifact["healthy_cohort"])

    print("\n" + "=" * 88)
    print(" ALL-CELLS TRAJECTORY WINDOW AUDIT (exploratory)")
    print("=" * 88)
    print(f"  All cells in dataset : {all_cells}")
    print(f"  Healthy cohort (07)  : {healthy_cohort}")
    print(f"  Extra cells included : {sorted(set(all_cells) - set(healthy_cohort))}")
    print("=" * 88)
    print("\n  WARNING: this bypasses the screening gate. Results are a")
    print("  sensitivity analysis, not a production configuration.")

    targets = ["PCE", "FF", "Jsc", "Voc"]

    # --- Experiment A: anchor sweep over all cells ---
    df_anchor = run_anchor_sweep(df_all, all_cells, targets=targets)
    print_sweep_table(
        df_anchor, "anchor",
        "ANCHOR SWEEP (ALL CELLS) — per-parameter MAE",
        all_cells,
    )
    print_anchor_verdict(df_anchor)

    # --- Experiment B: rolling window sweep over all cells ---
    df_roll = run_rolling_sweep(df_all, all_cells, targets=targets)
    print_sweep_table(
        df_roll, "rolling_window",
        "ROLLING WINDOW SWEEP (ALL CELLS) — per-parameter MAE",
        all_cells,
    )
    print_rolling_verdict(df_roll)

    # --- Persist ---
    out_anchor = (
        DIAGNOSTICS_TRAJECTORY_DIR / "audit_trajectory_anchor_sweep_all_cells.parquet"
    )
    out_roll = (
        DIAGNOSTICS_TRAJECTORY_DIR / "audit_trajectory_rolling_sweep_all_cells.parquet"
    )
    df_anchor.to_parquet(out_anchor, index=False)
    df_roll.to_parquet(out_roll, index=False)
    logger.info(f"Anchor sweep -> {out_anchor}")
    logger.info(f"Rolling sweep -> {out_roll}")

    # --- Figure ---
    out_fig = (
        DIAGNOSTICS_TRAJECTORY_DIR / "audit_trajectory_window_sweep_all_cells.png"
    )
    make_figure(df_anchor, df_roll, out_fig)
    logger.info(f"Figure -> {out_fig}")

    print("\n" + "=" * 88)
    print(" ALL-CELLS TRAJECTORY WINDOW AUDIT COMPLETE")
    print("=" * 88)


if __name__ == "__main__":
    main()
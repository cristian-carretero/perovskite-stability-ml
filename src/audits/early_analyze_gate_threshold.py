"""
Module: src/early_analyze_gate_threshold.py
Description: Diagnostic helper for the early-screening gate (module 07).

Loads the persisted screening artifacts and reports:

  1. Per-cell alert frequencies within the screening cohort.
  2. The production cohort that would result for a sweep of candidate
     ALERT_FREQUENCY_THRESHOLD_PCT values.

Purpose: expose the plateau of the gate decision (if any) so that the
choice of ALERT_FREQUENCY_THRESHOLD_PCT is grounded in an empirical
sensitivity analysis rather than in a hardcoded default. Read-only with
respect to the pipeline: does not modify any artifact produced by module 07.
Safe to run at any time after 07 has completed at least one successful run.

Outputs:
  - outputs/diagnostics/07_gate_threshold_sensitivity.parquet
"""

import logging

import joblib
import numpy as np
import pandas as pd

from src.config import (
    ALERT_FREQUENCY_THRESHOLD_PCT,
    DIAGNOSTICS_SCREENING_DIR,
    FILE_SCREENING_ARTIFACTS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("GateSensitivity")

# Candidate thresholds covering the plausible range. The current value from
# config is always included and flagged in the output for traceability.
CANDIDATE_THRESHOLDS = [5.0, 10.0, 25.0, 40.0, 50.0]

OUTPUT_PATH = DIAGNOSTICS_SCREENING_DIR / "07_gate_threshold_sensitivity.parquet"


def load_artifacts() -> dict:
    if not FILE_SCREENING_ARTIFACTS.exists():
        raise FileNotFoundError(
            f"Missing {FILE_SCREENING_ARTIFACTS}. Run module 07 first."
        )
    return joblib.load(FILE_SCREENING_ARTIFACTS)


def build_frequency_table(art: dict) -> pd.DataFrame:
    summary = art["summary_table"]
    screening = art["screening_cohort"]
    rows = []
    for cell in screening:
        if cell not in summary.index:
            logger.warning(f"Cell {cell} missing from summary_table; skipping.")
            continue
        rows.append({
            "cell_name": cell,
            "alert_freq_pct": float(summary.loc[cell, "alert_freq_pct"]),
            "alert_pce_pct": float(summary.loc[cell, "alert_pce_pct"]),
            "alert_pff_pct": float(summary.loc[cell, "alert_pff_pct"]),
            "combined_survival_days": float(summary.loc[cell, "combined_survival_days"]),
            "diagnostic_status": str(summary.loc[cell, "Diagnostic_Status"]),
        })
    return (
        pd.DataFrame(rows)
        .sort_values("alert_freq_pct", ascending=False)
        .reset_index(drop=True)
    )


def build_sensitivity_table(
    freq_df: pd.DataFrame, thresholds: list[float]
) -> pd.DataFrame:
    rows = []
    for thr in thresholds:
        gated = freq_df.loc[freq_df["alert_freq_pct"] > thr, "cell_name"].tolist()
        production = freq_df.loc[freq_df["alert_freq_pct"] <= thr, "cell_name"].tolist()
        rows.append({
            "threshold_pct": float(thr),
            "is_current": bool(np.isclose(thr, ALERT_FREQUENCY_THRESHOLD_PCT)),
            "n_gated": len(gated),
            "n_production": len(production),
            "gated_cells": ",".join(sorted(gated)),
            "production_cells": ",".join(sorted(production)),
        })
    return pd.DataFrame(rows)


def _current_production_set(
    freq_df: pd.DataFrame, sens_df: pd.DataFrame
) -> str:
    """Resolve the production cohort at the config threshold, robustly."""
    if sens_df["is_current"].any():
        return sens_df.loc[sens_df["is_current"], "production_cells"].iloc[0]
    logger.warning(
        f"Current threshold {ALERT_FREQUENCY_THRESHOLD_PCT}% not in candidate list; "
        f"computing production set directly from freq_df."
    )
    cells = freq_df.loc[
        freq_df["alert_freq_pct"] <= ALERT_FREQUENCY_THRESHOLD_PCT, "cell_name"
    ].tolist()
    return ",".join(sorted(cells))


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    art = load_artifacts()
    freq_df = build_frequency_table(art)
    sens_df = build_sensitivity_table(freq_df, CANDIDATE_THRESHOLDS)

    print("\n" + "=" * 72)
    print(" EARLY SCREENING GATE — SENSITIVITY ANALYSIS")
    print("=" * 72)
    print(f"  Current ALERT_FREQUENCY_THRESHOLD_PCT = {ALERT_FREQUENCY_THRESHOLD_PCT:.1f}%")
    print(f"  Screening cohort size: {len(freq_df)}")

    print("\n  Per-cell alert frequencies:")
    print(
        freq_df.to_string(
            index=False,
            formatters={
                "alert_freq_pct": "{:6.2f}".format,
                "alert_pce_pct": "{:6.2f}".format,
                "alert_pff_pct": "{:6.2f}".format,
                "combined_survival_days": "{:8.2f}".format,
            },
        )
    )

    print("\n  Gate sensitivity (production cohort per threshold):")
    print(
        sens_df[
            ["threshold_pct", "is_current", "n_gated", "n_production",
             "gated_cells", "production_cells"]
        ].to_string(index=False)
    )

    current_set = _current_production_set(freq_df, sens_df)
    invariant = sens_df.loc[sens_df["production_cells"] == current_set, "threshold_pct"]
    print("\n  Invariance summary:")
    print(f"    Current production set : {current_set}")
    if not invariant.empty:
        print(
            f"    Invariant for threshold ∈ "
            f"[{invariant.min():.1f}%, {invariant.max():.1f}%]"
        )
    else:
        print("    Current production set is unique to the current threshold.")

    out = sens_df.copy()
    out.insert(0, "generated_from_cohort_size", len(freq_df))
    out.insert(1, "current_threshold_pct", ALERT_FREQUENCY_THRESHOLD_PCT)
    out.to_parquet(OUTPUT_PATH, index=False)
    logger.info(f"Sensitivity table saved -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
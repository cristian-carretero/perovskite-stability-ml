"""
Module: src/xai_physical_forensic.py
Description: Independent Physical Forensic Diagnostic.
Diagnoses the environmental triggers of physical T80 collapse by contrasting
the prodromal (critical) window against the cell's own healthy history.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

from src.config import (
    EARLY_FAILURE_WINDOW_DAYS,
    FEATURE_IMPORTANCE_MIN,
    PRODROMAL_WINDOW_DAYS,
    SURROGATE_TREE_PARAMS,
    XAI_PHYSICAL_FEATURES,
    FILE_HEALTHY_COHORT,
    FILE_T80_TRUTH,
    FILE_SCREENING_ARTIFACTS,
    DIAGNOSTICS_XAI_DIR,       
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("XAI_Forensic")

# Minimum early-life records required to fit a meaningful surrogate.
MIN_EARLY_SAMPLES = 20

# Sentinel used when a survival metric is unavailable.
SURVIVAL_SENTINEL = 999.0


def extract_physical_forensic_surrogate(
    cell_data: pd.DataFrame,
    failure_day: float,
) -> dict[str, Any]:
    """
    Fit a shallow surrogate tree to distinguish the prodromal window (the
    `PRODROMAL_WINDOW_DAYS` immediately before the T80 collapse) from the
    cell's own earlier healthy history, using only physical covariates.

    The resulting rule tree reveals whether the collapse was preceded by
    anomalous environmental conditions or whether it looks more like an
    intrinsic defect.
    """
    early_data = cell_data[cell_data["Exposure_Days"] <= failure_day].copy()
    if len(early_data) < MIN_EARLY_SAMPLES:
        logger.debug(
            f"Not enough early-life samples to fit surrogate "
            f"({len(early_data)} < {MIN_EARLY_SAMPLES})."
        )
        return {}

    critical_start = max(0.0, failure_day - PRODROMAL_WINDOW_DAYS)
    early_data["Is_Critical"] = (early_data["Exposure_Days"] >= critical_start).astype(int)

    if early_data["Is_Critical"].nunique() < 2:
        logger.debug("Prodromal window coincides with all available history; skipping.")
        return {}

    X = early_data[XAI_PHYSICAL_FEATURES]
    y = early_data["Is_Critical"]

    tree = DecisionTreeClassifier(
        max_depth=int(SURROGATE_TREE_PARAMS.get("max_depth", 3)),
        min_samples_leaf=int(SURROGATE_TREE_PARAMS.get("min_samples_leaf", 5)),
        class_weight=str(SURROGATE_TREE_PARAMS.get("class_weight", "balanced")),
        random_state=int(SURROGATE_TREE_PARAMS.get("random_state", 42)),
    )
    tree.fit(X, y)

    # A single-node tree means no environmental split separates the prodromal
    # window from the healthy history: the collapse looks intrinsic.
    if tree.tree_.node_count <= 1:
        return {
            "T80_Day": float(failure_day),
            "Feature_Importances": {},
            "Rules": [
                "Intrinsic Defect: no anomalous environmental trigger separates "
                "the prodromal window from the cell's healthy history."
            ],
        }

    rules = export_text(tree, feature_names=XAI_PHYSICAL_FEATURES, decimals=1)
    importances = {
        feat: float(imp)
        for feat, imp in zip(XAI_PHYSICAL_FEATURES, tree.feature_importances_)
        if imp > FEATURE_IMPORTANCE_MIN
    }
    sorted_imps = dict(sorted(importances.items(), key=lambda item: item[1], reverse=True))

    return {
        "T80_Day": float(failure_day),
        "Feature_Importances": sorted_imps,
        "Rules": rules.strip().split("\n"),
    }


def _resolve_summary_table(artifacts: dict[str, Any]) -> pd.DataFrame | None:
    """
    Return the summary table with the survival metrics required to identify
    early-collapse cells. If the screening artifact does not carry the T80
    survival columns, merge them in from the T80 ground-truth parquet.
    """
    summary_table = artifacts.get("summary_table")
    if summary_table is None:
        logger.warning("'summary_table' not found in screening artifacts.")
        return None

    required_cols = {"survival_days_pce", "survival_days_pff"}
    if required_cols.issubset(summary_table.columns):
        return summary_table

    if not FILE_T80_TRUTH.exists():
        logger.warning(
            "Survival columns missing from summary_table and T80 ground-truth "
            "parquet not available. Forensic diagnostic cannot proceed."
        )
        return None

    logger.info("Merging survival metrics from T80 ground truth...")
    t80_metrics = pd.read_parquet(FILE_T80_TRUTH)

    # Only take the columns we actually need, tolerate either index or column
    # alignment with the summary table.
    t80_subset = t80_metrics[[c for c in required_cols if c in t80_metrics.columns]]
    if not required_cols.issubset(t80_subset.columns):
        logger.warning("T80 ground-truth parquet does not contain the required survival columns.")
        return None

    if "cell_name" in summary_table.columns and "cell_name" in t80_metrics.columns:
        return summary_table.merge(
            t80_subset.assign(cell_name=t80_metrics["cell_name"]),
            on="cell_name",
            how="left",
        )

    # Fallback: assume both share the cell_name index.
    return summary_table.merge(t80_subset, left_index=True, right_index=True, how="left")


def main() -> None:
    DIAGNOSTICS_XAI_DIR.mkdir(parents=True, exist_ok=True)

    if not FILE_HEALTHY_COHORT.exists() or not FILE_SCREENING_ARTIFACTS.exists():
        logger.error(
            "Required screening artifacts missing. "
            "Ensure src.07_jv_mppt_early_screening executed successfully."
        )
        raise SystemExit(1)

    df_scored = pd.read_parquet(FILE_HEALTHY_COHORT)
    artifacts = joblib.load(FILE_SCREENING_ARTIFACTS)

    summary_table = _resolve_summary_table(artifacts)
    if summary_table is None:
        logger.error("Forensic diagnostic aborted: no usable summary table.")
        raise SystemExit(1)

    # Identify cells with premature physical T80 collapse.
    t80_failed_cells = summary_table[
        (summary_table["survival_days_pce"] <= EARLY_FAILURE_WINDOW_DAYS)
        | (summary_table["survival_days_pff"] <= EARLY_FAILURE_WINDOW_DAYS)
    ].copy()

    logger.info(
        f"Extracting Physical Forensic Rules for {len(t80_failed_cells)} T80 collapsed devices..."
    )

    report_dict: dict[str, Any] = {}
    for cell, row in t80_failed_cells.iterrows():
        cell_data = df_scored[df_scored["cell_name"] == str(cell)]
        if cell_data.empty:
            logger.warning(f"No telemetry found for cell {cell}. Skipping.")
            continue

        t80_day = min(
            float(row.get("survival_days_pce", SURVIVAL_SENTINEL)),
            float(row.get("survival_days_pff", SURVIVAL_SENTINEL)),
        )

        report = extract_physical_forensic_surrogate(cell_data, t80_day)
        if report:
            report_dict[str(cell)] = report

    out_file = DIAGNOSTICS_XAI_DIR / "forensic_surrogate_rules.json"
    with open(out_file, "w") as f:
        json.dump(report_dict, f, indent=4)

    logger.info(f"Physical Forensic pipeline complete. Saved to {out_file.name}")


if __name__ == "__main__":
    main()
"""
Module: src/xai_digital_twin.py
Description: Explainability (XAI) for the Digital Twin ML models.
1. Global SHAP: validates the thermodynamic logic learned by XGBoost on healthy cells.
2. Local Surrogate: explains the environmental triggers behind early ML anomalies.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
from sklearn.tree import DecisionTreeClassifier, export_text

from src.config import (
    EARLY_FAILURE_WINDOW_DAYS,
    BURN_IN_DAYS,
    FEATURE_IMPORTANCE_MIN,
    FEATURES,
    RANDOM_STATE,
    SHAP_SAMPLE_SIZE,
    SURROGATE_TREE_PARAMS,
    XAI_PHYSICAL_FEATURES,
    FILE_HEALTHY_COHORT,
    FILE_SCREENING_ARTIFACTS,
    DIAGNOSTICS_XAI_DIR,       
    FIGURES_XAI_DIR,       
)

SHAP_OUT_DIR = FIGURES_XAI_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("XAI_DigitalTwin")

plt.rcParams.update({"font.size": 12, "figure.facecolor": "white"})


# Human-readable labels for the SHAP beeswarm plot. Explicit mapping avoids the
# silent corruption introduced by chained str.replace calls (e.g. 'C' -> '(°C)'
# would also mangle 'Cos' into '(°C)os').
FEATURE_DISPLAY_NAMES = {
    "POA_Irradiance_W_m2": "POA Irradiance (W/m²)",
    "ModuleTemp_C": "Module Temperature (°C)",
    "AbsoluteHumidity_g_m3": "Absolute Humidity (g/m³)",
    "Delta_Temp_C_per_h": "Δ Temperature (°C/h)",
    "Delta_Hum_g_m3_per_h": "Δ Humidity (g/m³·h)",
    "Hour_Sin": "Hour (sin)",
    "Hour_Cos": "Hour (cos)",
    "Day_Sin": "Day (sin)",
    "Day_Cos": "Day (cos)",
}


def generate_global_shap(
    model_pce,
    model_pff,
    df_data: pd.DataFrame,
    healthy_cells: list[str],
    output_dir: Path,
) -> None:
    """Generate the global SHAP beeswarm plots for both Digital Twin heads."""
    logger.info("Generating Global SHAP footprint for the Digital Twin...")

    healthy_data = df_data[
        (df_data["cell_name"].isin(healthy_cells))
        & (df_data["Exposure_Days"] > EARLY_FAILURE_WINDOW_DAYS)
    ]

    if healthy_data.empty:
        logger.warning("No healthy-window data available for global SHAP. Skipping.")
        return

    # Subsample to keep the beeswarm tractable; bound by the healthy subset size,
    # not by the total dataframe size.
    n_samples = min(SHAP_SAMPLE_SIZE, len(healthy_data))
    healthy_data = healthy_data.sample(n=n_samples, random_state=RANDOM_STATE)

    # SHAP audits the Digital Twin: it must use ALL features (including sin/cos).
    X_eval = healthy_data[FEATURES]
    display_names = [FEATURE_DISPLAY_NAMES.get(f, f) for f in FEATURES]

    for model, name in [(model_pce, "PCE"), (model_pff, "pFF")]:
        if model is None:
            logger.warning(f"Model for {name} is None. Skipping its SHAP plot.")
            continue

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_eval)

        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values,
            X_eval,
            feature_names=display_names,
            show=False,
            plot_type="dot",
        )
        plt.title(
            f"Global SHAP Impact on Healthy Baseline ({name})",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )
        plt.tight_layout()
        plt.savefig(
            output_dir / f"shap_global_beeswarm_{name.lower()}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


def _get_action_window(cell_data: pd.DataFrame) -> pd.DataFrame:
    """
    Return the early-life action window for a given cell.

    Uses the explicit `In_Action_Window` flag when present; otherwise,
    reconstructs it from `Exposure_Days <= BURN_IN_DAYS`. This keeps the
    diagnostic compatible with both the current and previous parquet schemas.
    """
    if "In_Action_Window" in cell_data.columns:
        return cell_data[cell_data["In_Action_Window"] == True].copy()

    if "Exposure_Days" in cell_data.columns:
        return cell_data[cell_data["Exposure_Days"] <= BURN_IN_DAYS].copy()

    logger.warning(
        "Neither 'In_Action_Window' nor 'Exposure_Days' present. "
        "Cannot isolate the action window."
    )
    return pd.DataFrame()


def extract_twin_surrogate(cell_data: pd.DataFrame, alert_day: float) -> dict[str, Any]:
    """
    Fit a shallow surrogate tree that explains the ML anomaly flag as a function
    of physical covariates only (temporal features are deliberately excluded).
    """
    action_window = _get_action_window(cell_data)
    if action_window.empty:
        return {}

    if "Digital_Twin_Alert" not in action_window.columns:
        logger.warning("Missing 'Digital_Twin_Alert' column. Cannot build surrogate.")
        return {}

    # Extreme case: zero variance in the alert flag (e.g. cell M0 always alerts).
    if action_window["Digital_Twin_Alert"].nunique() < 2:
        if bool(action_window["Digital_Twin_Alert"].iloc[0]):
            return {
                "ML_Alert_Day": float(alert_day),
                "Feature_Importances": {
                    "Catastrophic continuous failure (all features)": 1.0
                },
                "Rules": [
                    "Catastrophic Early Failure: the cell triggered ML anomalies "
                    "continuously across all environmental conditions."
                ],
            }
        return {}

    X = action_window[XAI_PHYSICAL_FEATURES]
    y = action_window["Digital_Twin_Alert"].astype(int)

    tree = DecisionTreeClassifier(
        max_depth=int(SURROGATE_TREE_PARAMS.get("max_depth", 3)),
        min_samples_leaf=int(SURROGATE_TREE_PARAMS.get("min_samples_leaf", 5)),
        class_weight=str(SURROGATE_TREE_PARAMS.get("class_weight", "balanced")),
        random_state=int(SURROGATE_TREE_PARAMS.get("random_state", 42)),
    )
    tree.fit(X, y)

    if tree.tree_.node_count <= 1:
        return {}

    rules = export_text(tree, feature_names=XAI_PHYSICAL_FEATURES, decimals=1)
    importances = {
        feat: float(imp)
        for feat, imp in zip(XAI_PHYSICAL_FEATURES, tree.feature_importances_)
        if imp > FEATURE_IMPORTANCE_MIN
    }
    sorted_imps = dict(sorted(importances.items(), key=lambda item: item[1], reverse=True))

    return {
        "ML_Alert_Day": float(alert_day),
        "Feature_Importances": sorted_imps,
        "Rules": rules.strip().split("\n"),
    }


def main() -> None:
    DIAGNOSTICS_XAI_DIR.mkdir(parents=True, exist_ok=True)
    SHAP_OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not FILE_HEALTHY_COHORT.exists() or not FILE_SCREENING_ARTIFACTS.exists():
        logger.error(
            "Required screening artifacts missing. "
            "Ensure src.07_jv_mppt_early_screening executed successfully."
        )
        raise SystemExit(1)

    df_scored = pd.read_parquet(FILE_HEALTHY_COHORT)
    artifacts = joblib.load(FILE_SCREENING_ARTIFACTS)

    summary_table = artifacts.get("summary_table")
    model_pce = artifacts.get("model_pce")
    model_pff = artifacts.get("model_pff")
    healthy_cohort = artifacts.get("healthy_cohort", [])

    # 1. Global SHAP on healthy cells.
    if healthy_cohort and model_pce is not None and model_pff is not None:
        generate_global_shap(model_pce, model_pff, df_scored, healthy_cohort, SHAP_OUT_DIR)
    else:
        logger.warning(
            "Global SHAP skipped: missing healthy_cohort or one of the twin models."
        )

    # 2. Local surrogates for cells flagged as ML-anomalous.
    if summary_table is None or "threshold_pct_day" not in summary_table.columns:
        logger.warning(
            "'summary_table' or 'threshold_pct_day' missing. Skipping local surrogates."
        )
        return

    ml_failed_cells = summary_table[summary_table["threshold_pct_day"].notna()].copy()
    logger.info(f"Extracting ML Surrogate Rules for {len(ml_failed_cells)} anomaly devices...")

    report_dict: dict[str, Any] = {}
    for cell, row in ml_failed_cells.iterrows():
        cell_data = df_scored[df_scored["cell_name"] == str(cell)]
        alert_day = float(row["threshold_pct_day"])

        report = extract_twin_surrogate(cell_data, alert_day)
        if report:
            report_dict[str(cell)] = report

    out_file = DIAGNOSTICS_XAI_DIR / "twin_surrogate_rules.json"
    with open(out_file, "w") as f:
        json.dump(report_dict, f, indent=4)

    logger.info(f"Digital Twin XAI complete. Saved to {out_file.name}")


if __name__ == "__main__":
    main()
"""
Module: src/trajectory_xgb_audit.py
Description: Empirical audit of the multivariate trajectory forecasting engines
             (PCE, FF, Jsc, Voc). Companion diagnostic to
             09_jv_mppt_trajectory_forecasting.py.

             Phase 1 — Diagnostic phase. Answers four questions per parameter:

               Q1. Is the target predictable in principle?
                   (signal strength: correlation with its own lag, variance,
                   proportion of zeros, skewness.)

               Q2. Which model family wins under strict cross-validation?
                   (trivial baselines, Ridge, RandomForest, XGBoost variants,
                   under both KFold and GroupKFold-by-cell.)

               Q3. Where does the error concentrate?
                   (per-cell breakdown and per-exposure-day breakdown.)

               Q4. Which features actually drive the prediction?
                   (Pearson, XGBoost gain, permutation importance.)

             The full audit report is automatically written to
             `outputs/diagnostics/trajectory_audit_summary.txt` while still
             being streamed to stdout.
"""

from __future__ import annotations

import importlib
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, TextIO, cast

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import skew
from sklearn.base import BaseEstimator, clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    make_scorer,
    mean_absolute_error,
    median_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    cross_val_predict,
    cross_validate,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import Bunch

from src.config import (
    RANDOM_STATE,
    XGB_PARAMS_RUL_PCE,
    FILE_HEALTHY_COHORT,
    FILE_SCREENING_ARTIFACTS,
    DIAGNOSTICS_TRAJECTORY_DIR,
)


# ==============================================================================
# Logging + dynamic import of the trajectory module
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("Trajectory-Audit")

_traj_mod = importlib.import_module("src.09_jv_mppt_trajectory_forecasting")
build_trajectory_matrix = _traj_mod.build_trajectory_matrix
BASE_FEATURES = _traj_mod.BASE_FEATURES


# ==============================================================================
# Audit configuration
# ==============================================================================
N_SPLITS = 5
N_PERMUTATION_REPEATS = 20
OVERFIT_GAP_RATIO = 0.30

# Signal-strength thresholds (percentage improvement over the trivial
# Dummy(mean) baseline under KFold).
SIGNAL_THRESHOLDS = {"none": 3.0, "weak": 10.0}

# Minimum absolute Pearson correlation between a target and its own lag
# feature required to consider the target "mean-reverting / autodependent".
AUTOCORR_MIN = 0.10

AUDIT_REPORT_PATH: Path = DIAGNOSTICS_TRAJECTORY_DIR / "trajectory_audit_summary.txt"

SCORING = {
    "MAE": make_scorer(mean_absolute_error, greater_is_better=False),
    "RMSE": make_scorer(root_mean_squared_error, greater_is_better=False),
    "MedAE": make_scorer(median_absolute_error, greater_is_better=False),
    "R2": make_scorer(r2_score),
}
METRIC_SIGNS = {"MAE": -1, "RMSE": -1, "MedAE": -1, "R2": 1}

TARGET_PARAMS = ["PCE", "FF", "Jsc", "Voc"]


# ==============================================================================
# Stdout tee
# ==============================================================================
class _Tee:
    """File-like object duplicating every write/flush to multiple streams."""

    def __init__(self, *streams: TextIO) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


# ==============================================================================
# Target specification
# ==============================================================================
@dataclass(frozen=True)
class TargetSpec:
    key: str
    target_column: str
    lag_feature: str
    features: list[str]
    xgb_params: Mapping


def _build_specs() -> list[TargetSpec]:
    """One spec per physical parameter, each with its own lag feature."""
    specs = []
    for param in TARGET_PARAMS:
        specs.append(TargetSpec(
            key=param.lower(),
            target_column=f"{param}_Delta",
            lag_feature=f"{param}_Lag1",
            features=BASE_FEATURES + [f"{param}_Lag1"],
            xgb_params=XGB_PARAMS_RUL_PCE,
        ))
    return specs


# ==============================================================================
# Data loading
# ==============================================================================
def load_trajectory_matrix() -> pd.DataFrame:
    """Load the healthy cohort and build the normalized-increment matrix."""
    df_healthy = pd.read_parquet(FILE_HEALTHY_COHORT)
    artifacts = joblib.load(FILE_SCREENING_ARTIFACTS)
    healthy_cohort = artifacts["healthy_cohort"]
    df_daily, _ = build_trajectory_matrix(df_healthy)
    df_daily = df_daily[df_daily["cell_name"].isin(healthy_cohort)].copy()
    return df_daily


# ==============================================================================
# Q1 — Descriptive statistics and signal strength
# ==============================================================================
def describe_target(y: pd.Series) -> pd.Series:
    n = len(y)
    return pd.Series({
        "n_samples": n,
        "mean": y.mean(),
        "median": y.median(),
        "std": y.std(),
        "min": y.min(),
        "max": y.max(),
        "skew": skew(y),
        "pct_zero": 100 * (y == 0).sum() / n,
        "pct_negative": 100 * (y < 0).sum() / n,
    })


def signal_strength_report(
    X: pd.DataFrame,
    y: pd.Series,
    lag_feature: str,
) -> dict:
    """
    Compute simple, interpretable signal-strength metrics on the target:
      - Pearson correlation with the lag feature (autodependence).
      - Fraction of zero values (degenerate target).
      - Relative std (std / |mean|), to gauge dynamic range.
      - Skewness.
    """
    corr_with_lag = float(X[lag_feature].corr(y)) if lag_feature in X.columns else np.nan
    pct_zero = float(100 * (y == 0).sum() / len(y))
    rel_std = float(y.std() / (abs(y.mean()) + 1e-9))
    return {
        "pearson_with_lag": corr_with_lag,
        "pct_zero": pct_zero,
        "relative_std": rel_std,
        "skew": float(skew(y)),
    }


# ==============================================================================
# Q2 — Model zoo and cross-validation
# ==============================================================================
def build_model_zoo(xgb_params: Mapping) -> dict[str, BaseEstimator]:
    xgb_alt_reg = dict(xgb_params)
    xgb_alt_reg["reg_lambda"] = 100.0
    xgb_alt_reg["reg_alpha"] = 1.0

    return {
        "Dummy(mean)": DummyRegressor(strategy="mean"),
        "Dummy(median)": DummyRegressor(strategy="median"),
        "Ridge(alpha=0.1)": make_pipeline(StandardScaler(), Ridge(alpha=0.1, random_state=RANDOM_STATE)),
        "Ridge(alpha=1.0)": make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=RANDOM_STATE)),
        "Ridge(alpha=10.0)": make_pipeline(StandardScaler(), Ridge(alpha=10.0, random_state=RANDOM_STATE)),
        "Ridge(alpha=100.0)": make_pipeline(StandardScaler(), Ridge(alpha=100.0, random_state=RANDOM_STATE)),
        "RandomForest(200,md4)": RandomForestRegressor(
            n_estimators=200, max_depth=4, min_samples_leaf=5,
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "XGBoost(production)": xgb.XGBRegressor(**xgb_params),
        "XGBoost(reg_lambda=100)": xgb.XGBRegressor(**xgb_alt_reg),
    }


def evaluate_cv(
    models: Mapping[str, BaseEstimator],
    X: pd.DataFrame,
    y: pd.Series,
    cv,
    cv_label: str,
) -> pd.DataFrame:
    rows = []
    for name, estimator in models.items():
        result = cross_validate(clone(estimator), X, y, cv=cv, scoring=SCORING, n_jobs=-1)
        row: dict = {"cv": cv_label, "model": name}
        for metric, sign in METRIC_SIGNS.items():
            vals = result[f"test_{metric}"]
            row[f"{metric}_mean"] = sign * vals.mean()
            row[f"{metric}_std"] = vals.std()
        rows.append(row)
    return pd.DataFrame(rows).sort_values("MAE_mean").reset_index(drop=True)


def print_cv_table(df_scores: pd.DataFrame, baseline_mae: float) -> None:
    for _, r in df_scores.iterrows():
        delta = baseline_mae - r["MAE_mean"]
        print(
            f"  [{r['cv']:>10s}] {r['model']:<28s} "
            f"MAE={r['MAE_mean']:.5f}±{r['MAE_std']:.5f}  "
            f"RMSE={r['RMSE_mean']:.5f}  MedAE={r['MedAE_mean']:.5f}  "
            f"R2={r['R2_mean']:+.4f}  ΔMAE_vs_baseline={delta:+.5f}"
        )


# ==============================================================================
# Q3 — Where does the error concentrate?
# ==============================================================================
def per_cell_breakdown(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    cv,
) -> pd.DataFrame:
    """
    Out-of-fold predictions per cell, aggregated to per-cell MAE and bias.
    Requires GroupKFold-like split or any cv that produces out-of-fold preds.
    """
    oof = cross_val_predict(clone(model), X, y, cv=cv, n_jobs=-1)
    df = pd.DataFrame({
        "cell_name": groups.values,
        "y_true": y.values,
        "y_pred": oof,
    })
    df["abs_err"] = (df["y_true"] - df["y_pred"]).abs()
    df["bias"] = df["y_true"] - df["y_pred"]

    agg = df.groupby("cell_name").agg(
        n=("y_true", "size"),
        mae=("abs_err", "mean"),
        bias=("bias", "mean"),
    ).sort_values("mae", ascending=False)
    return agg


def per_phase_breakdown(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    exposure: pd.Series,
    cv,
    n_bins: int = 4,
) -> pd.DataFrame:
    """
    Out-of-fold error aggregated across quantile bins of the exposure day.
    Reveals whether the model is systematically better/worse at early vs
    late stages of the cell's life.
    """
    oof = cross_val_predict(clone(model), X, y, cv=cv, n_jobs=-1)
    df = pd.DataFrame({
        "exposure": exposure.values,
        "y_true": y.values,
        "y_pred": oof,
    })
    df["abs_err"] = (df["y_true"] - df["y_pred"]).abs()

    try:
        df["phase"] = pd.qcut(df["exposure"], q=n_bins, labels=[
            "Q1 (earliest)", "Q2", "Q3", "Q4 (latest)",
        ], duplicates="drop")
    except ValueError:
        return pd.DataFrame()

    agg = df.groupby("phase", observed=True).agg(
        n=("y_true", "size"),
        exposure_range_min=("exposure", "min"),
        exposure_range_max=("exposure", "max"),
        mae=("abs_err", "mean"),
    )
    return agg


# ==============================================================================
# Q4 — Feature importance
# ==============================================================================
def compute_feature_importances(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
) -> pd.DataFrame:
    corr = X.corrwith(y)
    fitted = clone(model).fit(X, y)
    gain = pd.Series(fitted.feature_importances_, index=X.columns)

    perm = cast(
        Bunch,
        permutation_importance(
            fitted, X, y,
            n_repeats=N_PERMUTATION_REPEATS,
            random_state=RANDOM_STATE,
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
        ),
    )
    perm_series = pd.Series(perm.importances_mean, index=X.columns)

    return pd.DataFrame({
        "pearson_corr": corr,
        "xgb_gain": gain,
        "permutation_importance": perm_series,
    }).sort_values("permutation_importance", ascending=False)


# ==============================================================================
# Overfit gap
# ==============================================================================
def compute_overfit_gap(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    cv_mae: float,
) -> tuple[float, float, str]:
    fitted = clone(model).fit(X, y)
    train_mae = mean_absolute_error(y, fitted.predict(X))
    gap = cv_mae - train_mae
    verdict = "possible overfit" if gap > OVERFIT_GAP_RATIO * cv_mae else "reasonable gap"
    return train_mae, gap, verdict


# ==============================================================================
# Verdict per parameter
# ==============================================================================
def parameter_verdict(
    best_model: str,
    best_mae: float,
    baseline_mae: float,
    signal: dict,
) -> str:
    """
    Classify the parameter's predictability using both the CV improvement and
    the signal-strength metrics computed above.
    """
    improvement_pct = 100 * (baseline_mae - best_mae) / baseline_mae if baseline_mae > 0 else 0.0

    if improvement_pct < SIGNAL_THRESHOLDS["none"]:
        cv_verdict = "NO SIGNAL"
    elif improvement_pct < SIGNAL_THRESHOLDS["weak"]:
        cv_verdict = "WEAK SIGNAL"
    else:
        cv_verdict = "REAL SIGNAL"

    lag_corr = signal["pearson_with_lag"]
    if np.isnan(lag_corr):
        auto_verdict = "lag unavailable"
    elif abs(lag_corr) < AUTOCORR_MIN:
        auto_verdict = "no autocorrelation"
    else:
        auto_verdict = "autocorrelated"

    print(f"  Best model (KFold)   : {best_model} (MAE={best_mae:.5f})")
    print(f"  Improvement vs base  : {improvement_pct:+.1f}%")
    print(f"  Signal with own lag  : r={lag_corr:+.4f}  ({auto_verdict})")
    print(f"  Degenerate target?   : {signal['pct_zero']:.1f}% zeros, "
          f"rel_std={signal['relative_std']:.3f}, skew={signal['skew']:+.2f}")
    print(f"  => {cv_verdict}")

    return cv_verdict


# ==============================================================================
# Per-parameter diagnostic
# ==============================================================================
def diagnose_target(df_daily: pd.DataFrame, spec: TargetSpec) -> dict:
    """
    Run the full diagnostic for one parameter and return a one-line summary
    dict for the final cross-parameter table.
    """
    print("=" * 90)
    print(f" TARGET: {spec.target_column} ({spec.key.upper()})")
    print("=" * 90)

    summary = {
        "parameter": spec.key.upper(),
        "target": spec.target_column,
        "verdict": "SKIPPED",
        "mae_sensor": np.nan,
        "mae_api": np.nan,
    }

    if spec.target_column not in df_daily.columns:
        print(f"  [SKIP] Column '{spec.target_column}' not present in the matrix.\n")
        return summary

    missing = [f for f in spec.features if f not in df_daily.columns]
    if missing:
        print(f"  [SKIP] Missing features: {missing}\n")
        return summary

    X = df_daily[spec.features]
    y = df_daily[spec.target_column]
    groups = df_daily["cell_name"] if "cell_name" in df_daily.columns else None
    exposure = df_daily["Exposure_Days"] if "Exposure_Days" in df_daily.columns else None

    # ---------------------------------------------------------------- Q1
    print("\n--- Q1. Target descriptive statistics ---")
    print(describe_target(y).to_string())
    print("\n  Signal strength:")
    signal = signal_strength_report(X, y, spec.lag_feature)
    print(f"    Pearson with own lag : {signal['pearson_with_lag']:+.4f}")
    print(f"    Fraction of zeros    : {signal['pct_zero']:.2f}%")
    print(f"    Relative std         : {signal['relative_std']:.4f}")
    print(f"    Skewness             : {signal['skew']:+.4f}")

    # ---------------------------------------------------------------- Q2
    print("\n--- Q2. Baseline vs models: KFold(shuffle) vs GroupKFold(cell_name) ---")
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    models = build_model_zoo(dict(spec.xgb_params))

    scores_kfold = evaluate_cv(models, X, y, kf, "KFold")
    baseline_mae = scores_kfold.loc[scores_kfold["model"] == "Dummy(mean)", "MAE_mean"].iloc[0]
    print_cv_table(scores_kfold, baseline_mae)

    if groups is None:
        print("  (GroupKFold omitted: 'cell_name' column missing)")
    elif groups.nunique() < 2:
        print("  (GroupKFold omitted: less than 2 unique cells)")
    else:
        gkf = GroupKFold(n_splits=min(N_SPLITS, groups.nunique()))
        scores_groupkfold = evaluate_cv(models, X, y, list(gkf.split(X, y, groups)), "GroupKFold")
        print()
        print_cv_table(scores_groupkfold, baseline_mae)

    # ---------------------------------------------------------------- Q3
    print("\n--- Q3a. Per-cell error breakdown (XGBoost production, out-of-fold) ---")
    prod_model = xgb.XGBRegressor(**spec.xgb_params)
    if groups is not None:
        cell_table = per_cell_breakdown(prod_model, X, y, groups, kf)
        print(cell_table.to_string())
    else:
        print("  (Skipped: no cell_name column)")

    print("\n--- Q3b. Per-exposure-phase error breakdown ---")
    if exposure is not None:
        phase_table = per_phase_breakdown(prod_model, X, y, exposure, kf)
        if phase_table.empty:
            print("  (Skipped: insufficient data for quantile binning)")
        else:
            print(phase_table.to_string())
    else:
        print("  (Skipped: no Exposure_Days column)")

    # ---------------------------------------------------------------- Q4
    print("\n--- Q4. Feature importance (Pearson / XGBoost gain / permutation) ---")
    print(compute_feature_importances(prod_model, X, y).to_string())

    # ---------------------------------------------------------------- gap
    print("\n--- Overfitting gap (XGBoost production) ---")
    cv_mae_prod = scores_kfold.loc[scores_kfold["model"] == "XGBoost(production)", "MAE_mean"].iloc[0]
    train_mae, gap, overfit_verdict = compute_overfit_gap(prod_model, X, y, cv_mae_prod)
    print(f"  MAE full train     : {train_mae:.5f}")
    print(f"  MAE CV (KFold)     : {cv_mae_prod:.5f}")
    print(f"  Gap (CV - train)   : {gap:+.5f} ({overfit_verdict})")

    # ---------------------------------------------------------------- verdict
    print("\n--- Verdict ---")
    best_row = scores_kfold.iloc[0]
    verdict = parameter_verdict(
        best_row["model"], best_row["MAE_mean"], baseline_mae, signal,
    )
    print()

    summary.update({
        "verdict": verdict,
        "mae_best_cv": float(best_row["MAE_mean"]),
        "baseline_mae": float(baseline_mae),
        "improvement_pct": float(100 * (baseline_mae - best_row["MAE_mean"]) / baseline_mae),
        "best_model": best_row["model"],
    })
    return summary


# ==============================================================================
# Global summary across parameters
# ==============================================================================
def print_global_summary(summaries: list[dict]) -> None:
    print("=" * 90)
    print(" GLOBAL SUMMARY — PREDICTABILITY OF EACH PHYSICAL PARAMETER")
    print("=" * 90)

    df = pd.DataFrame(summaries)
    display_cols = [
        c for c in [
            "parameter", "best_model", "mae_best_cv", "baseline_mae",
            "improvement_pct", "verdict",
        ] if c in df.columns
    ]
    if display_cols:
        print(df[display_cols].to_string(index=False))
    print("=" * 90)
    print()


# ==============================================================================
# Entry point
# ==============================================================================
def _run_audit() -> None:
    logger.info("Loading trajectory feature matrix for multivariate audit...")
    df_daily = load_trajectory_matrix()
    logger.info(f"Rows: {len(df_daily)} | Cells: {df_daily['cell_name'].nunique()}")

    summaries: list[dict] = []
    for spec in _build_specs():
        summaries.append(diagnose_target(df_daily, spec))

    print_global_summary(summaries)


def main() -> None:
    AUDIT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    original_stdout = sys.stdout
    with open(AUDIT_REPORT_PATH, "w", encoding="utf-8") as report_file:
        sys.stdout = _Tee(original_stdout, report_file)
        try:
            _run_audit()
        finally:
            sys.stdout = original_stdout

    logger.info(f"Audit report saved to {AUDIT_REPORT_PATH}")


if __name__ == "__main__":
    main()
"""
Module: src/rul_xgb_audit.py
Description: Empirical audit of the isolated XGBoost component on the daily
             PCE damage-increment target. Quantifies the real contribution of
             Machine Learning against trivial baselines and characterises
             residual structure, overfitting gap, and per-cell systematic
             bias. Companion diagnostic to 08_mppt_rul_forecasting.py.

             Scope note: only the PCE damage increment is audited. That is the
             sole increment-style target the pipeline actually consumes (module
             08). pFF is used by module 07 as an absolute Digital Twin target
             (different formulation, different model), and FF/Jsc/Voc are
             forecast by module 09 with a different engine (normalized
             increments). Auditing pFF's daily increment here would test a
             formulation that no production module uses.

             The full audit report is automatically written to
             `outputs/diagnostics/xgb_audit_summary.txt` while still being
             streamed to stdout.
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
    BaseCrossValidator,
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
    FILE_T80_TRUTH,
    FILE_SCREENING_ARTIFACTS,
    DIAGNOSTICS_RUL_DIR,
)

# ------------------------------------------------------------------------------
# Module-level logging
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("XGB-Audit")


# ------------------------------------------------------------------------------
# Dynamic import of the RUL module (its filename starts with a digit, which is
# not importable via regular `from src.08_... import X` syntax).
# ------------------------------------------------------------------------------
_rul_mod = importlib.import_module("src.08_mppt_rul_forecasting")
FEATURES_RUL_PCE: list[str] = _rul_mod.FEATURES_RUL_PCE
build_rul_matrix = _rul_mod.build_rul_matrix


# ------------------------------------------------------------------------------
# Audit configuration
# ------------------------------------------------------------------------------
N_SPLITS = 5
N_PERMUTATION_REPEATS = 20
OVERFIT_GAP_RATIO = 0.30
VERDICT_THRESHOLDS = {"weak": 3.0, "real": 10.0}

# Auto-export path for the human-readable audit report.
AUDIT_REPORT_PATH: Path = DIAGNOSTICS_RUL_DIR / "xgb_audit_summary.txt"

SCORING = {
    "MAE": make_scorer(mean_absolute_error, greater_is_better=False),
    "RMSE": make_scorer(root_mean_squared_error, greater_is_better=False),
    "MedAE": make_scorer(median_absolute_error, greater_is_better=False),
    "R2": make_scorer(r2_score),
}
# Sign flips applied to `cross_validate` outputs so all metrics are reported
# with their natural orientation (higher is better).
METRIC_SIGNS = {"MAE": -1, "RMSE": -1, "MedAE": -1, "R2": 1}


# ------------------------------------------------------------------------------
# Stdout tee utility
# ------------------------------------------------------------------------------
class _Tee:
    """
    File-like object that duplicates every write/flush to multiple streams.

    Used to mirror the audit's stdout to a persistent text report without
    touching a single print() statement in the diagnostic code.
    """

    def __init__(self, *streams: TextIO) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


# ------------------------------------------------------------------------------
# Target specification
# ------------------------------------------------------------------------------
@dataclass(frozen=True)
class TargetSpec:
    key: str
    target_column: str
    features: list[str]
    xgb_params: Mapping[str, float | int]


# Only PCE is audited (see module docstring for the rationale).
TARGETS = [
    TargetSpec("pce", "Daily_Damage_Increment", list(FEATURES_RUL_PCE), XGB_PARAMS_RUL_PCE),
]


# ------------------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------------------
def load_daily_matrix() -> pd.DataFrame:
    """
    Load the action-window scored dataset, inject PCE_initial from the T80
    ground truth when missing, and build the daily RUL feature matrix.
    """
    df_twin = pd.read_parquet(FILE_HEALTHY_COHORT)
    t80_metrics = pd.read_parquet(FILE_T80_TRUTH)
    healthy_cohort = joblib.load(FILE_SCREENING_ARTIFACTS)["healthy_cohort"]

    # Inject PCE_initial (same pattern as the RUL forecasting module).
    if "PCE_initial" not in df_twin.columns:
        df_twin = df_twin.merge(
            t80_metrics[["PCE_initial"]],
            left_on="cell_name",
            right_index=True,
            how="left",
        )

    df_daily = build_rul_matrix(df_twin, healthy_cohort)
    return df_daily


# ------------------------------------------------------------------------------
# Descriptive statistics
# ------------------------------------------------------------------------------
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


# ------------------------------------------------------------------------------
# Model zoo and cross-validation
# ------------------------------------------------------------------------------
def build_model_zoo(xgb_params: Mapping) -> dict[str, BaseEstimator]:
    """
    Construct the comparison zoo: trivial baselines, linear models, tree
    ensembles, and two XGBoost variants (production params and an
    alternatively-regularised counterpart).
    """
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
            n_estimators=200,
            max_depth=4,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGBoost(producción)": xgb.XGBRegressor(**xgb_params),
        "XGBoost(reg_lambda=100)": xgb.XGBRegressor(**xgb_alt_reg),
    }


def evaluate_cv(
    models: Mapping[str, BaseEstimator],
    X: pd.DataFrame,
    y: pd.Series,
    cv: BaseCrossValidator | list,
    cv_label: str,
) -> pd.DataFrame:
    """Run `cross_validate` for every model in the zoo under a given CV scheme."""
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


# ------------------------------------------------------------------------------
# Feature importance
# ------------------------------------------------------------------------------
def compute_feature_importances(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
) -> pd.DataFrame:
    """
    Report three complementary importance signals:
      - Pearson correlation with the target,
      - XGBoost internal gain,
      - Permutation importance (20 repetitions, scoring = -MAE).
    """
    corr = X.corrwith(y)
    fitted = clone(model).fit(X, y)
    gain = pd.Series(fitted.feature_importances_, index=X.columns)

    # With a single scorer, `permutation_importance` returns a Bunch (the
    # dict[str, Bunch] variant only applies when scoring is a list or dict);
    # the cast silences the Union type declared in the stub.
    perm = cast(
        Bunch,
        permutation_importance(
            fitted,
            X,
            y,
            n_repeats=N_PERMUTATION_REPEATS,
            random_state=RANDOM_STATE,
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
        ),
    )
    perm_series = pd.Series(perm.importances_mean, index=X.columns)

    df_importance = pd.DataFrame({
        "pearson_corr": corr,
        "xgb_gain": gain,
        "permutation_importance": perm_series,
    })
    return df_importance.sort_values("permutation_importance", ascending=False)


# ------------------------------------------------------------------------------
# Overfitting gap
# ------------------------------------------------------------------------------
def compute_overfit_gap(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    cv_mae: float,
) -> tuple[float, float, str]:
    fitted = clone(model).fit(X, y)
    train_mae = mean_absolute_error(y, fitted.predict(X))
    gap = cv_mae - train_mae
    verdict = "posible sobreajuste" if gap > OVERFIT_GAP_RATIO * cv_mae else "gap razonable"
    return train_mae, gap, verdict


# ------------------------------------------------------------------------------
# Residual diagnostics
# ------------------------------------------------------------------------------
def compute_residual_diagnostics(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    cv: BaseCrossValidator,
    groups: pd.Series | None,
) -> tuple[np.ndarray, pd.Series, pd.Series | None]:
    """Compute out-of-fold predictions, residuals, and their correlation with
    each covariate (and per-cell mean bias, if groups are provided)."""
    oof_pred = cross_val_predict(clone(model), X, y, cv=cv, n_jobs=-1)
    residuals = y.to_numpy() - oof_pred
    resid_corr = pd.Series(
        {col: np.corrcoef(X[col], residuals)[0, 1] for col in X.columns}
    ).sort_values(key=np.abs, ascending=False)
    resid_by_group = None
    if groups is not None:
        resid_by_group = pd.Series(residuals, index=X.index).groupby(groups).mean().sort_values()
    return residuals, resid_corr, resid_by_group


# ------------------------------------------------------------------------------
# Automatic verdict
# ------------------------------------------------------------------------------
def print_verdict(best_model: str, best_mae: float, baseline_mae: float) -> None:
    """
    Compute the automatic verdict for a target.

    A trivial Dummy winning the MAE ranking is a red flag: it means no
    non-trivial model found exploitable signal, and the apparent "improvement"
    over the Dummy(mean) baseline is just the Dummy(median) exploiting a
    skewed target. We surface that explicitly instead of declaring a false
    positive.
    """
    improvement_pct = 100 * (baseline_mae - best_mae) / baseline_mae

    if "Dummy" in best_model:
        verdict = "SIN SEÑAL RELEVANTE (el mejor modelo es un Dummy)"
    elif improvement_pct < VERDICT_THRESHOLDS["weak"]:
        verdict = "SIN SEÑAL RELEVANTE"
    elif improvement_pct < VERDICT_THRESHOLDS["real"]:
        verdict = "SEÑAL DÉBIL"
    else:
        verdict = "SEÑAL PREDICTIVA REAL"

    print(f"  Mejor modelo (KFold) : {best_model} (MAE={best_mae:.5f})")
    print(f"  Mejora vs baseline   : {improvement_pct:+.1f}%")
    print(f"  => {verdict}")


# ------------------------------------------------------------------------------
# Per-target diagnostic driver
# ------------------------------------------------------------------------------
def diagnose_target(df_daily: pd.DataFrame, spec: TargetSpec) -> None:
    print("=" * 90)
    print(f" TARGET: {spec.target_column} ({spec.key.upper()})")
    print("=" * 90)

    X = df_daily[spec.features]
    y = df_daily[spec.target_column]
    groups = df_daily["cell_name"] if "cell_name" in df_daily.columns else None

    # 1. Descriptive statistics
    print("\n--- 1. Estadística descriptiva del target ---")
    print(describe_target(y).to_string())

    # 2. Model comparison: KFold and GroupKFold
    print("\n--- 2. Baseline vs modelos: KFold(shuffle) vs GroupKFold(cell_name) ---")
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    models = build_model_zoo(dict(spec.xgb_params))

    scores_kfold = evaluate_cv(models, X, y, kf, "KFold")
    baseline_mae = scores_kfold.loc[scores_kfold["model"] == "Dummy(mean)", "MAE_mean"].iloc[0]
    print_cv_table(scores_kfold, baseline_mae)

    if groups is None:
        print("  (GroupKFold omitido: columna 'cell_name' ausente)")
    elif groups.nunique() < N_SPLITS:
        print(f"  (GroupKFold omitido: solo {groups.nunique()} celdas únicas, "
            f"se necesitan al menos {N_SPLITS} para {N_SPLITS}-fold)")
    else:
        gkf = GroupKFold(n_splits=min(N_SPLITS, groups.nunique()))
        scores_groupkfold = evaluate_cv(models, X, y, list(gkf.split(X, y, groups)), "GroupKFold")
        print()
        print_cv_table(scores_groupkfold, baseline_mae)

    # 3. Feature importance
    print("\n--- 3. Importancia de variables (Pearson / ganancia XGBoost / permutación) ---")
    prod_model = xgb.XGBRegressor(**spec.xgb_params)
    print(compute_feature_importances(prod_model, X, y).to_string())

    # 4. Overfitting gap
    print("\n--- 4. Overfitting gap (XGBoost producción) ---")
    cv_mae_prod = scores_kfold.loc[scores_kfold["model"] == "XGBoost(producción)", "MAE_mean"].iloc[0]
    train_mae, gap, overfit_verdict = compute_overfit_gap(prod_model, X, y, cv_mae_prod)
    print(f"  MAE train completo : {train_mae:.5f}")
    print(f"  MAE CV (KFold)     : {cv_mae_prod:.5f}")
    print(f"  Gap (CV - train)   : {gap:+.5f} ({overfit_verdict})")

    # 5. Out-of-fold residual diagnostics
    print("\n--- 5. Residuos fuera de muestra (cross_val_predict) ---")
    residuals, resid_corr, resid_by_cell = compute_residual_diagnostics(prod_model, X, y, kf, groups)
    print(f"  Sesgo global (media residuo) : {residuals.mean():+.6f}")
    print(f"  Std residuo                  : {residuals.std():.6f}")
    print("\n  Correlación residuo-feature:")
    print(resid_corr.to_string())
    if resid_by_cell is not None:
        print("\n  Sesgo medio del residuo por celda (out-of-fold):")
        print(resid_by_cell.to_string())

    # 6. Automatic verdict
    print("\n--- 6. Veredicto automático ---")
    best_row = scores_kfold.iloc[0]
    print_verdict(best_row["model"], best_row["MAE_mean"], baseline_mae)
    print()


# ------------------------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------------------------
def _run_audit() -> None:
    """Inner routine executed while stdout is teed to the report file."""
    logger.info("Loading daily feature matrix for XGBoost audit...")
    df_daily = load_daily_matrix()
    logger.info(f"Rows: {len(df_daily)} | Cells: {df_daily['cell_name'].nunique()}")

    for spec in TARGETS:
        diagnose_target(df_daily, spec)


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
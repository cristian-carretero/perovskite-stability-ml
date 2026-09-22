"""
Module: src/audit_rul_features.py
Description: Leave-one-feature-out (LOFO) ablation audit of the isolated
             XGBoost regressor used in module 08 (PCE RUL forecasting).

             Goal: quantify each feature's contribution to the LOOCV MAE on
             ΔDamage, BEFORE attempting full-engine retraining (Experiment B).

             Design (pre-registered, see DECISION CRITERIA below):
               - Baseline: XGBoost on all 6 features in FEATURES_RUL_PCE.
               - Ablation: for each feature, retrain on the remaining 5.
               - LOOCV: leave-one-CELL-out (whole cell, not row), so no
                 temporal leakage between train and test.
               - Metric: MAE on ΔDamage, per fold (mean-of-means).
               - Paired comparison: baseline and every ablation use the SAME
                 fold and the SAME test points → ΔMAE_fold is paired.
               - Uncertainty: SE of the mean paired ΔMAE across the N=4
                 folds. This is the correct error for the ablation decision;
                 the SE of the aggregate MAE would overstate it (pooled per
                 point) or understate it (unpaired per cell).
               - Feature importances (gain) are reported as a redundancy
                 cross-check: high gain + zero ΔMAE ⇒ feature duplicated by
                 another.

             Addendum (null model):
               The ablation decides whether each feature contributes to the
               trained XGBoost. But if the trained XGBoost is itself no better
               than a constant predictor, the whole ablation measures noise.
               This module therefore also compares the XGBoost baseline
               against the strongest constant predictor: the per-fold
               training median of ΔDamage. If the null model is within
               1·SE of the XGBoost baseline, the ablation is ininterpretable
               and Experiment B must not be run.

             Addendum (sign consistency):
               With N=4 folds and small ΔMAE values, the mean alone is a
               weak signal. The per-fold sign of ΔMAE is reported as a
               complementary, distribution-free indicator: a feature whose
               removal helps on 4/4 folds is a stronger elimination candidate
               than one whose mean is dominated by a single fold.

             Outputs:
               - outputs/diagnostics/rul/audit_rul_features_experimentA.parquet
               - outputs/diagnostics/rul/audit_rul_features_experimentA_baseline.parquet
               - outputs/diagnostics/rul/audit_rul_features_experimentA_null.parquet
               - outputs/diagnostics/rul/audit_rul_features_experimentA.png
"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

from src.config import (
    FILE_HEALTHY_COHORT,
    FILE_T80_TRUTH,
    FILE_SCREENING_ARTIFACTS,
    XGB_PARAMS_RUL_PCE,
    DIAGNOSTICS_RUL_DIR,
)

# Dynamic import (module filename starts with a digit)
rul = importlib.import_module("src.08_mppt_rul_forecasting")

FEATURES_FULL: List[str] = list(rul.FEATURES_RUL_PCE)
TARGET: str = "Daily_Damage_Increment"
build_rul_matrix = rul.build_rul_matrix

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("AuditRULFeatures")


# ==============================================================================
# PRE-REGISTERED DECISION CRITERIA
# ==============================================================================
# ΔMAE is defined as MAE_ablated − MAE_baseline, computed PAIRED per fold:
#     ΔMAE_fold = MAE_ablated(cell) − MAE_baseline(cell)
# Point estimate = mean of the per-fold deltas.
# Uncertainty    = std(deltas, ddof=1) / sqrt(N_folds).
#
# Interpretation:
#     ΔMAE > 0  → removing the feature HURTS  → feature carries signal.
#     ΔMAE < 0  → removing the feature HELPS  → feature injects noise.
#
# Thresholds:
#     ΔMAE ≥ 2·SE                → IMPORTANT   (keep)
#     SE ≤ ΔMAE < 2·SE           → INFORMATIVE (keep)
#     ΔMAE ≤ −SE                 → HARMFUL     (strong elimination candidate)
#     |ΔMAE| < SE and |ΔMAE| < PRACTICAL_FLOOR → REDUNDANT (eliminate)
#     otherwise within ±SE       → WEAK        (investigate)
PRACTICAL_FLOOR_DAYS: float = 0.005


def classify_delta(delta_mean: float, delta_se: float) -> str:
    """Apply the pre-registered decision rule to a single feature."""
    if not np.isfinite(delta_mean) or not np.isfinite(delta_se):
        return "UNDEFINED"
    if delta_mean <= -delta_se:
        return "HARMFUL"
    if delta_mean >= 2.0 * delta_se:
        return "IMPORTANT"
    if delta_mean >= delta_se:
        return "INFORMATIVE"
    if abs(delta_mean) < PRACTICAL_FLOOR_DAYS and abs(delta_mean) < delta_se:
        return "REDUNDANT"
    return "WEAK"


# ==============================================================================
# LOOCV EVALUATOR
# ==============================================================================
def evaluate_loocv(
    df_daily: pd.DataFrame,
    features: List[str],
    healthy_cohort: List[str],
) -> Dict[str, object]:
    """
    Leave-one-CELL-out evaluation of an XGBoost regressor trained to predict
    ΔDamage from `features`.

    Returns:
        per_fold_mae : {cell_name: MAE on that cell's test points}
        mean_mae     : mean of per-fold MAEs (mean-of-means, honest metric)
        pooled_mae   : MAE over all pooled test points (reference only)
        n_folds      : number of folds that produced predictions
    """
    per_fold: Dict[str, float] = {}
    all_abs_errs: List[np.ndarray] = []

    for test_cell in healthy_cohort:
        train_cells = [c for c in healthy_cohort if c != test_cell]
        df_train = df_daily[df_daily["cell_name"].isin(train_cells)]
        df_test = df_daily[df_daily["cell_name"] == test_cell]

        if df_train.empty or df_test.empty:
            logger.warning(f"[{test_cell}] Empty train/test slice. Skipping fold.")
            continue

        # Fixed seed (XGB_PARAMS_RUL_PCE) → same fit across reruns.
        model = xgb.XGBRegressor(**XGB_PARAMS_RUL_PCE)
        model.fit(df_train[features], df_train[TARGET])

        pred = model.predict(df_test[features])
        errs = np.abs(pred - df_test[TARGET].to_numpy(dtype=float))
        per_fold[test_cell] = float(np.mean(errs))
        all_abs_errs.append(errs)

    pooled = (
        float(np.mean(np.concatenate(all_abs_errs))) if all_abs_errs else float("nan")
    )
    mean_mae = (
        float(np.mean(list(per_fold.values()))) if per_fold else float("nan")
    )

    return {
        "per_fold_mae": per_fold,
        "mean_mae": mean_mae,
        "pooled_mae": pooled,
        "n_folds": len(per_fold),
    }


# ==============================================================================
# NULL MODEL (per-fold constant predictor)
# ==============================================================================
def null_model_loocv(
    df_daily: pd.DataFrame,
    healthy_cohort: List[str],
    target_col: str = TARGET,
) -> Dict[str, float]:
    """
    Per-fold MAE of the strongest constant predictor: the median of the
    training fold's target. The median (not the mean) minimises the MAE
    in-sample, so it is the strongest possible constant baseline. If the
    XGBoost baseline cannot beat it, the ablation is measuring noise.
    """
    per_fold: Dict[str, float] = {}
    for test_cell in healthy_cohort:
        train = df_daily[df_daily["cell_name"] != test_cell]
        test = df_daily[df_daily["cell_name"] == test_cell]
        if train.empty or test.empty:
            continue
        c = float(np.median(train[target_col]))
        errs = np.abs(test[target_col].to_numpy(dtype=float) - c)
        per_fold[test_cell] = float(np.mean(errs))
    return per_fold


def null_model_report(
    df_daily: pd.DataFrame,
    healthy_cohort: List[str],
    baseline: dict,
) -> Dict[str, object]:
    """
    Paired comparison of the XGBoost baseline vs. the per-fold constant.

    Returns a dict with the numbers, and prints the verdict.
    """
    null_per_fold = null_model_loocv(df_daily, healthy_cohort)
    null_mean = float(np.mean(list(null_per_fold.values()))) if null_per_fold else float("nan")

    common = [c for c in baseline["per_fold_mae"] if c in null_per_fold]
    diffs = np.array(
        [baseline["per_fold_mae"][c] - null_per_fold[c] for c in common],
        dtype=float,
    )
    delta_mean = float(np.mean(diffs)) if diffs.size else float("nan")
    delta_se = (
        float(np.std(diffs, ddof=1) / np.sqrt(len(diffs)))
        if len(diffs) > 1 else float("nan")
    )

    print("\n" + "=" * 88)
    print(" ADDENDUM — NULL MODEL COMPARISON (does XGBoost beat a constant?)")
    print("=" * 88)
    print(f"  NULL model (per-fold training median): MAE = {null_mean:.5f} d")
    print(f"  XGBoost baseline                     : MAE = {baseline['mean_mae']:.5f} d")
    print(
        f"  Paired Δ = XGB − null                : {delta_mean:+.5f} d "
        f"(SE = {delta_se:.5f})"
    )
    print()
    for c in common:
        print(
            f"    fold [{c}]:  XGB = {baseline['per_fold_mae'][c]:.5f}  "
            f"| null = {null_per_fold[c]:.5f}  "
            f"| Δ = {baseline['per_fold_mae'][c] - null_per_fold[c]:+.5f}"
        )

    if not np.isfinite(delta_se) or abs(delta_mean) < delta_se:
        verdict = (
            "INDISTINGUISHABLE — XGBoost no bate al nulo. "
            "La ablación es ininterpretable en esta cohorte."
        )
    elif delta_mean < -delta_se:
        verdict = (
            "XGBoost PIERDE contra el nulo. Hay bug o leak en el pipeline; "
            "investigar antes de continuar."
        )
    else:
        verdict = (
            "XGBoost bate al nulo por ≥1 SE. La ablación es interpretable "
            "pero marginal."
        )
    print(f"\n  VERDICT: {verdict}")
    print("=" * 88)

    return {
        "null_mean_mae": null_mean,
        "delta_xgb_minus_null": delta_mean,
        "delta_se": delta_se,
        "per_fold_null": null_per_fold,
        "verdict": verdict,
    }


# ==============================================================================
# BASELINE + ABLATION
# ==============================================================================
def run_ablation(
    df_daily: pd.DataFrame,
    healthy_cohort: List[str],
    features_full: List[str],
) -> Tuple[Dict[str, dict], dict]:
    """Run the baseline and one ablation per feature. Returns (results, baseline)."""
    print("\n" + "=" * 88)
    print(" EXPERIMENT A — LEAVE-ONE-FEATURE-OUT ABLATION (isolated XGBoost)")
    print("=" * 88)
    print(f"  Features ({len(features_full)}): {features_full}")
    print(f"  Target  : {TARGET}")
    print(f"  Folds   : leave-one-CELL-out over {len(healthy_cohort)} cells")
    print(f"  Params  : {XGB_PARAMS_RUL_PCE}")

    results: Dict[str, dict] = {}

    # --- Baseline -------------------------------------------------------------
    print(f"\n  [baseline] all {len(features_full)} features")
    baseline = evaluate_loocv(df_daily, features_full, healthy_cohort)
    results["__baseline__"] = baseline
    print(
        f"    mean_MAE   = {baseline['mean_mae']:.5f} days "
        f"(mean-of-means over {baseline['n_folds']} folds)"
    )
    print(f"    pooled_MAE = {baseline['pooled_mae']:.5f} days")
    for cell, mae in baseline["per_fold_mae"].items():
        print(f"      fold [{cell}]: MAE = {mae:.5f}")

    # --- Ablations ------------------------------------------------------------
    for feat in features_full:
        ablated = [f for f in features_full if f != feat]
        print(f"\n  [ablate] remove '{feat}' → {len(ablated)} features remain")
        res = evaluate_loocv(df_daily, ablated, healthy_cohort)
        results[feat] = res
        print(f"    mean_MAE   = {res['mean_mae']:.5f} days")
        print(f"    pooled_MAE = {res['pooled_mae']:.5f} days")

    return results, baseline


# ==============================================================================
# PAIRED ΔMAE TABLE
# ==============================================================================
def build_delta_table(
    results: Dict[str, dict],
    baseline: dict,
    features_full: List[str],
) -> pd.DataFrame:
    """
    For each feature, compute the paired per-fold ΔMAE and its SE.
    Baseline and ablation share folds and test points → paired comparison.
    """
    base_per_fold: Dict[str, float] = baseline["per_fold_mae"]

    rows = []
    for feat in features_full:
        abl_per_fold: Dict[str, float] = results[feat]["per_fold_mae"]

        common_cells = [c for c in base_per_fold if c in abl_per_fold]
        if not common_cells:
            continue

        deltas = np.array(
            [abl_per_fold[c] - base_per_fold[c] for c in common_cells],
            dtype=float,
        )
        n = len(deltas)
        delta_mean = float(np.mean(deltas))
        delta_se = (
            float(np.std(deltas, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
        )

        # Sign consistency: how often did removing the feature HELP?
        n_helped = int(np.sum(deltas < 0))

        rows.append({
            "feature": feat,
            "n_folds": n,
            "mae_baseline": baseline["mean_mae"],
            "mae_ablated": results[feat]["mean_mae"],
            "delta_mae_mean": delta_mean,
            "delta_mae_se": delta_se,
            "delta_ci_lo": delta_mean - delta_se,
            "delta_ci_hi": delta_mean + delta_se,
            "per_fold_deltas": list(deltas),
            "n_folds_helped": n_helped,
            "verdict": classify_delta(delta_mean, delta_se),
        })

    return (
        pd.DataFrame(rows)
        .sort_values("delta_mae_mean", ascending=False)
        .reset_index(drop=True)
    )


# ==============================================================================
# BASELINE FEATURE IMPORTANCE (redundancy cross-check)
# ==============================================================================
def baseline_gain_importance(
    df_daily: pd.DataFrame, features: List[str]
) -> pd.Series:
    """Fit XGBoost on all data and return gain-based importance per feature."""
    model = xgb.XGBRegressor(**XGB_PARAMS_RUL_PCE)
    model.fit(df_daily[features], df_daily[TARGET])
    gain = model.get_booster().get_score(importance_type="gain")
    return pd.Series(
        {f: float(gain.get(f, 0.0)) for f in features}
    ).sort_values(ascending=False)


# ==============================================================================
# REPORTING
# ==============================================================================
def print_summary(
    df: pd.DataFrame, baseline: dict, gain: pd.Series
) -> None:
    print("\n" + "=" * 88)
    print(" SUMMARY — PAIRED ΔMAE  (ablated − baseline, per fold)")
    print("=" * 88)
    print(
        f"  Baseline MAE = {baseline['mean_mae']:.5f} days "
        f"(mean-of-means over {baseline['n_folds']} folds)"
    )
    print(f"  Practical floor for 'worth keeping': {PRACTICAL_FLOOR_DAYS} days")
    print()
    print(
        f"  {'feature':<28s} {'ΔMAE':>10s} {'SE':>10s} "
        f"{'[−1SE,+1SE]':>24s}  {'helped':>7s}  {'verdict':<12s}"
    )
    print("  " + "-" * 96)
    for _, row in df.iterrows():
        ci = f"[{row['delta_ci_lo']:+.5f}, {row['delta_ci_hi']:+.5f}]"
        helped = f"{int(row['n_folds_helped'])}/{int(row['n_folds'])}"
        print(
            f"  {row['feature']:<28s} "
            f"{row['delta_mae_mean']:>+10.5f} "
            f"{row['delta_mae_se']:>10.5f} "
            f"{ci:>24s}  "
            f"{helped:>7s}  "
            f"{row['verdict']:<12s}"
        )

    print("\n  Sign consistency of per-fold ΔMAE ('helped' = removing it reduced MAE):")
    for _, row in df.iterrows():
        deltas = np.asarray(row["per_fold_deltas"], dtype=float)
        n_neg = int(np.sum(deltas < 0))
        n = len(deltas)
        flag = ""
        if n_neg == n:
            flag = "  ← unanimous elimination signal"
        elif n_neg == 0:
            flag = "  ← unanimous retention signal"
        print(f"    {row['feature']:<28s}  helped on {n_neg}/{n} folds{flag}")

    print("\n  Baseline XGBoost gain importance (redundancy cross-check):")
    total = gain.sum() or 1.0
    for feat, g in gain.items():
        print(f"    {feat:<28s} gain = {g:>10.4f}   ({100*g/total:5.1f}%)")
    print("=" * 88)


def make_figure(df: pd.DataFrame, baseline: dict, out_path: Path) -> None:
    """Forest plot of ΔMAE per feature, with ±1 SE error bars."""
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(df) + 2.5))

    colors = {
        "IMPORTANT":   "#1b7837",
        "INFORMATIVE": "#7fbf7b",
        "WEAK":        "#bdbdbd",
        "REDUNDANT":   "#f4a582",
        "HARMFUL":     "#b2182b",
        "UNDEFINED":   "#999999",
    }

    y = np.arange(len(df))
    for i, (_, row) in enumerate(df.iterrows()):
        c = colors.get(row["verdict"], "#999999")
        ax.errorbar(
            row["delta_mae_mean"], i,
            xerr=row["delta_mae_se"],
            fmt="o", color=c, ecolor=c,
            capsize=4, markersize=8, linewidth=1.6,
        )

    ax.axvline(0.0, color="black", linewidth=1.0)
    ax.axvline(PRACTICAL_FLOOR_DAYS, color="gray", linestyle=":", alpha=0.7)
    ax.axvline(-PRACTICAL_FLOOR_DAYS, color="gray", linestyle=":", alpha=0.7)

    ax.set_yticks(y)
    ax.set_yticklabels(df["feature"])
    ax.set_xlabel("ΔMAE (days)   →   positive means removing the feature hurts")
    ax.set_title(
        f"Experiment A — Leave-one-feature-out (isolated XGBoost, "
        f"N={baseline['n_folds']} folds)\n"
        f"Baseline MAE = {baseline['mean_mae']:.5f} days"
    )
    ax.grid(True, axis="x", alpha=0.3)
    ax.invert_yaxis()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ==============================================================================
# ENTRY POINT
# ==============================================================================
def _run() -> None:
    DIAGNOSTICS_RUL_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading inputs...")
    df_twin = pd.read_parquet(FILE_HEALTHY_COHORT)
    t80_metrics = pd.read_parquet(FILE_T80_TRUTH)
    healthy_cohort = joblib.load(FILE_SCREENING_ARTIFACTS)["healthy_cohort"]
    logger.info(f"Production cohort: {healthy_cohort}")

    if "PCE_initial" not in df_twin.columns:
        df_twin = df_twin.merge(
            t80_metrics[["PCE_initial"]],
            left_on="cell_name", right_index=True, how="left",
        )

    df_daily = build_rul_matrix(df_twin, healthy_cohort)
    logger.info(
        f"Built RUL matrix: {len(df_daily)} rows, "
        f"{df_daily['cell_name'].nunique()} cells"
    )

    # Sanity check: the matrix must be built once with the full feature set
    # and reused across every ablation. Never re-filter after removing a
    # feature — that would change the test set and break the pairing.
    results, baseline = run_ablation(df_daily, healthy_cohort, FEATURES_FULL)

    df_delta = build_delta_table(results, baseline, FEATURES_FULL)
    gain = baseline_gain_importance(df_daily, FEATURES_FULL)

    print_summary(df_delta, baseline, gain)

    # ---- Addendum: null model comparison ------------------------------------
    null_report = null_model_report(df_daily, healthy_cohort, baseline)

    # ---- Parquet: main delta table ------------------------------------------
    out_parquet = DIAGNOSTICS_RUL_DIR / "audit_rul_features_experimentA.parquet"
    df_out = df_delta.copy()
    df_out["per_fold_deltas"] = df_out["per_fold_deltas"].apply(
        lambda v: list(map(float, v))
    )
    df_out.to_parquet(out_parquet, index=False)
    logger.info(f"ΔMAE table saved → {out_parquet}")

    # ---- Parquet: baseline reference ----------------------------------------
    baseline_parquet = (
        DIAGNOSTICS_RUL_DIR / "audit_rul_features_experimentA_baseline.parquet"
    )
    pd.DataFrame([{
        "n_folds": baseline["n_folds"],
        "mean_mae": baseline["mean_mae"],
        "pooled_mae": baseline["pooled_mae"],
        **{f"fold_mae_{c}": v for c, v in baseline["per_fold_mae"].items()},
        **{f"gain_{f}": float(g) for f, g in gain.items()},
    }]).to_parquet(baseline_parquet, index=False)

    # ---- Parquet: null model comparison -------------------------------------
    null_parquet = (
        DIAGNOSTICS_RUL_DIR / "audit_rul_features_experimentA_null.parquet"
    )
    null_row: Dict[str, object] = {
        "null_mean_mae": null_report["null_mean_mae"],
        "xgb_mean_mae": baseline["mean_mae"],
        "delta_xgb_minus_null": null_report["delta_xgb_minus_null"],
        "delta_se": null_report["delta_se"],
        "verdict": null_report["verdict"],
    }
    for c, v in null_report["per_fold_null"].items():
        null_row[f"null_fold_mae_{c}"] = float(v)
    pd.DataFrame([null_row]).to_parquet(null_parquet, index=False)

    # ---- Figure --------------------------------------------------------------
    out_fig = DIAGNOSTICS_RUL_DIR / "audit_rul_features_experimentA.png"
    make_figure(df_delta, baseline, out_fig)
    logger.info(f"Figure saved → {out_fig}")


def main() -> None:
    _run()


if __name__ == "__main__":
    main()
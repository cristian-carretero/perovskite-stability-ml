"""
Module: src/audit_rul_engine_no_xgb.py
Description: Companion to audit_rul_features.py.

             Feature ablation showed the isolated XGBoost cannot beat a
             per-fold constant predictor on ΔDamage (null result). This
             script asks the ENGINE-level question: does the full kinematic
             engine produce a lower LOOCV MAE on RUL when its XGBoost
             velocity is replaced by a constant velocity derived from the
             training fold?

             Design:
               - Two velocity sources per LOOCV fold:
                   A) XGBoost trained on the other 3 cells (production)
                   B) Constant = mean of ΔDamage over the training cells
               - All other engine components (structural floor, soft-
                 countdown, temporal blend, API calibration) are identical
                 between A and B.
               - Metric: MAE on RUL, separately for Sensor and API modes,
                 both per-cell and pooled (to match module 08's own
                 evaluate_model_performance output).
               - Paired comparison per fold with SE across folds.

             Interpretation:
               - |ΔMAE| < 1·SE  → XGBoost is decorative. Consider simplifying.
               - ΔMAE > 1·SE    → XGBoost contributes even though it does
                                   not generalise in isolation.
               - ΔMAE < −1·SE   → XGBoost injects noise. Consider disabling.

             Outputs:
               - outputs/diagnostics/rul/audit_rul_engine_no_xgb.parquet
               - stdout report with per-cell and pooled MAE tables
"""

from __future__ import annotations

import contextlib
import importlib
import io
import logging
import sys
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

from src.config import (
    DIAGNOSTICS_RUL_DIR,
    FILE_HEALTHY_COHORT,
    FILE_SCREENING_ARTIFACTS,
    FILE_T80_TRUTH,
)

# Dynamic import (module filename starts with a digit).
rul = importlib.import_module("src.08_mppt_rul_forecasting")

build_rul_matrix = rul.build_rul_matrix
fetch_api_history = rul.fetch_api_history
train_rul_engine = rul.train_rul_engine
run_dynamic_backtesting = rul.run_dynamic_backtesting
evaluate_model_performance = rul.evaluate_model_performance

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("AuditRulEngineNoXGB")


@contextlib.contextmanager
def _quiet_stdout():
    """Suppress module 08's verbose per-anchor prints during the audit."""
    old = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = old


# ==============================================================================
# CONSTANT VELOCITY MOCK
# ==============================================================================
class ConstantVelocityModel:
    """
    Mock regressor that mimics xgb.XGBRegressor's predict() API.

    Returns a constant for every input row. When plugged into
    simulate_rul_kinematics(), it yields a velocity equal to that constant
    because the forward simulation accumulates `constant * N` over the
    N-day window and divides by N.

    No fit() needed: the constant is computed by the caller from the
    training fold's ΔDamage distribution.
    """

    def __init__(self, constant: float) -> None:
        self.constant = float(constant)

    def predict(self, X) -> np.ndarray:
        n = len(X) if hasattr(X, "__len__") else 1
        return np.full(n, self.constant, dtype=float)


# ==============================================================================
# MAE HELPERS
# ==============================================================================
def per_cell_mae(records: List[dict]) -> Dict[Tuple[str, str], float]:
    """
    Return {(cell, mode): MAE_on_RUL} from a list of backtesting records.

    MAE is computed per cell as mean(|RUL_Pred - (True_Survival_Days - Anchor_Day)|)
    over anchors with positive true RUL.
    """
    if not records:
        return {}
    df = pd.DataFrame(records)
    df["real_rul"] = df["True_Survival_Days"] - df["Anchor_Day"]
    df = df[(df["real_rul"] > 0) & df["RUL_Pred"].notna()]
    out: Dict[Tuple[str, str], float] = {}
    for (cell, mode), sub in df.groupby(["cell_name", "Type"]):
        errs = np.abs(
            sub["RUL_Pred"].to_numpy(dtype=float)
            - sub["real_rul"].to_numpy(dtype=float)
        )
        out[(str(cell), str(mode))] = float(np.mean(errs))
    return out


def pooled_mae(records: List[dict]) -> Dict[str, float]:
    """Pooled MAE per mode, matching module 08's evaluate_model_performance."""
    if not records:
        return {}
    df = pd.DataFrame(records)
    df["real_rul"] = df["True_Survival_Days"] - df["Anchor_Day"]
    df = df[(df["real_rul"] > 0) & df["RUL_Pred"].notna()]
    out: Dict[str, float] = {}
    for mode in ["Sensor", "API"]:
        sub = df[df["Type"] == mode]
        if not sub.empty:
            errs = np.abs(
                sub["RUL_Pred"].to_numpy(dtype=float)
                - sub["real_rul"].to_numpy(dtype=float)
            )
            out[mode] = float(np.mean(errs))
    return out


def paired_stats(
    a: List[float], b: List[float]
) -> Tuple[float, float, int]:
    """Return (mean_delta, se_delta, n) for the per-fold paired difference b - a."""
    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b, dtype=float)
    d = arr_b - arr_a
    n = len(d)
    mean = float(np.mean(d)) if n > 0 else float("nan")
    se = float(np.std(d, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return mean, se, n


# ==============================================================================
# MAIN
# ==============================================================================
def main() -> None:
    DIAGNOSTICS_RUL_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading inputs...")
    df_twin = pd.read_parquet(FILE_HEALTHY_COHORT)
    t80_metrics = pd.read_parquet(FILE_T80_TRUTH)

    artifact = joblib.load(FILE_SCREENING_ARTIFACTS)
    healthy_cohort = artifact["healthy_cohort"]
    target_days = artifact.get("t80_target_days", {})
    logger.info(f"Production cohort: {healthy_cohort}")

    if "PCE_initial" not in df_twin.columns:
        df_twin = df_twin.merge(
            t80_metrics[["PCE_initial"]],
            left_on="cell_name", right_index=True, how="left",
        )

    df_daily = build_rul_matrix(df_twin, healthy_cohort)
    df_api_raw = fetch_api_history(df_daily)
    logger.info(f"RUL matrix: {len(df_daily)} rows, {df_daily['cell_name'].nunique()} cells")

    print("\n" + "=" * 84)
    print(" ENGINE-LEVEL AUDIT — XGBoost velocity vs constant velocity")
    print("=" * 84)
    print("  Each fold trains both:")
    print("    A) XGBoost on the other 3 cells          (production)")
    print("    B) Constant = mean(ΔDamage_train)        (baseline)")
    print("  The rest of the engine is identical between A and B.")

    records_xgb: List[dict] = []
    records_const: List[dict] = []
    v_const_per_fold: Dict[str, float] = {}

    for cell in healthy_cohort:
        train_cells = [c for c in healthy_cohort if c != cell]
        df_train = df_daily[df_daily["cell_name"].isin(train_cells)]

        # ---- A) XGBoost engine ----
        model_xgb = train_rul_engine(df_train)
        with _quiet_stdout():
            rec_xgb = run_dynamic_backtesting(
                df_daily, df_api_raw, cell, model_xgb, t80_metrics,
                train_cells=train_cells, target_days=target_days,
            )
        records_xgb.extend(rec_xgb)

        # ---- B) Constant-velocity engine ----
        v_const = float(np.mean(df_train["Daily_Damage_Increment"]))
        v_const_per_fold[cell] = v_const
        model_const = ConstantVelocityModel(v_const)
        with _quiet_stdout():
            rec_const = run_dynamic_backtesting(
                df_daily, df_api_raw, cell, model_const, t80_metrics,
                train_cells=train_cells, target_days=target_days,
            )
        records_const.extend(rec_const)

        logger.info(
            f"[{cell}] fold done — v_const = {v_const:.5f}, "
            f"records: xgb={len(rec_xgb)}, const={len(rec_const)}"
        )

    # ---- Per-cell table ----
    mae_xgb = per_cell_mae(records_xgb)
    mae_const = per_cell_mae(records_const)

    print("\n" + "=" * 84)
    print(" PER-CELL MAE ON RUL (days)")
    print("=" * 84)
    header = (
        f"  {'cell':<15} {'mode':<8} {'MAE_xgb':>9} {'MAE_const':>11} "
        f"{'Δ (const−xgb)':>15} {'v_const':>10}"
    )
    print(header)
    print("  " + "-" * len(header.strip()))
    for cell in healthy_cohort:
        for mode in ["Sensor", "API"]:
            a = mae_xgb.get((cell, mode))
            b = mae_const.get((cell, mode))
            if a is None or b is None:
                continue
            print(
                f"  {cell:<15} {mode:<8} {a:>9.3f} {b:>11.3f} "
                f"{b - a:>+15.3f} {v_const_per_fold[cell]:>10.5f}"
            )

    # ---- Pooled MAE (matches module 08 output) ----
    pooled_xgb = pooled_mae(records_xgb)
    pooled_const = pooled_mae(records_const)

    print("\n" + "=" * 84)
    print(" POOLED MAE (matches module 08's evaluate_model_performance)")
    print("=" * 84)
    for mode in ["Sensor", "API"]:
        a = pooled_xgb.get(mode, float("nan"))
        b = pooled_const.get(mode, float("nan"))
        print(f"  [{mode:<6}] XGB = {a:.3f} d   |   const = {b:.3f} d   |   Δ = {b - a:+.3f} d")

    # ---- Paired comparison per mode ----
    print("\n" + "=" * 84)
    print(" PAIRED COMPARISON (mean-of-means across folds, with SE)")
    print("=" * 84)
    for mode in ["Sensor", "API"]:
        cells_with_both = [
            c for c in healthy_cohort
            if (c, mode) in mae_xgb and (c, mode) in mae_const
        ]
        a_list = [mae_xgb[(c, mode)] for c in cells_with_both]
        b_list = [mae_const[(c, mode)] for c in cells_with_both]
        mean_a = float(np.mean(a_list)) if a_list else float("nan")
        mean_b = float(np.mean(b_list)) if b_list else float("nan")
        mean_d, se_d, n = paired_stats(a_list, b_list)

        print(f"\n  [{mode}]  MAE_XGB = {mean_a:.3f} d   |   MAE_const = {mean_b:.3f} d")
        print(f"           Δ (const − xgb) = {mean_d:+.3f} d   (SE = {se_d:.3f}, N = {n} folds)")

        if not np.isfinite(se_d):
            verdict = "INDEFINIDO (pocos folds)"
        elif abs(mean_d) < se_d:
            verdict = "INDISTINGUIBLE — el XGBoost no aporta señal distinguible del ruido"
        elif mean_d > se_d:
            verdict = "XGBoost APORTA — el motor pierde precisión sin él"
        else:
            verdict = "XGBoost PERJUDICA — el motor gana al sustituirlo por constante"
        print(f"           VERDICT: {verdict}")

    # ---- Persist ----
    rows: List[dict] = []
    for cell in healthy_cohort:
        for mode in ["Sensor", "API"]:
            if (cell, mode) not in mae_xgb or (cell, mode) not in mae_const:
                continue
            rows.append({
                "cell": cell,
                "mode": mode,
                "mae_xgb": mae_xgb[(cell, mode)],
                "mae_const": mae_const[(cell, mode)],
                "delta_const_minus_xgb": mae_const[(cell, mode)] - mae_xgb[(cell, mode)],
                "v_const": v_const_per_fold.get(cell, float("nan")),
            })
    df_out = pd.DataFrame(rows)
    out_parquet = DIAGNOSTICS_RUL_DIR / "audit_rul_engine_no_xgb.parquet"
    df_out.to_parquet(out_parquet, index=False)
    logger.info(f"Results saved → {out_parquet}")

    print("\n" + "=" * 84)
    print(" AUDIT COMPLETE")
    print("=" * 84)


if __name__ == "__main__":
    main()
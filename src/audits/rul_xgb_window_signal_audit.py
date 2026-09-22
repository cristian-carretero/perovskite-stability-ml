"""
Module: src/rul_xgb_window_signal_audit.py
Description: Audit of the isolated XGBoost ML signal as a function of the
             rolling-median smoothing window W.

             Companion diagnostic to rul_xgb_audit.py. Where that module
             audits the production W=2 in isolation, this one answers a
             different question across the full window grid:

                 "Does the ML component still beat a trivial predictor
                  for each candidate W, or does the rolling-median smoothing
                  eventually kill the signal?"

             For every W in SMOOTHING_WINDOW_GRID the script:
               1. Rebuilds the daily RUL feature matrix with that W
                  (build_rul_matrix, module 08).
               2. Computes the dispersion of the damage-increment target,
                  sigma(Delta D) = std(Daily_Damage_Increment).
               3. Trains an XGBoost regressor (production params) and a
                  Dummy(mean) baseline under identical KFold(5, shuffle).
               4. Reports the relative ML utility:
                       Delta_MAE% = 100 * (MAE_dummy - MAE_xgb) / MAE_dummy

             Rationale: the production choice W=2 is motivated by the API MAE
             (deployment regime, see 08_mppt_rul_forecasting). This audit gives
             the complementary ML-side argument: a small W preserves the
             high-frequency structure of the damage-increment target, letting
             XGBoost exploit real meteorological signal; a large W (>= 7)
             low-passes the target into a delayed derivative and the ML
             component loses its edge over the clock baseline.

             Outputs:
               - outputs/diagnostics/rul_xgb_audit_by_window.parquet
               - outputs/diagnostics/rul_xgb_audit_by_window.txt
"""

from __future__ import annotations

import importlib
import logging
import sys
from typing import TextIO

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import skew
from sklearn.dummy import DummyRegressor
from sklearn.metrics import make_scorer, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_validate

from src.config import (
    RANDOM_STATE,
    XGB_PARAMS_RUL_PCE,
    FILE_HEALTHY_COHORT,
    FILE_T80_TRUTH,
    FILE_SCREENING_ARTIFACTS,
    DIAGNOSTICS_RUL_DIR,
)

# ------------------------------------------------------------------------------
# Dynamic import of the RUL module (its filename starts with a digit, which
# is not importable via regular `from src.08_... import X` syntax).
# ------------------------------------------------------------------------------
rul = importlib.import_module("src.08_mppt_rul_forecasting")
FEATURES_RUL_PCE: list[str] = rul.FEATURES_RUL_PCE
build_rul_matrix = rul.build_rul_matrix


# ==============================================================================
# Configuration
# ==============================================================================
# Candidates mirror SMOOTHING_WINDOW_GRID in rul_calibration_optimizer.py so
# this audit is directly comparable to the W sweep reported in the paper.
SMOOTHING_WINDOW_GRID: list[int] = [1, 2, 3, 5, 7, 10, 14]
N_SPLITS: int = 5

REPORT_PATH = DIAGNOSTICS_RUL_DIR / "rul_xgb_audit_by_window.txt"
RESULTS_PATH = DIAGNOSTICS_RUL_DIR / "rul_xgb_audit_by_window.parquet"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("XGB-Audit-By-Window")


# ==============================================================================
# Stdout tee (same idiom as the other diagnostics in this repo)
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
# Scalar coercion helper (pandas-stubs workaround)
# ==============================================================================
def _as_int(value: object) -> int:
    """Coerce a pandas scalar (or 1-element Series) to a plain Python int.

    pandas-stubs type `Series.__getitem__(Hashable)` as `Any | Series[Any]`,
    so mypy rejects a direct `int(...)` even though at runtime the value is
    always a scalar. This helper centralises the coercion cleanly.
    """
    if isinstance(value, pd.Series):
        value = value.iloc[0]
    if isinstance(value, (int, float, np.integer, np.floating)):
        return int(value)
    raise TypeError(f"Expected numeric scalar, got {type(value).__name__}")


def _as_float(value: object) -> float:
    if isinstance(value, pd.Series):
        value = value.iloc[0]
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    raise TypeError(f"Expected numeric scalar, got {type(value).__name__}")


# ==============================================================================
# Data loading (same pattern as rul_xgb_audit.py)
# ==============================================================================
def load_twin() -> tuple[pd.DataFrame, list[str]]:
    """
    Load the healthy-cohort digital twin and inject PCE_initial from the T80
    ground truth if missing. Returns the raw twin and the cohort cell list;
    the daily matrix is rebuilt per W inside the sweep.
    """
    df_twin = pd.read_parquet(FILE_HEALTHY_COHORT)
    t80_metrics = pd.read_parquet(FILE_T80_TRUTH)
    healthy_cohort = joblib.load(FILE_SCREENING_ARTIFACTS)["healthy_cohort"]

    if "PCE_initial" not in df_twin.columns:
        df_twin = df_twin.merge(
            t80_metrics[["PCE_initial"]],
            left_on="cell_name",
            right_index=True,
            how="left",
        )
    return df_twin, healthy_cohort


# ==============================================================================
# Per-W evaluation
# ==============================================================================
SCORING = {
    "MAE": make_scorer(mean_absolute_error, greater_is_better=False),
    "R2": make_scorer(r2_score),
}


def evaluate_window(
    df_twin: pd.DataFrame,
    healthy_cohort: list[str],
    W: int,
) -> dict:
    """
    Rebuild the daily matrix with smoothing_window=W and evaluate XGBoost
    (production params) and Dummy(mean) under identical KFold splits.

    The CV scheme is KFold(shuffle, K=5) — matching rul_xgb_audit.py, where
    GroupKFold is omitted because the cohort has only 4 unique cells.
    """
    df_daily = build_rul_matrix(df_twin, healthy_cohort, smoothing_window=W)
    X = df_daily[FEATURES_RUL_PCE]
    y = df_daily["Daily_Damage_Increment"]

    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    xgb_model = xgb.XGBRegressor(**XGB_PARAMS_RUL_PCE)
    dummy = DummyRegressor(strategy="mean")

    res_xgb = cross_validate(xgb_model, X, y, cv=cv, scoring=SCORING, n_jobs=-1)
    res_dummy = cross_validate(dummy, X, y, cv=cv, scoring=SCORING, n_jobs=-1)

    mae_xgb = -float(res_xgb["test_MAE"].mean())
    mae_dummy = -float(res_dummy["test_MAE"].mean())
    r2_xgb = float(res_xgb["test_R2"].mean())

    # Relative ML utility: how much MAE the ML component removes on top of the
    # trivial mean predictor. Positive => signal alive; <= 0 => signal dead.
    delta_pct = (
        100.0 * (mae_dummy - mae_xgb) / mae_dummy if mae_dummy > 0 else float("nan")
    )

    return {
        "W": int(W),
        "n_samples": int(len(y)),
        "std_delta_damage": float(y.std()),
        "skew_delta_damage": float(skew(y)),
        "pct_zero_delta": float(100.0 * (y == 0).sum() / len(y)),
        "mae_dummy": mae_dummy,
        "mae_xgb": mae_xgb,
        "r2_xgb": r2_xgb,
        "delta_mae_pct": delta_pct,
    }


# ==============================================================================
# Reporting
# ==============================================================================
def print_report(df: pd.DataFrame) -> None:
    print("\n" + "=" * 104)
    print(" XGBoost SIGNAL AUDIT AS A FUNCTION OF THE SMOOTHING WINDOW W")
    print("=" * 104)
    print(
        " Target : Daily_Damage_Increment\n"
        " Models : XGBoost(production params)  vs  Dummy(mean)\n"
        f" CV     : KFold({N_SPLITS}, shuffle, seed={RANDOM_STATE})\n"
        " Metric : Delta_MAE% = 100 * (MAE_dummy - MAE_xgb) / MAE_dummy\n"
    )

    # Convert to a list of plain dicts so mypy sees `Any` on every access
    # (avoids the `Any | Series` union that pandas-stubs produce for
    #  `Series.__getitem__(Hashable)`).
    records: list[dict] = df.to_dict(orient="records")

    header = (
        f"  {'W':>3s}  {'N':>5s}  {'sigma(dD)':>11s}  {'skew(dD)':>9s}  "
        f"{'%zero(dD)':>10s}  {'MAE_dummy':>11s}  {'MAE_xgb':>11s}  "
        f"{'R2_xgb':>8s}  {'dMAE vs Dummy':>14s}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    for rec in records:
        w_val = _as_int(rec["W"])
        if w_val == 2:
            flag = "   <-- produccion"
        elif _as_float(rec["delta_mae_pct"]) <= 0.0:
            flag = "   [!] senal muerta"
        elif _as_float(rec["delta_mae_pct"]) < 3.0:
            flag = "   [~] senal debil"
        else:
            flag = ""
        print(
            f"  {w_val:>3d}  {_as_int(rec['n_samples']):>5d}  "
            f"{_as_float(rec['std_delta_damage']):>11.5f}  "
            f"{_as_float(rec['skew_delta_damage']):>+9.3f}  "
            f"{_as_float(rec['pct_zero_delta']):>9.2f}%  "
            f"{_as_float(rec['mae_dummy']):>11.5f}  "
            f"{_as_float(rec['mae_xgb']):>11.5f}  "
            f"{_as_float(rec['r2_xgb']):>+8.4f}  "
            f"{_as_float(rec['delta_mae_pct']):>+13.2f}%{flag}"
        )
    print("=" * 104)

    # Highlight the optimum and the W=7 row explicitly.
    best_rec = max(records, key=lambda r: _as_float(r["delta_mae_pct"]))
    best_w = _as_int(best_rec["W"])
    best_delta = _as_float(best_rec["delta_mae_pct"])
    best_mae_xgb = _as_float(best_rec["mae_xgb"])
    best_mae_dummy = _as_float(best_rec["mae_dummy"])

    w7_rec = next((r for r in records if _as_int(r["W"]) == 7), None)

    print("\n  Lectura:")
    print(
        f"    - Maxima utilidad del ML en W = {best_w}  "
        f"(dMAE = {best_delta:+.2f}%,  "
        f"MAE_xgb = {best_mae_xgb:.5f}  vs  MAE_dummy = {best_mae_dummy:.5f})"
    )
    if w7_rec is not None:
        w7_delta = _as_float(w7_rec["delta_mae_pct"])
        if w7_delta <= 0.0:
            estado = "MUERTA"
        elif w7_delta < 3.0:
            estado = "DEBIL"
        else:
            estado = "VIVA"
        print(
            f"    - En W = 7 el XGBoost aporta dMAE = {w7_delta:+.2f}%  "
            f"->  senal {estado}"
        )
    print()


# ==============================================================================
# Entry point
# ==============================================================================
def _run() -> None:
    logger.info("Loading digital twin and healthy cohort...")
    df_twin, healthy_cohort = load_twin()

    print("\n" + "=" * 104)
    print(" PER-W XGBoost AUDIT — rebuilding daily matrix and retraining per W")
    print("=" * 104)

    rows: list[dict] = []
    for W in SMOOTHING_WINDOW_GRID:
        print(f"  Evaluating W = {W:>2d} ...", flush=True)
        rows.append(evaluate_window(df_twin, healthy_cohort, W))

    df = pd.DataFrame(rows).sort_values("W").reset_index(drop=True)
    print_report(df)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(RESULTS_PATH, index=False)
    print(f"  Results table saved -> {RESULTS_PATH}")


def main() -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    original_stdout = sys.stdout
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        sys.stdout = _Tee(original_stdout, f)
        try:
            _run()
        finally:
            sys.stdout = original_stdout
    logger.info(f"Full audit report saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
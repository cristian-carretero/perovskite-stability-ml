"""
Module: src/rul_calibration_optimizer.py
Description: Empirical calibration of the eight coefficients of the hybrid
             kinematics engine in 08_mppt_rul_forecasting.py.

             Strategy:
               1. Pre-train the 4 blind XGBoost models (one per LOOCV fold).
               2. Pre-compute every simulation unit (cell x anchor x mode).
                  A simulation unit contains all the quantities that do NOT
                  depend on the kinematic coefficients, so a full LOOCV
                  re-evaluation reduces to re-running only the closed-form
                  simulation loop.
                3. Sweep coefficients via a 1D sensitivity analysis, then a
                  focused N-D TPE search (Optuna) over the same discrete
                  GRID_ND, using the COMBINED MAE
                  (0.5 * sensor + 0.5 * api) as the objective.
               4. Persist the winning coefficients to a JSON file that the 08
                  module loads automatically on its next run.

             In addition, a dedicated sweep of the smoothing window W is
             performed. Because W changes the daily feature matrix and the
             trained model, its anchors are frozen (using the canonical W=7
             matrix) so every candidate W is evaluated over exactly the same
             set of (cell, anchor) pairs.

             Outputs:
               - outputs/diagnostics/rul_coeffs_sensitivity.parquet
               - outputs/diagnostics/rul_coeffs_search.parquet
               - outputs/diagnostics/rul_coeffs_calibrated.json
               - outputs/diagnostics/rul_smoothing_window_sweep.parquet
               - outputs/diagnostics/rul_coeffs_optimization.txt (mirror of stdout)
"""

from __future__ import annotations

import importlib
import itertools
import logging
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from src.config import (
    BURN_IN_DAYS,
    FILE_HEALTHY_COHORT,
    FILE_T80_TRUTH,
    FILE_SCREENING_ARTIFACTS,
    FILE_RUL_COEFFS_CALIBRATED,
    DIAGNOSTICS_RUL_DIR,
)

# ------------------------------------------------------------------------------
# Dynamic import of the RUL module (its filename starts with a digit).
# ------------------------------------------------------------------------------
rul = importlib.import_module("src.08_mppt_rul_forecasting")

build_rul_matrix = rul.build_rul_matrix
fetch_api_history = rul.fetch_api_history
fit_calibration = rul.fit_calibration
apply_calibration = rul.apply_calibration
train_rul_engine = rul.train_rul_engine
simulate_rul_kinematics = rul.simulate_rul_kinematics

# The optimizer always uses the immutable hardcoded baseline as reference,
# regardless of whether a previously calibrated JSON exists.
PHI_0_DEFAULT = rul.PHI_0_HARDCODED
PHI_1_DEFAULT = rul.PHI_1_HARDCODED
PHI_2_DEFAULT = rul.PHI_2_HARDCODED
LAMBDA_W_DEFAULT = rul.LAMBDA_W_HARDCODED
MU_DEFAULT = rul.MU_HARDCODED
EPS_DEFAULT = rul.EPS_HARDCODED

# Temporal baseline blend (Strategy B)
K_BLEND_DEFAULT = 0.0   # 0 = pure engine, 1 = pure clock
T_REF_DEFAULT = 54.0    # median lifetime of the cohort (days)

SMOOTHING_WINDOW = rul.SMOOTHING_WINDOW
ANCHOR_SPACING   = rul.ANCHOR_SPACING
SIMULATION_WINDOW = rul.SIMULATION_WINDOW
T80_DAMAGE_LIMIT = rul.T80_DAMAGE_LIMIT


# ==============================================================================
# Logging + report paths
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("RUL-Calib-Optimizer")

REPORT_PATH = DIAGNOSTICS_RUL_DIR / "rul_coeffs_optimization.txt"
SENSITIVITY_PATH = DIAGNOSTICS_RUL_DIR / "rul_coeffs_sensitivity.parquet"
SEARCH_PATH = DIAGNOSTICS_RUL_DIR / "rul_coeffs_search.parquet"
CALIBRATED_COEFFS_PATH = FILE_RUL_COEFFS_CALIBRATED


class _Tee:
    """File-like object duplicating every write/flush to multiple streams."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)
        return len(data)

    def flush(self):
        for s in self._streams:
            s.flush()


# ==============================================================================
# Baseline registry (used for reporting only)
# ==============================================================================
BASELINE_REGISTRY = {
    "phi_0": PHI_0_DEFAULT,
    "phi_1": PHI_1_DEFAULT,
    "phi_2": PHI_2_DEFAULT,
    "lambda_w": LAMBDA_W_DEFAULT,
    "mu": MU_DEFAULT,
    "eps": EPS_DEFAULT,
    "k_blend": K_BLEND_DEFAULT,
    "t_ref": T_REF_DEFAULT,
}


# ==============================================================================
# Pre-computed simulation units
# ==============================================================================
@dataclass
class SimulationUnit:
    cell: str
    mode: str                                  # 'Sensor' or 'API'
    anchor_day: float
    current_damage: float
    rolling_irr: list
    rolling_temp: list
    future_weather: pd.DataFrame
    model: xgb.XGBRegressor
    true_survival_days: Optional[float]


def _coerce_optional_float(value) -> Optional[float]:
    """Return a plain float or None; used to satisfy strict type checkers."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(f):
        return None
    return f


def precompute_units(
    df_daily: pd.DataFrame,
    df_api_raw: pd.DataFrame,
    t80_metrics: pd.DataFrame,
    healthy_cohort: List[str],
    blind_models_by_fold: Dict[str, xgb.XGBRegressor],
    anchors_per_cell: Optional[Dict[str, List[float]]] = None,
) -> List[SimulationUnit]:
    """
    Pre-compute every (cell, anchor, mode) simulation unit. Everything that
    does NOT depend on the kinematic coefficients is computed once here.

    If `anchors_per_cell` is provided, the anchors are taken verbatim from it
    and the early-stop on `cum_damage >= T80_DAMAGE_LIMIT` is disabled. This
    guarantees that every candidate W in the smoothing-window sweep is
    evaluated over the same (cell, anchor) pairs.
    """
    units: List[SimulationUnit] = []

    for cell in healthy_cohort:
        cell_data = df_daily[df_daily["cell_name"] == cell].sort_values("Exposure_Days")
        if cell_data.empty:
            continue

        true_survival_days: Optional[float] = (
            _coerce_optional_float(t80_metrics.loc[cell, "survival_days_pce"])
            if cell in t80_metrics.index
            else None
        )

        if anchors_per_cell is not None and cell in anchors_per_cell:
            anchors = anchors_per_cell[cell]
            fixed_anchors = True
        else:
            max_days = cell_data["Exposure_Days"].max()
            anchors = list(range(int(BURN_IN_DAYS), int(max_days) + 1, ANCHOR_SPACING))
            if int(max_days) not in anchors:
                anchors.append(int(max_days))
            fixed_anchors = False

        train_cells = [c for c in healthy_cohort if c != cell]
        model = blind_models_by_fold[cell]

        for anchor in anchors:
            hist_cutoff = cell_data[cell_data["Exposure_Days"] <= anchor]
            if hist_cutoff.empty:
                continue
            actual_day = float(hist_cutoff.iloc[-1]["Exposure_Days"])
            cum_damage = float(hist_cutoff.iloc[-1]["Cumulative_Damage"])

            if not fixed_anchors and cum_damage >= T80_DAMAGE_LIMIT:
                break

            rolling_irr = list(hist_cutoff["Daily_Irradiance_Dose"].tail(SMOOTHING_WINDOW))
            rolling_temp = list(hist_cutoff["Daily_Max_Temp_C"].tail(SMOOTHING_WINDOW))

            # --- Sensor mode unit ---
            future_sensor = cell_data[cell_data["Exposure_Days"] > anchor].head(SIMULATION_WINDOW)
            if not future_sensor.empty:
                units.append(SimulationUnit(
                    cell=cell, mode="Sensor", anchor_day=actual_day,
                    current_damage=cum_damage,
                    rolling_irr=list(rolling_irr),
                    rolling_temp=list(rolling_temp),
                    future_weather=future_sensor,
                    model=model,
                    true_survival_days=true_survival_days,
                ))

            # --- API mode unit (walk-forward calibration does not depend on kinematic coeffs) ---
            if not df_api_raw.empty:
                anchor_date = hist_cutoff.iloc[-1]["Date_Day"]
                try:
                    reg_temp_wf, reg_irr_wf = fit_calibration(
                        df_daily, df_api_raw,
                        max_date=anchor_date,
                        train_cells=train_cells,
                        window_days=90,
                    )
                except ValueError:
                    continue

                future_raw = df_api_raw[df_api_raw["Date_Day"] > anchor_date].head(SIMULATION_WINDOW)
                if not future_raw.empty:
                    future_api = apply_calibration(reg_temp_wf, reg_irr_wf, future_raw)
                    units.append(SimulationUnit(
                        cell=cell, mode="API", anchor_day=actual_day,
                        current_damage=cum_damage,
                        rolling_irr=list(rolling_irr),
                        rolling_temp=list(rolling_temp),
                        future_weather=future_api,
                        model=model,
                        true_survival_days=true_survival_days,
                    ))

    return units


def compute_reference_anchors(
    df_daily: pd.DataFrame, healthy_cohort: List[str],
) -> Dict[str, List[float]]:
    """
    Compute the reference anchor list per cell using the current SMOOTHING_WINDOW
    daily matrix. Used by the W sweep to guarantee that every candidate W is
    evaluated over the same set of (cell, anchor) pairs.
    """
    anchors_by_cell: Dict[str, List[float]] = {}
    for cell in healthy_cohort:
        cell_data = df_daily[df_daily["cell_name"] == cell].sort_values("Exposure_Days")
        if cell_data.empty:
            continue
        max_days = cell_data["Exposure_Days"].max()
        anchors = [
            float(d) for d in range(int(BURN_IN_DAYS), int(max_days) + 1, ANCHOR_SPACING)
        ]
        if float(int(max_days)) not in anchors:
            anchors.append(float(int(max_days)))
        anchors_by_cell[cell] = anchors
    return anchors_by_cell


# ==============================================================================
# Fast LOOCV re-evaluation given pre-computed units
# ==============================================================================
def evaluate_coefficients(
    units: List[SimulationUnit],
    phi_0: float, phi_1: float, phi_2: float, lambda_w: float,
    mu: float, eps: float,
    k_blend: float = K_BLEND_DEFAULT,
    t_ref: float = T_REF_DEFAULT,
) -> Dict[str, float]:
    """
    Fast re-evaluation: applies the soft-countdown sequentially per
    (cell, mode) group and aggregates the LOOCV MAE for both modes.
    """
    grouped: Dict[tuple, List[SimulationUnit]] = {}
    for u in units:
        grouped.setdefault((u.cell, u.mode), []).append(u)

    records = []
    for (cell, mode), cell_units in grouped.items():
        cell_units.sort(key=lambda u: u.anchor_day)
        prev_anchor: Optional[float] = None
        prev_rul: Optional[float] = None

        for u in cell_units:
            rul_raw = simulate_rul_kinematics(
                u.current_damage,
                list(u.rolling_irr),
                list(u.rolling_temp),
                u.future_weather,
                u.model,
                phi_0=phi_0, phi_1=phi_1, phi_2=phi_2, lambda_w=lambda_w,
            )
            if prev_rul is not None and prev_anchor is not None:
                elapsed = u.anchor_day - prev_anchor
                expected = max(0.0, prev_rul - elapsed)
                rul_val = mu * rul_raw + (1.0 - mu) * expected
                rul_val = min(rul_val, prev_rul + eps)
            else:
                rul_val = rul_raw

            # Strategy B: blend with temporal baseline
            if k_blend > 0.0:
                rul_temporal = max(0.0, t_ref - u.anchor_day)
                rul_val = (1.0 - k_blend) * rul_val + k_blend * rul_temporal

            prev_anchor, prev_rul = u.anchor_day, rul_val
            records.append({
                "cell_name": cell,
                "Anchor_Day": u.anchor_day,
                "True_Survival_Days": u.true_survival_days,
                "RUL_Pred": rul_val,
                "Type": mode,
            })

    df = pd.DataFrame(records)
    df["RUL_Real"] = df["True_Survival_Days"] - df["Anchor_Day"]
    valid = df[df["RUL_Real"] > 0]

    def _mae(mode_label: str) -> float:
        sub = valid[valid["Type"] == mode_label]
        if sub.empty:
            return float("nan")
        return float(np.mean(np.abs(sub["RUL_Pred"] - sub["RUL_Real"])))

    mae_s = _mae("Sensor")
    mae_a = _mae("API")
    # Combined objective: 50/50 weighting so the optimizer cannot
    # sacrifice one regime (API/production) to improve the other (Sensor/LOOCV).
    # If either mode is missing (e.g. API unreachable), the combined metric is
    # undefined and we surface it as +inf so every such combination ranks last
    # and the guard in persist_calibrated_coefficients refuses to write a JSON.
    if np.isnan(mae_s) or np.isnan(mae_a):
        mae_c = float("inf")
    else:
        mae_c = 0.5 * mae_s + 0.5 * mae_a
    return {
        "mae_sensor": mae_s,
        "mae_api": mae_a,
        "mae_combined": mae_c,
        "n": int(len(valid)),
    }


# ==============================================================================
# Grids
# ==============================================================================
# Phase 1: 1D sensitivity analysis. Each coefficient is swept in isolation,
# with the others fixed at their production defaults.
GRID_1D = {
    "phi_0":    [0.0005, 0.0010, 0.0015, 0.0025, 0.0050, 0.0100],
    "phi_1":    [0.0010, 0.0025, 0.0050, 0.0100, 0.0200],
    "phi_2":    [0.5, 1.0, 1.5, 2.0, 3.0, 5.0],
    "lambda_w": [0.5, 0.8, 1.0, 1.5, 2.0, 3.0],
    "mu":       [0.2, 0.3, 0.5, 0.7, 0.9],
    "eps":      [0.0, 0.5, 1.0, 2.0, 5.0],
    "k_blend":  [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    "t_ref":    [45.0, 50.0, 54.0, 58.0, 65.0],
}

# Phase 2: focused N-D grid search. phi_1 and phi_2 are now also calibrated
# (previously held at their design values). See the ablation study in
# src/audit_v_floor.py for the empirical justification of the floor.
# NOTE: This grid is tuned for W=2 (the deployment-aligned smoothing window)
# and for the COMBINED 50/50 objective (sensor + api). The optimum region
# for the combined metric sits around:
#   phi_0 ~ 0.0025, phi_1 ~ 0.0025, phi_2 ~ 1.5, lambda_w ~ 0.5,
#   mu ~ 0.2, k_blend ~ 0.2, t_ref ~ 54
GRID_ND = {
    "phi_0":    [0.0010, 0.0015, 0.0025],
    "phi_1":    [0.0010, 0.0025, 0.0050],
    "phi_2":    [1.0,    1.5,    2.0],
    "lambda_w": [0.5, 0.8, 1.0],
    "mu":       [0.20, 0.30, 0.50],
    "eps":      [0.0, 0.5, 1.0],
    "k_blend":  [0.0, 0.1, 0.2],
    "t_ref":    [50.0, 54.0, 58.0],
}

# Dedicated sweep: the smoothing window is NOT a kinematic coefficient.
# It requires rebuilding the daily matrix and retraining the XGBoost, so it
# gets its own dedicated pass with anchors frozen across W.
SMOOTHING_WINDOW_GRID = [1, 2, 3, 5, 7, 10, 14]


# ==============================================================================
# Phase 1 — 1D sensitivity analysis
# ==============================================================================
def run_phase_1_sensitivity(units: List[SimulationUnit]) -> pd.DataFrame:
    print("\n" + "=" * 90)
    print(" PHASE 1: 1D SENSITIVITY ANALYSIS")
    print("=" * 90)
    print(
        f"  Baseline defaults: phi_0={PHI_0_DEFAULT}, phi_1={PHI_1_DEFAULT}, "
        f"phi_2={PHI_2_DEFAULT}, lambda_w={LAMBDA_W_DEFAULT}, "
        f"mu={MU_DEFAULT}, eps={EPS_DEFAULT}"
    )

    rows = []
    for coeff_name, values in GRID_1D.items():
        print(f"\n  --- Sweeping {coeff_name} ---")
        for v in values:
            kwargs = dict(
                phi_0=PHI_0_DEFAULT, phi_1=PHI_1_DEFAULT,
                phi_2=PHI_2_DEFAULT, lambda_w=LAMBDA_W_DEFAULT,
                mu=MU_DEFAULT, eps=EPS_DEFAULT,
                k_blend=K_BLEND_DEFAULT, t_ref=T_REF_DEFAULT,
            )
            kwargs[coeff_name] = v
            res = evaluate_coefficients(units, **kwargs)
            rows.append({"coefficient": coeff_name, "value": v, **res})
            print(
                f"    {coeff_name:<10s} = {v:<10.4f} | "
                f"MAE_sensor = {res['mae_sensor']:6.3f} | "
                f"MAE_api = {res['mae_api']:6.3f}"
            )

    return pd.DataFrame(rows)


# ==============================================================================
# Phase 2 — Focused N-D grid search
# ==============================================================================
def run_phase_2_grid(units: List[SimulationUnit]) -> pd.DataFrame:
    print("\n" + "=" * 90)
    print(" PHASE 2: FOCUSED N-D GRID SEARCH")
    print("=" * 90)

    keys = list(GRID_ND.keys())
    combos = list(itertools.product(*[GRID_ND[k] for k in keys]))
    print(f"  Evaluating {len(combos)} combinations on {len(units)} simulation units...")

    rows = []
    best_so_far = float("inf")
    for i, combo in enumerate(combos, 1):
        kwargs = dict(
            phi_0=PHI_0_DEFAULT, phi_1=PHI_1_DEFAULT,
            phi_2=PHI_2_DEFAULT, lambda_w=LAMBDA_W_DEFAULT,
            mu=MU_DEFAULT, eps=EPS_DEFAULT,
            k_blend=K_BLEND_DEFAULT, t_ref=T_REF_DEFAULT,
        )
        for k, v in zip(keys, combo):
            kwargs[k] = v
        res = evaluate_coefficients(units, **kwargs)
        rows.append({**dict(zip(keys, combo)), **res})
        if res["mae_combined"] < best_so_far:
            best_so_far = res["mae_combined"]
        if i % 10 == 0 or i == len(combos):
            print(
                f"  [{i:>4d}/{len(combos)}] best so far: "
                f"MAE_combined = {best_so_far:.3f}"
            )

    df = pd.DataFrame(rows).sort_values("mae_combined").reset_index(drop=True)
    print("\n  Top 10 combinations by COMBINED MAE (50/50 sensor/api):")
    print(df.head(10).to_string(index=False))
    return df

# ==============================================================================
# Phase 2 — Optuna TPE search (drop-in replacement for run_phase_2_grid)
# ==============================================================================
def run_phase_2_search(
    units: List[SimulationUnit],
    n_trials: int = 300,
    seed: int = 42,
    n_startup_trials: int = 50,
) -> pd.DataFrame:
    """
    TPE-based search over the discrete GRID_ND. Same objective, same output
    schema and same guardrails as `run_phase_2_grid`, but samples only a
    fraction of the search space.

    Why not the exhaustive grid: with phi_1 and phi_2 now calibrated, GRID_ND
    has 3^8 = 6561 combinations at ~4 s/combo ≈ 7.3 h. The ablation study
    (src/audit_v_floor.py) showed the MAE surface is very flat around the
    optimum, so a plateau-wide search is sufficient and TPE converges in
    100-200 trials. `run_phase_2_grid` is kept as a reference/fallback.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    full_grid_size = 1
    for v in GRID_ND.values():
        full_grid_size *= len(v)

    print("\n" + "=" * 90)
    print(f" PHASE 2: TPE SEARCH (Optuna) — {n_trials} trials, "
          f"{n_startup_trials} startup")
    print("=" * 90)
    print(f"  Full grid cardinality : {full_grid_size} combinations")
    print(f"  Sampling ratio        : {100 * n_trials / full_grid_size:.2f}% of the grid")

    def objective(trial: "optuna.Trial") -> float:
        kwargs = dict(
            phi_0=PHI_0_DEFAULT, phi_1=PHI_1_DEFAULT,
            phi_2=PHI_2_DEFAULT, lambda_w=LAMBDA_W_DEFAULT,
            mu=MU_DEFAULT, eps=EPS_DEFAULT,
            k_blend=K_BLEND_DEFAULT, t_ref=T_REF_DEFAULT,
        )
        for k in GRID_ND:
            kwargs[k] = trial.suggest_categorical(k, GRID_ND[k])
        res = evaluate_coefficients(units, **kwargs)
        trial.set_user_attr("mae_sensor", res["mae_sensor"])
        trial.set_user_attr("mae_api", res["mae_api"])
        trial.set_user_attr("n", res["n"])
        # Non-finite (API unreachable) → +inf so Optuna rejects the trial
        return res["mae_combined"] if np.isfinite(res["mae_combined"]) else float("inf")

    sampler = optuna.samplers.TPESampler(
        seed=seed,
        n_startup_trials=n_startup_trials,
        multivariate=True,
    )
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    rows = []
    for t in study.trials:
        if t.value is None or not np.isfinite(t.value):
            continue
        rows.append({
            **t.params,
            "mae_combined": t.value,
            "mae_sensor":   t.user_attrs.get("mae_sensor", float("nan")),
            "mae_api":      t.user_attrs.get("mae_api", float("nan")),
            "n":            t.user_attrs.get("n", 0),
        })

    df = pd.DataFrame(rows).sort_values("mae_combined").reset_index(drop=True)

    print(f"\n  Best trial        : #{study.best_trial.number}")
    print(f"  Best MAE_combined : {study.best_value:.4f}")
    print(f"  Best params       : {study.best_params}")
    print("\n  Top 10 combinations by COMBINED MAE (50/50 sensor/api):")
    print(df.head(10).to_string(index=False))
    return df

# ==============================================================================
# Blind model pre-training
# ==============================================================================
def pretrain_blind_models(
    df_daily: pd.DataFrame, healthy_cohort: List[str]
) -> Dict[str, xgb.XGBRegressor]:
    print("\n" + "=" * 90)
    print(" PRE-TRAINING BLIND XGBOOST MODELS (ONE PER LOOCV FOLD)")
    print("=" * 90)
    models: Dict[str, xgb.XGBRegressor] = {}
    for cell in healthy_cohort:
        train_cells = [c for c in healthy_cohort if c != cell]
        df_train = df_daily[df_daily["cell_name"].isin(train_cells)]
        models[cell] = train_rul_engine(df_train)
        print(f"  [{cell}] trained on {len(df_train)} rows from {len(train_cells)} cells")
    return models


# ==============================================================================
# Mini grid for the local calibration inside the smoothing-window sweep.
# ==============================================================================
# This is a *small* subset of GRID_ND. It is run once per candidate W so that
# each W is compared under its own best-case coefficients, not under the
# (arbitrary) hardcoded baseline. Without this, the sweep picks W by sensor MAE
# alone and collapses to W=1, which destroys the API regime.
SWEEP_LOCAL_GRID = {
    "phi_0":    [0.0010, 0.0015, 0.0025],   # spans baseline .. win-win corner
    "lambda_w": [0.5, 0.8, 1.0],            # includes the baseline
    "mu":       [0.20, 0.35, 0.50],         # spans calibrated .. baseline
    "k_blend":  [0.0, 0.1, 0.2],            # includes the win-win 0.1
}
# 3 * 3 * 3 * 3 = 81 combos per W. eps and t_ref are held at their defaults
# (they have negligible isolated effect) to keep the sweep tractable.


def _sweep_local_calibrate(units_w: List[SimulationUnit]) -> Dict[str, float]:
    """
    Run a small local grid search on a given W's simulation units.

    Returns a dict with the best combined MAE found for this W and the
    coefficient combination that achieved it. Used by `sweep_smoothing_window`
    to score each candidate W under its own best-case calibration, so the
    sweep reflects what each W can *achieve*, not what the hardcoded baseline
    does at that W.
    """
    keys = list(SWEEP_LOCAL_GRID.keys())
    combos = list(itertools.product(*[SWEEP_LOCAL_GRID[k] for k in keys]))

    best: Optional[Dict[str, float]] = None
    for combo in combos:
        kwargs = dict(
            phi_0=PHI_0_DEFAULT, phi_1=PHI_1_DEFAULT,
            phi_2=PHI_2_DEFAULT, lambda_w=LAMBDA_W_DEFAULT,
            mu=MU_DEFAULT, eps=EPS_DEFAULT,
            k_blend=K_BLEND_DEFAULT, t_ref=T_REF_DEFAULT,
        )
        for k, v in zip(keys, combo):
            kwargs[k] = v
        res = evaluate_coefficients(units_w, **kwargs)
        if best is None or res["mae_combined"] < best["mae_combined"]:
            best = {**res, **dict(zip(keys, combo))}

    assert best is not None, "SWEEP_LOCAL_GRID must produce at least one combination"
    return best


# ==============================================================================
# Dedicated sweep — smoothing window (rebuilds the full pipeline per W)
# ==============================================================================
def sweep_smoothing_window(
    df_twin: pd.DataFrame,
    healthy_cohort: List[str],
    t80_metrics: pd.DataFrame,
    df_api_raw: pd.DataFrame,
    window_grid: list = SMOOTHING_WINDOW_GRID,
) -> pd.DataFrame:
    """
    For each candidate smoothing window W, rebuild the entire RUL pipeline
    (daily matrix -> blind models -> simulation units) and evaluate the
    **best achievable** MAE for that W, using a small local calibration.

    Why local calibration: the kinematic coefficients interact with W. The
    optimal coefficients for W=1 differ from those for W=7. Comparing W's
    under the hardcoded baseline (as the previous version did) tells us
    which W works best under *that specific* coefficient set, not which W
    works best in general. Running a small local grid per W removes that
    confound.

    Anchors are frozen across W: the reference anchor set is computed once
    from the canonical W=7 daily matrix, and every candidate W is evaluated
    over exactly those (cell, anchor) pairs. This eliminates the survival
    bias of the earlier sweep, where W=1 produced 32 anchors vs 40 for W>=5.

    Objective: the sweep is scored by MAE_combined (50/50 sensor/API), not
    by sensor alone. Scoring by sensor would collapse to W=1, which hurts
    the API regime that the production live forecast actually uses.
    """
    print("\n" + "=" * 90)
    print(" DEDICATED SWEEP: SMOOTHING WINDOW (rebuilds the whole pipeline per W)")
    print("=" * 90)
    print(f"  Local calibration per W: {len(list(itertools.product(*SWEEP_LOCAL_GRID.values())))} combos")
    print(f"  Objective: MAE_combined = 0.5 * MAE_sensor + 0.5 * MAE_api")

    # Fix the anchor set once, using the canonical W=7 daily matrix.
    df_daily_ref = build_rul_matrix(
        df_twin, healthy_cohort, smoothing_window=SMOOTHING_WINDOW,
    )
    anchors_by_cell = compute_reference_anchors(df_daily_ref, healthy_cohort)
    print("  Reference anchors (fixed across W):")
    for cell, anchors in anchors_by_cell.items():
        print(f"    [{cell}] {len(anchors)} anchors")

    rows = []
    for w in window_grid:
        # ----------------------------------------------------------------
        # Monkey-patch: `precompute_units` and `simulate_rul_kinematics`
        # read the module-level SMOOTHING_WINDOW constant (fixed at 1 in the
        # production module). During the sweep we need them to see the
        # candidate `w` so that the rolling features are trimmed to `w`
        # observations, not to the production default. Without this patch,
        # every candidate W>1 would be evaluated with mismatched rolling
        # features (trained on W-day rolling, predicted with 1-day rolling),
        # systematically underestimating its performance.
        _original_sw = getattr(rul, "SMOOTHING_WINDOW")
        setattr(rul, "SMOOTHING_WINDOW", w)
        try:
            df_daily_w = build_rul_matrix(df_twin, healthy_cohort, smoothing_window=w)
            blind_models_w = pretrain_blind_models(df_daily_w, healthy_cohort)
            units_w = precompute_units(
                df_daily_w, df_api_raw, t80_metrics, healthy_cohort, blind_models_w,
                anchors_per_cell=anchors_by_cell,
            )

            best_w = _sweep_local_calibrate(units_w)
        finally:
            setattr(rul, "SMOOTHING_WINDOW", _original_sw)

        rows.append({
            "smoothing_window": w,
            "mae_sensor":   best_w["mae_sensor"],
            "mae_api":      best_w["mae_api"],
            "mae_combined": best_w["mae_combined"],
            "phi_0":        best_w["phi_0"],
            "lambda_w":     best_w["lambda_w"],
            "mu":           best_w["mu"],
            "k_blend":      best_w["k_blend"],
            "n":            best_w["n"],
        })
        print(
            f"  W = {w:>2d} | MAE_sensor = {best_w['mae_sensor']:6.3f} d | "
            f"MAE_api = {best_w['mae_api']:6.3f} d | "
            f"MAE_combined = {best_w['mae_combined']:6.3f} d | N = {best_w['n']} | "
            f"best: phi_0={best_w['phi_0']:.4f}, lambda_w={best_w['lambda_w']:.2f}, "
            f"mu={best_w['mu']:.2f}, k_blend={best_w['k_blend']:.2f}"
        )

    df = pd.DataFrame(rows).sort_values("mae_combined").reset_index(drop=True)
    best_w_row = df.iloc[0]
    print("\n" + "-" * 90)
    print(f"  Best W by Combined MAE (50/50): {int(best_w_row['smoothing_window'])}")
    print(f"  Expected MAE at that W: sensor={best_w_row['mae_sensor']:.3f} d, "
          f"api={best_w_row['mae_api']:.3f} d, combined={best_w_row['mae_combined']:.3f} d")
    print("-" * 90)
    print("=" * 90)
    return df


# ==============================================================================
# Final summary
# ==============================================================================
def print_final_summary(best: pd.Series, baseline: Dict[str, float]) -> None:
    print("\n" + "=" * 90)
    print(" BEST COMBINATION FOUND (by COMBINED MAE, 50/50 sensor/api)")
    print("=" * 90)

    print("  Swept coefficients:")
    for k in GRID_ND.keys():
        if k in best.index:
            base_val = BASELINE_REGISTRY[k]
            print(f"    {k:<10s} = {float(best[k]):<10.4f}   (baseline {base_val})")

    print("\n  Held coefficients (not swept in Phase 2):")
    held = [k for k in BASELINE_REGISTRY if k not in GRID_ND]
    if held:
        for k in held:
            print(f"    {k:<10s} = {BASELINE_REGISTRY[k]:<10.4f}   (default)")
    else:
        print("    (none — all coefficients swept in Phase 2)")

    base_combined = 0.5 * baseline["mae_sensor"] + 0.5 * baseline["mae_api"]
    print("\n  Metrics:")
    print(f"    MAE_sensor   = {best['mae_sensor']:.3f}  (baseline {baseline['mae_sensor']:.3f})")
    print(f"    MAE_api      = {best['mae_api']:.3f}  (baseline {baseline['mae_api']:.3f})")
    if "mae_combined" in best.index:
        print(f"    MAE_combined = {best['mae_combined']:.3f}  (baseline {base_combined:.3f})")

    improvement = base_combined - float(best["mae_combined"])
    pct = 100.0 * improvement / base_combined
    print(f"\n  Improvement (Combined 50/50) = {improvement:+.3f} days ({pct:+.1f}%)")
    print("=" * 90)


# ==============================================================================
# Persist calibrated coefficients
# ==============================================================================
def persist_calibrated_coefficients(
    best_row: pd.Series,
    baseline: Dict[str, float],
    grid_size: int,
    search_method: str = "TPE_optuna",
) -> None:
    """
    Write the winning coefficients to a JSON file consumed by the 08 module.
    Includes a metadata block for traceability.

    Guard: refuses to persist if the best combination does not beat the
    hardcoded baseline on the COMBINED objective (50/50 sensor/api). This
    prevents the optimizer from silently degrading production with a JSON
    that improves one regime at the expense of the other.
    """
    best_combined = float(best_row["mae_combined"])
    base_combined = 0.5 * float(baseline["mae_sensor"]) + 0.5 * float(baseline["mae_api"])

    # Hard guard 1: if the combined metric is not a finite number (e.g. the
    # API was unreachable and mae_api came back as NaN), refuse to persist.
    # This protects against transient network failures silently poisoning
    # production with an unvalidated JSON.
    if not np.isfinite(best_combined) or not np.isfinite(base_combined):
        print(
            f"\n  [GUARD] Combined MAE is not finite "
            f"(best={best_combined}, baseline={base_combined}). "
            f"Most likely the API fetch failed and there are no API units. "
            f"Coefficients NOT persisted."
        )
        print("  The 08 module will keep using the hardcoded defaults.")
        return

    # Hard guard 2: refuse to persist a combination that does not strictly
    # improve on the hardcoded baseline under the combined objective.
    if best_combined >= base_combined:
        print(
            f"\n  [GUARD] Best combined MAE ({best_combined:.3f}) does NOT beat "
            f"baseline combined ({base_combined:.3f}). "
            f"Coefficients NOT persisted."
        )
        print("  The 08 module will keep using the hardcoded defaults.")
        return

    import json
    from datetime import datetime

    # Derived search-space diagnostics. Computed from the current GRID_ND
    # definition and the number of finite trials actually evaluated, so they
    # stay consistent if the grid or the trial budget is ever changed.
    grid_cardinality = 1
    for v in GRID_ND.values():
        grid_cardinality *= len(v)
    search_ratio_pct = (
        100.0 * grid_size / grid_cardinality if grid_cardinality else 0.0
    )

    payload = {
        "smoothing_window": int(SMOOTHING_WINDOW),
        "phi_0":    float(best_row["phi_0"]) if "phi_0" in best_row.index else PHI_0_DEFAULT,
        "phi_1":    float(best_row["phi_1"]) if "phi_1" in best_row.index else PHI_1_DEFAULT,
        "phi_2":    float(best_row["phi_2"]) if "phi_2" in best_row.index else PHI_2_DEFAULT,
        "lambda_w": float(best_row["lambda_w"]) if "lambda_w" in best_row.index else LAMBDA_W_DEFAULT,
        "mu":       float(best_row["mu"]) if "mu" in best_row.index else MU_DEFAULT,
        "eps":      float(best_row["eps"]) if "eps" in best_row.index else EPS_DEFAULT,
        "k_blend":  float(best_row["k_blend"]) if "k_blend" in best_row.index else K_BLEND_DEFAULT,
        "t_ref":    float(best_row["t_ref"]) if "t_ref" in best_row.index else T_REF_DEFAULT,
        "metadata": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "generated_by": "src/rul_calibration_optimizer.py",
            "grid_size": grid_size,
            "grid_cardinality": grid_cardinality,
            "search_method": search_method,
            "search_ratio_pct": round(search_ratio_pct, 2),
            "mae_sensor": float(best_row["mae_sensor"]),
            "mae_api": float(best_row["mae_api"]),
            "mae_combined": float(best_row["mae_combined"]),
            "baseline_mae_sensor": float(baseline["mae_sensor"]),
            "baseline_mae_api": float(baseline["mae_api"]),
            "baseline_mae_combined": base_combined,
            "objective": "mae_combined_50_50",
            "n_observations": int(best_row["n"]),
            "smoothing_window": int(SMOOTHING_WINDOW),
        },
    }

    CALIBRATED_COEFFS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CALIBRATED_COEFFS_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\n  Calibrated coefficients persisted -> {CALIBRATED_COEFFS_PATH}")
    print("  The 08 module will automatically use these on its next run.")


# ==============================================================================
# Entry point
# ==============================================================================
def _run() -> None:
    # ---- Data ----
    logger.info("Loading inputs...")
    df_twin = pd.read_parquet(FILE_HEALTHY_COHORT)
    t80_metrics = pd.read_parquet(FILE_T80_TRUTH)
    healthy_cohort = joblib.load(FILE_SCREENING_ARTIFACTS)["healthy_cohort"]

    if "PCE_initial" not in df_twin.columns:
        df_twin = df_twin.merge(
            t80_metrics[["PCE_initial"]],
            left_on="cell_name", right_index=True, how="left",
        )

    df_daily = build_rul_matrix(df_twin, healthy_cohort)
    df_api_raw = fetch_api_history(df_daily)

    # ---- Dedicated sweep of the smoothing window (anchors frozen across W) ----
    df_window = sweep_smoothing_window(
        df_twin, healthy_cohort, t80_metrics, df_api_raw,
    )
    WINDOW_SWEEP_PATH = DIAGNOSTICS_RUL_DIR / "rul_smoothing_window_sweep.parquet"
    df_window.to_parquet(WINDOW_SWEEP_PATH, index=False)
    print(f"\n  Smoothing-window sweep saved -> {WINDOW_SWEEP_PATH}")

    # ---- Pre-training ----
    blind_models = pretrain_blind_models(df_daily, healthy_cohort)

    # ---- Pre-compute simulation units ----
    print("\n" + "=" * 90)
    print(" PRE-COMPUTING SIMULATION UNITS")
    print("=" * 90)
    units = precompute_units(df_daily, df_api_raw, t80_metrics, healthy_cohort, blind_models)
    sensor_units = [u for u in units if u.mode == "Sensor"]
    api_units = [u for u in units if u.mode == "API"]
    print(f"  Sensor units: {len(sensor_units)}")
    print(f"  API units   : {len(api_units)}")
    print(f"  Total       : {len(units)}")

    if not api_units:
        print(
            "\n  [ABORT] No API simulation units were produced. "
            "The Open-Meteo history fetch must have failed. "
            "The combined objective cannot be evaluated without API data. "
            "Retry when the API is reachable."
        )
        print("  No calibration performed; the 08 module will keep using hardcoded defaults.")
        return

    # ---- Baseline ----
    baseline = evaluate_coefficients(
        units,
        phi_0=PHI_0_DEFAULT, phi_1=PHI_1_DEFAULT,
        phi_2=PHI_2_DEFAULT, lambda_w=LAMBDA_W_DEFAULT,
        mu=MU_DEFAULT, eps=EPS_DEFAULT,
    )
    print("\n" + "=" * 90)
    print(" BASELINE (current production defaults)")
    print("=" * 90)
    print(f"  MAE_sensor = {baseline['mae_sensor']:.3f} days")
    print(f"  MAE_api    = {baseline['mae_api']:.3f} days")
    print(f"  N          = {baseline['n']}")

    # ---- Phase 1 ----
    df_sens = run_phase_1_sensitivity(units)
    SENSITIVITY_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_sens.to_parquet(SENSITIVITY_PATH, index=False)
    print(f"\n  Sensitivity table saved -> {SENSITIVITY_PATH}")

    # ---- Phase 2 (Optuna TPE) ----
    df_grid = run_phase_2_search(units, n_trials=300, seed=42)
    df_grid.to_parquet(SEARCH_PATH, index=False)
    print(f"\n  TPE search table saved -> {SEARCH_PATH}")

    # ---- Final summary ----
    print_final_summary(df_grid.iloc[0], baseline)

    # ---- Persist calibrated coefficients ----
    persist_calibrated_coefficients(
        best_row=df_grid.iloc[0],
        baseline=baseline,
        grid_size=len(df_grid),
    )


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
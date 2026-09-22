"""
Module: src/rul_audit_v_floor.py
Description: Ablation study of the asymptotic structural floor in the hybrid
             kinematics engine (module 08).

             v_floor(rho) = phi_0 + phi_1 * exp(phi_2 * rho)

             Compares the calibrated production configuration against
             configurations where the floor is disabled or partially
             disabled, to quantify its empirical contribution to the combined
             MAE (50/50 sensor/API). All other coefficients (lambda_w, mu,
             eps, k_blend, t_ref) are held at their calibrated values so the
             comparison isolates the floor's contribution.

             Outputs:
               - outputs/diagnostics/audit_v_floor.parquet
               - outputs/diagnostics/audit_v_floor.txt
"""

from __future__ import annotations

import importlib
import json
import logging
import sys
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd

from src.config import (
    FILE_HEALTHY_COHORT,
    FILE_T80_TRUTH,
    FILE_SCREENING_ARTIFACTS,
    FILE_RUL_COEFFS_CALIBRATED,
    DIAGNOSTICS_RUL_DIR,
)
from src.rul_calibration_optimizer import (
    precompute_units,
    pretrain_blind_models,
    evaluate_coefficients,
)

rul = importlib.import_module("src.08_mppt_rul_forecasting")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("Audit-vFloor")

REPORT_PATH = DIAGNOSTICS_RUL_DIR / "audit_v_floor.txt"
TABLE_PATH  = DIAGNOSTICS_RUL_DIR / "audit_v_floor.parquet"


class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)
        return len(data)

    def flush(self):
        for s in self._streams:
            s.flush()


# ------------------------------------------------------------------------------
# Loader of the production coefficients (JSON if exists, else hardcoded)
# ------------------------------------------------------------------------------
def load_calibrated() -> Dict[str, float]:
    path = Path(FILE_RUL_COEFFS_CALIBRATED)
    if not path.exists():
        logger.warning("No calibrated JSON found; falling back to hardcoded.")
        return {
            "phi_0": rul.PHI_0_HARDCODED,
            "phi_1": rul.PHI_1_HARDCODED,
            "phi_2": rul.PHI_2_HARDCODED,
            "lambda_w": rul.LAMBDA_W_HARDCODED,
            "mu": rul.MU_HARDCODED,
            "eps": rul.EPS_HARDCODED,
            "k_blend": rul.K_BLEND_HARDCODED,
            "t_ref": rul.T_REF_HARDCODED,
        }
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return {
        "phi_0":    float(data["phi_0"]),
        "phi_1":    float(data["phi_1"]),
        "phi_2":    float(data["phi_2"]),
        "lambda_w": float(data["lambda_w"]),
        "mu":       float(data["mu"]),
        "eps":      float(data["eps"]),
        "k_blend":  float(data.get("k_blend", rul.K_BLEND_HARDCODED)),
        "t_ref":    float(data.get("t_ref", rul.T_REF_HARDCODED)),
    }


# ------------------------------------------------------------------------------
# Ablation configurations
# ------------------------------------------------------------------------------
def build_configs(base: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    """
    Each entry is a full coefficient dict. Only phi_0/phi_1/phi_2 (and, for
    one config, lambda_w) are mutated; everything else stays calibrated so
    the delta isolates the floor.
    """
    def _with(**overrides):
        c = dict(base); c.update(overrides); return c

    return {
        # Reference
        "full_calibrated":       dict(base),

        # Full removal of the floor: v_floor ≡ 0
        "no_floor":              _with(phi_0=0.0, phi_1=0.0, phi_2=0.0),

        # Constant floor only (no asymptotic acceleration)
        "constant_only":         _with(phi_1=0.0),

        # Exponential floor only (no constant term)
        "exponential_only":      _with(phi_0=0.0),

        # No rho-dependence in the exponential (phi_2=0 ⇒ exp(0)=1)
        "no_asymptotic_accel":   _with(phi_2=0.0),

        # Weight always saturated to 1 (lambda_w=0 ⇒ w(rho)=1 ∀ρ)
        "weight_always_one":     _with(lambda_w=0.0),

        # Pre-calibration hardcoded floor (isolates the calibration step)
        "hardcoded_floor":       _with(
            phi_0=rul.PHI_0_HARDCODED,
            phi_1=rul.PHI_1_HARDCODED,
            phi_2=rul.PHI_2_HARDCODED,
        ),
    }


# ------------------------------------------------------------------------------
# Driver
# ------------------------------------------------------------------------------
def _run() -> None:
    logger.info("Loading inputs...")
    df_twin = pd.read_parquet(FILE_HEALTHY_COHORT)
    t80_metrics = pd.read_parquet(FILE_T80_TRUTH)
    healthy_cohort = joblib.load(FILE_SCREENING_ARTIFACTS)["healthy_cohort"]

    if "PCE_initial" not in df_twin.columns:
        df_twin = df_twin.merge(
            t80_metrics[["PCE_initial"]],
            left_on="cell_name", right_index=True, how="left",
        )

    df_daily = rul.build_rul_matrix(df_twin, healthy_cohort)
    df_api_raw = rul.fetch_api_history(df_daily)

    if df_api_raw.empty:
        logger.warning(
            "API history unavailable — ablation will be Sensor-only and the "
            "combined 50/50 objective will be NaN. Retry when reachable."
        )

    # Pre-train blind models + pre-compute units ONCE. The ablation only
    # re-runs the closed-form simulation loop, so it takes seconds.
    blind_models = pretrain_blind_models(df_daily, healthy_cohort)
    units = precompute_units(
        df_daily, df_api_raw, t80_metrics, healthy_cohort, blind_models,
    )
    n_sensor = sum(1 for u in units if u.mode == "Sensor")
    n_api    = sum(1 for u in units if u.mode == "API")
    print(f"\n  Simulation units: {n_sensor} sensor + {n_api} API = {len(units)} total")

    base = load_calibrated()
    configs = build_configs(base)

    print("\n" + "=" * 90)
    print(" ABLATION STUDY — STRUCTURAL FLOOR  v_floor(rho) = phi_0 + phi_1 * exp(phi_2 * rho)")
    print("=" * 90)
    print("  Reference (calibrated production coefficients):")
    for k, v in base.items():
        print(f"    {k:<10s} = {v}")

    rows = []
    print()
    for name, cfg in configs.items():
        res = evaluate_coefficients(units, **cfg)
        rows.append({
            "config":       name,
            "phi_0":        cfg["phi_0"],
            "phi_1":        cfg["phi_1"],
            "phi_2":        cfg["phi_2"],
            "lambda_w":     cfg["lambda_w"],
            "mae_sensor":   res["mae_sensor"],
            "mae_api":      res["mae_api"],
            "mae_combined": res["mae_combined"],
            "n":            res["n"],
        })
        print(
            f"  {name:<22s} | "
            f"MAE_sensor={res['mae_sensor']:7.3f} d | "
            f"MAE_api={res['mae_api']:7.3f} d | "
            f"MAE_combined={res['mae_combined']:7.3f} d"
        )

    df = pd.DataFrame(rows)
    ref = df[df["config"] == "full_calibrated"].iloc[0]
    df["delta_sensor"]   = df["mae_sensor"]   - ref["mae_sensor"]
    df["delta_api"]      = df["mae_api"]      - ref["mae_api"]
    df["delta_combined"] = df["mae_combined"] - ref["mae_combined"]

    print("\n" + "-" * 90)
    print(" DELTA vs. FULL CALIBRATED  (positive ⇒ the mutation HURTS the engine)")
    print("-" * 90)
    for _, r in df.iterrows():
        if r["config"] == "full_calibrated":
            continue
        print(
            f"  {r['config']:<22s} | "
            f"Δsensor={r['delta_sensor']:+7.3f} d | "
            f"Δapi={r['delta_api']:+7.3f} d | "
            f"Δcombined={r['delta_combined']:+7.3f} d"
        )

    # -------------------------------------------------------------------------
    # Verdict
    # -------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print(" VERDICT")
    print("=" * 90)
    nf = df[df["config"] == "no_floor"].iloc[0]
    print(f"  Full removal of the floor (phi_0=phi_1=phi_2=0):")
    print(f"    ΔMAE_sensor   = {nf['delta_sensor']:+7.3f} d")
    print(f"    ΔMAE_api      = {nf['delta_api']:+7.3f} d")
    print(f"    ΔMAE_combined = {nf['delta_combined']:+7.3f} d")

    # Use combined when available, else fall back to sensor
    metric_label = "combined"
    delta_main = nf["delta_combined"]
    if not np.isfinite(delta_main):
        metric_label = "sensor (API unreachable)"
        delta_main = nf["delta_sensor"]

    if not np.isfinite(delta_main):
        print("  => No se puede emitir veredicto: métricas no finitas.")
    elif delta_main > 0.50:
        print(f"  => El suelo estructural APORTA valor SUSTANCIAL "
              f"(Δ{metric_label} = {delta_main:+.3f} d).")
    elif delta_main > 0.10:
        print(f"  => El suelo estructural APORTA valor REAL "
              f"(Δ{metric_label} = {delta_main:+.3f} d).")
    elif delta_main > 0.0:
        print(f"  => El suelo estructural aporta una mejora MARGINAL "
              f"(Δ{metric_label} = {delta_main:+.3f} d < 0.10 d).")
    else:
        print(f"  => El suelo estructural NO aporta valor; la ecuación es un vestigio "
              f"(Δ{metric_label} = {delta_main:+.3f} d).")

    print("\n  Notas sobre los sub-experimentos:")
    for cfg_name, desc in [
        ("constant_only",      "solo término constante (φ₁=0)"),
        ("exponential_only",   "solo término exponencial (φ₀=0)"),
        ("no_asymptotic_accel","sin aceleración asintótica (φ₂=0)"),
        ("weight_always_one",  "peso del suelo saturado a 1 (λ_w=0)"),
        ("hardcoded_floor",    "suelo pre-calibración (aisla el paso de calibración)"),
    ]:
        r = df[df["config"] == cfg_name].iloc[0]
        tag = "PEOR" if r["delta_combined"] > 0 else "MEJOR"
        print(f"    {desc:<58s} Δcombined = {r['delta_combined']:+7.3f} d  [{tag}]")

    # -------------------------------------------------------------------------
    # Persist
    # -------------------------------------------------------------------------
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(TABLE_PATH, index=False)
    print(f"\n  Ablation table saved -> {TABLE_PATH}")
    print("=" * 90)


def main() -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    original_stdout = sys.stdout
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        sys.stdout = _Tee(original_stdout, f)
        try:
            _run()
        finally:
            sys.stdout = original_stdout
    logger.info(f"Report saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
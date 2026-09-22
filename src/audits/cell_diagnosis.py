"""
Module: src/audit_cell_diagnosis.py
Description: Master diagnostic script for investigating anomalous cell behavior.

Purpose
-------
When a photovoltaic cell exhibits unexpected behavior (early failure, unusual
degradation pattern, poor RUL prediction), this script produces a complete
forensic report. Set DIAG_CELL at the top and run:

    python -m src.audit_cell_diagnosis

Layers (general -> specific)
----------------------------
  1  Cohort overview          Where does the target sit in the population?
  2  Trajectory comparison    How does its behavior differ from peers?
  3  Matched-peer analysis    Target vs a filtered reference group.
  4  Target forensics         Continuity, raw telemetry, pre-failure window.
  5  Statistical diagnostics  Thermal dose, annealing, cross-correlations.
  6  Failure mode assessment  Automatic classification with reasoning.

Reference peer selection
------------------------
The reference peers used throughout the diagnosis come from the healthy
cohort declared by module 07 in the screening artifact. Using the pipeline's
own validated cohort avoids the contamination that occurs when degraded
cells (e.g. P12, M0) are included in the peer statistics: those cells have
distinct temperature profiles and PCE behavior and would skew the
comparison. If the artifact is missing or malformed, the script falls back
to a data-driven peer selection with a warning.

Outputs
-------
  outputs/figures/diagnostics/{DIAG_CELL}_*.png
  outputs/diagnostics/cell_diagnosis/{DIAG_CELL}/
      audit_cell_diagnosis_{DIAG_CELL}.parquet   (summary)
      audit_cell_diagnosis_{DIAG_CELL}.log       (full report)
      cohort_overview.parquet
      early_phase_comparison.parquet
      pre_failure_window.parquet
      thermal_dose.parquet
      annealing.parquet
      cross_correlation.parquet
      feature_correlations.parquet
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import (
    BURN_IN_DAYS,
    DAYLIGHT_IRRADIANCE_MIN_W_M2,
    DIAGNOSTICS_CELL_DIAGNOSIS_DIR,
    FIGURES_DIR,
    FILE_MERGED_FEATURES,
    FILE_SCREENING_ARTIFACTS,
    FILE_T80_TRUTH,
    T80_FRACTION,
)

# ==============================================================================
# CONFIGURATION
# ==============================================================================
DIAG_CELL: str = "M83AB302"
REFERENCE_CELLS: Optional[List[str]] = None

FAILURE_WINDOW_DAYS: float = 5.0
EARLY_PHASE_CUTOFF: float = 10.0
TEMP_THRESHOLDS: List[float] = [35.0, 40.0, 45.0, 50.0]
MIN_REFERENCE_DAYS: int = 10

# Thresholds used by the failure-mode classifier in Layer 6. Each one
# represents "target is X times the peer baseline" for the corresponding
# metric. Values above the threshold generate an evidence flag.
THRESHOLD_SUDDEN_JUMP: float = 0.20         # absolute damage jump
THRESHOLD_THERMAL_DOSE_RATIO: float = 1.30  # cumulative excess over 35 C
THRESHOLD_LOW_PCE_RATIO: float = 0.85       # early-phase mean PCE
THRESHOLD_LOW_PCE_STD_RATIO: float = 0.60   # early-phase PCE std
THRESHOLD_LOW_RECOVERY_PCT: float = 30.0    # % of days with damage recovery

T80_DAMAGE_LIMIT: float = float(1.0 - float(T80_FRACTION))
BURN_IN_DAYS_F: float = float(BURN_IN_DAYS)
T80_FRACTION_F: float = float(T80_FRACTION)

FIG_DIR: Path = FIGURES_DIR / "diagnostics"
CELL_DIAG_DIR: Path = DIAGNOSTICS_CELL_DIAGNOSIS_DIR / DIAG_CELL
OUT_PARQUET: Path = CELL_DIAG_DIR / f"audit_cell_diagnosis_{DIAG_CELL}.parquet"
OUT_LOG: Path = CELL_DIAG_DIR / f"audit_cell_diagnosis_{DIAG_CELL}.log"
DETAILED_DIR: Path = CELL_DIAG_DIR  # tables live alongside the summary

# ==============================================================================
# LOGGING
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("CellDiagnosis")


class _Tee:
    """Duplicate stdout writes to a file and the terminal."""

    def __init__(self, *streams: Any) -> None:
        self._streams: Tuple[Any, ...] = streams

    def write(self, data: str) -> int:
        for s in self._streams:
            s.write(data)
        return len(data)

    def flush(self) -> None:
        for s in self._streams:
            s.flush()


# ==============================================================================
# SAFE NUMERIC COERCION
# ==============================================================================
def _f(value: Any) -> float:
    """
    Coerce any pandas/numpy scalar (or 0-d/1-d array) to a Python float.

    Rationale: Pylance cannot infer that df.loc[row, col] returns a scalar,
    so it complains when we pass the result to float(). This helper accepts
    Any and performs the coercion robustly, returning NaN on failure.
    """
    if value is None:
        return float("nan")
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float, np.floating, np.integer)):
        return float(value)
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
        if arr.size == 0:
            return float("nan")
        return float(arr[0])
    except (TypeError, ValueError):
        return float("nan")


def _i(value: Any) -> int:
    """Coerce to int; returns 0 on failure."""
    f = _f(value)
    if np.isnan(f):
        return 0
    return int(f)


# ==============================================================================
# DATA LOADING AND PREPROCESSING
# ==============================================================================
def load_and_prepare() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load merged features + T80 truth, filter daylight, aggregate daily.

    PCE_initial is not present in the merged feature matrix produced by
    module 05; it lives in the T80 truth table indexed by cell_name
    (produced by module 06). We merge it in before aggregation so that
    the daily PCE_initial and the Instant_Loss computation below can be
    defined.
    """
    df: pd.DataFrame = pd.read_parquet(FILE_MERGED_FEATURES)
    t80: pd.DataFrame = pd.read_parquet(FILE_T80_TRUTH)

    if df.index.name == "Timestamp":
        df = df.reset_index()

    # --- Ensure PCE_initial is available before aggregation -----------
    if "PCE_initial" not in df.columns:
        if "PCE_initial" not in t80.columns:
            raise KeyError(
                "PCE_initial not found in either FILE_MERGED_FEATURES "
                "or FILE_T80_TRUTH. Check the output of module 06."
            )
        df = df.merge(
            t80[["PCE_initial"]],
            left_on="cell_name",
            right_index=True,
            how="left",
        )

    df["Datetime"] = pd.to_datetime(df["Timestamp"], utc=True)
    df["Day_Zero"] = df.groupby("cell_name")["Datetime"].transform("min")
    df["Exposure_Days"] = (
        (df["Datetime"] - df["Day_Zero"]).dt.total_seconds() / 86400.0
    )
    df = df[df["POA_Irradiance_W_m2"] > DAYLIGHT_IRRADIANCE_MIN_W_M2].copy()
    df["Date_Day"] = df["Datetime"].dt.date

    daily: pd.DataFrame = (
        df.groupby(["cell_name", "Date_Day"])
        .agg(
            PCE=("PCE", "max"),
            pFF=("pFF", "max"),
            Exposure_Days=("Exposure_Days", "max"),
            PCE_initial=("PCE_initial", "first"),
            n_points=("PCE", "count"),
            Daily_Irradiance_Dose=("POA_Irradiance_W_m2", "sum"),
            Daily_Mean_POA=("POA_Irradiance_W_m2", "mean"),
            Daily_Max_Temp_C=("ModuleTemp_C", "max"),
            Daily_Mean_Temp_C=("ModuleTemp_C", "mean"),
            Daily_Median_Humidity=("AbsoluteHumidity_g_m3", "median"),
            Daily_Mean_Delta_Temp=("Delta_Temp_C_per_h", "mean"),
            Daily_Max_Delta_Temp=("Delta_Temp_C_per_h", "max"),
            Daily_Mean_Delta_Hum=("Delta_Hum_g_m3_per_h", "mean"),
            Daily_Std_Delta_Temp=("Delta_Temp_C_per_h", "std"),
        )
        .reset_index()
        .sort_values(["cell_name", "Date_Day"])
    )
    daily["Instant_Loss"] = 1.0 - daily["PCE"] / daily["PCE_initial"]
    daily["Cumulative_Damage"] = (
        daily.groupby("cell_name")["Instant_Loss"]
        .rolling(2, min_periods=1)
        .median()
        .reset_index(level=0, drop=True)
    )
    daily["Daily_Damage_Increment"] = (
        daily.groupby("cell_name")["Cumulative_Damage"].diff().fillna(0.0)
    )
    return daily, t80


def select_reference_cells(daily: pd.DataFrame) -> List[str]:
    """
    Pick reference peers for the target cell.

    Priority order:
      1. If REFERENCE_CELLS is set explicitly at the top of the module,
         use it verbatim (manual override).
      2. Otherwise, use the healthy_cohort declared by module 07 in the
         screening artifact. This is the cohort that the pipeline itself
         has validated as healthy; using it as reference avoids the
         contamination that occurs when degraded cells (P12, M0) are
         included in the peer statistics.
      3. If the artifact is missing (pre-v2 run), fall back to "all other
         cells with >= MIN_REFERENCE_DAYS daily records", with a warning.
    """
    if REFERENCE_CELLS is not None:
        return list(REFERENCE_CELLS)

    if FILE_SCREENING_ARTIFACTS.exists():
        try:
            artifact = joblib.load(FILE_SCREENING_ARTIFACTS)
            cohort = artifact.get("healthy_cohort", [])
            peers = [str(c) for c in cohort if str(c) != DIAG_CELL]
            if peers:
                return sorted(peers)
            logger.warning(
                "Screening artifact does not declare any healthy peers; "
                "falling back to data-driven peer selection."
            )
        except Exception as exc:
            logger.warning(
                f"Failed to load screening artifact ({exc}); "
                f"falling back to data-driven peer selection."
            )
    else:
        logger.warning(
            f"Screening artifact not found at {FILE_SCREENING_ARTIFACTS}; "
            f"falling back to data-driven peer selection."
        )

    counts = daily.groupby("cell_name")["Exposure_Days"].count()
    peers: List[str] = []
    for c in counts.index:
        cs = str(c)
        if cs == DIAG_CELL:
            continue
        if _i(counts.at[c]) >= MIN_REFERENCE_DAYS:
            peers.append(cs)
    return sorted(peers)


# ==============================================================================
# LAYER 1 — COHORT OVERVIEW
# ==============================================================================
def layer1_cohort_overview(daily: pd.DataFrame, t80: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 80)
    print("# LAYER 1 — COHORT OVERVIEW")
    print(f"# Target: {DIAG_CELL}")
    print("=" * 80)

    rows: List[Dict[str, Any]] = []
    for cell in daily["cell_name"].unique():
        cell_str: str = str(cell)
        sub = daily[daily["cell_name"] == cell]
        inc = sub["Daily_Damage_Increment"]

        survival: float = float("nan")
        if cell_str in t80.index:
            survival = _f(t80.at[cell_str, "combined_survival_days"])

        rows.append({
            "cell": cell_str,
            "is_target": cell_str == DIAG_CELL,
            "n_days": int(len(sub)),
            "survival_days": survival,
            "temp_mean": _f(sub["Daily_Max_Temp_C"].mean()),
            "humidity_mean": _f(sub["Daily_Median_Humidity"].mean()),
            "irr_mean": _f(sub["Daily_Irradiance_Dose"].mean()),
            "pce_mean": _f(sub["PCE"].mean()),
            "pce_std": _f(sub["PCE"].std()),
            "damage_final": _f(sub["Cumulative_Damage"].iloc[-1]),
            "damage_rate_mean": _f(inc.mean()),
            "damage_rate_std": _f(inc.std()),
            "max_daily_jump": _f(inc.max()),
            "recovery_days_pct": _f(100.0 * (inc < -0.001).sum() / max(1, len(sub))),
        })

    table = pd.DataFrame(rows).set_index("cell").sort_values("survival_days")

    print("\n  Full cohort metrics (sorted by survival):")
    with pd.option_context("display.width", 240, "display.max_columns", 20):
        print(table.round(3).to_string())

    target_survival = _f(table.at[DIAG_CELL, "survival_days"])
    rank = int((table["survival_days"] < target_survival).sum()) + 1
    median_survival = _f(table["survival_days"].median())

    print(f"\n  Target '{DIAG_CELL}' rank by survival: {rank} of {len(table)}")
    print(f"  Survival: {target_survival:.2f} d (cohort median: {median_survival:.2f} d)")
    return table


# ==============================================================================
# LAYER 2 — TRAJECTORY COMPARISON
# ==============================================================================
def layer2_trajectory_comparison(daily: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print("# LAYER 2 — TRAJECTORY COMPARISON")
    print("=" * 80)

    # --- Figure 2a: cumulative damage ---
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for cell in daily["cell_name"].unique():
        cs = str(cell)
        sub = daily[daily["cell_name"] == cell].sort_values("Exposure_Days")
        x = np.asarray(sub["Exposure_Days"].to_numpy(), dtype=float)
        y = np.asarray(sub["Cumulative_Damage"].to_numpy(), dtype=float)
        is_target = cs == DIAG_CELL
        ax.plot(x, y,
                linewidth=2.5 if is_target else 1.2,
                alpha=1.0 if is_target else 0.55,
                label=cs,
                zorder=3 if is_target else 1)
    ax.axhline(T80_DAMAGE_LIMIT, color="black", linestyle="--",
               linewidth=1, label=f"T80 = {T80_DAMAGE_LIMIT:.2f}")
    ax.axvline(BURN_IN_DAYS_F, color="gray", linestyle=":",
               linewidth=1, label=f"Burn-in = {BURN_IN_DAYS_F:.0f} d")
    ax.set_xlabel("Exposure (days)")
    ax.set_ylabel("Cumulative damage (rolling median W=2)")
    ax.set_title(f"Layer 2a — Cumulative damage · target: {DIAG_CELL}")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{DIAG_CELL}_02a_damage_curves.png", dpi=150)
    plt.close(fig)

    # --- Figure 2b: raw PCE normalized ---
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for cell in daily["cell_name"].unique():
        cs = str(cell)
        sub = daily[daily["cell_name"] == cell].sort_values("Exposure_Days")
        pce0 = _f(sub["PCE"].iloc[0])
        if pce0 <= 0.0:
            continue
        x = np.asarray(sub["Exposure_Days"].to_numpy(), dtype=float)
        y = np.asarray(sub["PCE"].to_numpy(), dtype=float) / pce0
        is_target = cs == DIAG_CELL
        ax.plot(x, y,
                linewidth=2.5 if is_target else 1.2,
                alpha=1.0 if is_target else 0.55,
                label=cs,
                zorder=3 if is_target else 1)
    ax.axhline(T80_FRACTION_F, color="black", linestyle="--", linewidth=1,
               label=f"T80 = {T80_FRACTION_F:.2f}")
    ax.set_xlabel("Exposure (days)")
    ax.set_ylabel("PCE / PCE_initial (daily max)")
    ax.set_title(f"Layer 2b — Raw daily PCE · target: {DIAG_CELL}")
    ax.legend(loc="lower left", fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{DIAG_CELL}_02b_pce_curves.png", dpi=150)
    plt.close(fig)

    # --- Figure 2c: pFF normalized ---
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for cell in daily["cell_name"].unique():
        cs = str(cell)
        sub = daily[daily["cell_name"] == cell].sort_values("Exposure_Days")
        pff0 = _f(sub["pFF"].iloc[0])
        if pff0 <= 0.0:
            continue
        x = np.asarray(sub["Exposure_Days"].to_numpy(), dtype=float)
        y = np.asarray(sub["pFF"].to_numpy(), dtype=float) / pff0
        is_target = cs == DIAG_CELL
        ax.plot(x, y,
                linewidth=2.5 if is_target else 1.2,
                alpha=1.0 if is_target else 0.55,
                label=cs,
                zorder=3 if is_target else 1)
    ax.axhline(T80_FRACTION_F, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Exposure (days)")
    ax.set_ylabel("pFF / pFF_initial (daily max)")
    ax.set_title(f"Layer 2c — Normalized pFF · target: {DIAG_CELL}")
    ax.legend(loc="lower left", fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{DIAG_CELL}_02c_pff_curves.png", dpi=150)
    plt.close(fig)

    print("  Trajectory figures written (02a damage, 02b PCE, 02c pFF).")


# ==============================================================================
# LAYER 3 — MATCHED-PEER ANALYSIS
# ==============================================================================
def layer3_matched_peers(
    daily: pd.DataFrame, reference_cells: List[str]
) -> pd.DataFrame:
    print("\n" + "=" * 80)
    print("# LAYER 3 — MATCHED-PEER ANALYSIS")
    print(f"# Reference peers ({len(reference_cells)}): {reference_cells}")
    print("=" * 80)

    target = daily[daily["cell_name"] == DIAG_CELL].sort_values("Exposure_Days")
    peers = daily[daily["cell_name"].isin(reference_cells)].sort_values("Exposure_Days")

    target_early = target[target["Exposure_Days"] <= EARLY_PHASE_CUTOFF]
    peers_early = peers[peers["Exposure_Days"] <= EARLY_PHASE_CUTOFF]

    def _stats(sub: pd.DataFrame) -> Dict[str, float]:
        inc = sub["Daily_Damage_Increment"]
        return {
            "n_days": float(len(sub)),
            "pce_mean": _f(sub["PCE"].mean()),
            "pce_std": _f(sub["PCE"].std()),
            "damage_mean": _f(sub["Cumulative_Damage"].mean()),
            "damage_increment_std": _f(inc.std()),
            "temp_mean": _f(sub["Daily_Max_Temp_C"].mean()),
            "humidity_mean": _f(sub["Daily_Median_Humidity"].mean()),
            "delta_temp_std": _f(sub["Daily_Std_Delta_Temp"].mean()),
        }

    target_stats = _stats(target_early)
    peers_stats = _stats(peers_early)

    early_table = pd.DataFrame(
        [target_stats, peers_stats],
        index=[DIAG_CELL, "PEERS"],
    )
    print(f"\n  Early-phase comparison (first {EARLY_PHASE_CUTOFF:.0f} days):")
    print(early_table.round(3).to_string())

    print("\n  Target / peers ratios in early phase:")
    for col in early_table.columns:
        if col == "n_days":
            continue
        p_val = _f(early_table.at["PEERS", col])
        t_val = _f(early_table.at[DIAG_CELL, col])
        if abs(p_val) > 1e-9:
            ratio = t_val / p_val
            flag = " [WARN]" if abs(ratio - 1.0) > 0.3 else ""
            print(f"    {col:<25} target={t_val:>10.4f}  peers={p_val:>10.4f}  ratio={ratio:>6.2f}x{flag}")

    # --- Figure 3a: early-phase bar plot ---
    metrics_to_plot: List[Tuple[str, str]] = [
        ("pce_mean", "Mean PCE"),
        ("pce_std", "PCE std"),
        ("damage_increment_std", "Damage inc. std"),
        ("temp_mean", "Mean max temp (C)"),
        ("humidity_mean", "Median humidity"),
        ("delta_temp_std", "Std of dTemp"),
    ]
    fig, axes = plt.subplots(1, 6, figsize=(20, 4))
    for ax, (key, title) in zip(axes, metrics_to_plot):
        t_val = _f(early_table.at[DIAG_CELL, key])
        p_val = _f(early_table.at["PEERS", key])
        ax.bar(["Target", "Peers"], [t_val, p_val], color=["#d62728", "#888888"])
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.3, axis="y")
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle(
        f"Layer 3a — Early-phase ({EARLY_PHASE_CUTOFF:.0f} d) feature comparison",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{DIAG_CELL}_03a_early_phase.png", dpi=150)
    plt.close(fig)

    # --- Figure 3b: lifecycle environmental features ---
    metrics_lifecycle: List[Tuple[str, str]] = [
        ("Daily_Max_Temp_C", "Daily max temperature (C)"),
        ("Daily_Median_Humidity", "Daily median humidity (g/m3)"),
        ("Daily_Mean_Delta_Temp", "Daily mean dTemp (C/h)"),
        ("Daily_Irradiance_Dose", "Daily irradiance dose (Wh/m2)"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    for ax, (key, title) in zip(axes, metrics_lifecycle):
        for cell in [DIAG_CELL] + reference_cells:
            cs = str(cell)
            sub = daily[daily["cell_name"] == cell].sort_values("Exposure_Days")
            x = np.asarray(sub["Exposure_Days"].to_numpy(), dtype=float)
            y = np.asarray(sub[key].to_numpy(), dtype=float)
            is_target = cs == DIAG_CELL
            ax.plot(x, y,
                    linewidth=2.2 if is_target else 1.0,
                    alpha=1.0 if is_target else 0.55,
                    label=cs,
                    zorder=3 if is_target else 1)
        ax.axvline(BURN_IN_DAYS_F, color="gray", linestyle=":", linewidth=1)
        ax.set_xlabel("Exposure (days)")
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.suptitle(
        f"Layer 3b — Full-lifecycle environmental features · target: {DIAG_CELL}",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{DIAG_CELL}_03b_lifecycle_features.png", dpi=150)
    plt.close(fig)

    return early_table


# ==============================================================================
# LAYER 4 — TARGET FORENSICS
# ==============================================================================
def layer4_target_forensics(
    daily: pd.DataFrame, reference_cells: List[str]
) -> pd.DataFrame:
    print("\n" + "=" * 80)
    print(f"# LAYER 4 — TARGET FORENSICS ({DIAG_CELL})")
    print("=" * 80)

    target = daily[daily["cell_name"] == DIAG_CELL].sort_values("Exposure_Days")

    # --- Continuity ---
    print("\n  [4.1] Telemetry continuity:")
    counts = target["n_points"]
    print(f"    Days with data: {len(target)}")
    print(f"    Mean points/day: {_f(counts.mean()):.1f}")
    print(f"    Median points/day: {_f(counts.median()):.1f}")
    print(f"    Min points/day: {_i(counts.min())}")

    threshold_cov = _f(counts.median()) * 0.5
    low_coverage = target[counts < threshold_cov]
    if not low_coverage.empty:
        print(f"    Days with < 50% coverage: {len(low_coverage)}")
        print(low_coverage[["Exposure_Days", "n_points"]].round(2).to_string(index=False))
    else:
        print("    No low-coverage days detected.")

    # --- Pre-failure window ---
    failure_day = _f(target["Exposure_Days"].max())
    window_start = max(0.0, failure_day - FAILURE_WINDOW_DAYS)

    print(f"\n  [4.2] Last observed day: {failure_day:.2f}")
    print(f"    Pre-failure window: [{window_start:.2f}, {failure_day:.2f}]")

    pre_failure = target[
        (target["Exposure_Days"] >= window_start)
        & (target["Exposure_Days"] <= failure_day)
    ].copy()
    cols = [
        "Exposure_Days", "PCE", "pFF", "Instant_Loss", "Cumulative_Damage",
        "Daily_Damage_Increment", "Daily_Max_Temp_C", "Daily_Median_Humidity",
        "Daily_Irradiance_Dose", "n_points",
    ]
    print("\n  Full feature record of the pre-failure window:")
    print(pre_failure[cols].round(4).to_string(index=False))

    # --- Figure 4a: pre-failure zoom ---
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    x_full = np.asarray(target["Exposure_Days"].to_numpy(), dtype=float)
    y_pce = np.asarray(target["PCE"].to_numpy(), dtype=float)
    y_dmg = np.asarray(target["Cumulative_Damage"].to_numpy(), dtype=float)

    axes[0].plot(x_full, y_pce, color="#1f77b4", linewidth=1.5,
                 marker="o", markersize=3, label="PCE (daily max)")
    axes[0].set_ylabel("PCE (%)")
    axes[0].set_title(f"Layer 4a — Pre-failure zoom · {DIAG_CELL}")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=9)

    axes[1].plot(x_full, y_dmg, color="#d62728", linewidth=2,
                 marker="o", markersize=3, label="Cumulative damage")
    axes[1].axhline(T80_DAMAGE_LIMIT, color="black", linestyle="--",
                    linewidth=1, label=f"T80 = {T80_DAMAGE_LIMIT:.2f}")
    axes[1].axvspan(window_start, failure_day, color="orange", alpha=0.15,
                    label=f"Pre-failure window ({FAILURE_WINDOW_DAYS:.0f} d)")
    axes[1].set_xlabel("Exposure (days)")
    axes[1].set_ylabel("Damage")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{DIAG_CELL}_04a_prefailure_zoom.png", dpi=150)
    plt.close(fig)

    # --- Figure 4b: forensics panel ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    panel: List[Tuple[Any, str, str]] = [
        (axes[0, 0], "Daily_Max_Temp_C", "Daily max temperature (C)"),
        (axes[0, 1], "Daily_Median_Humidity", "Daily median humidity (g/m3)"),
        (axes[1, 0], "Daily_Mean_Delta_Temp", "Mean dTemp (C/h)"),
        (axes[1, 1], "n_points", "Telemetry points per day"),
    ]
    for ax, key, title in panel:
        y = np.asarray(target[key].to_numpy(), dtype=float)
        ax.plot(x_full, y, color="#d62728", linewidth=1.5,
                marker="o", markersize=3)
        ax.axvspan(window_start, failure_day, color="orange", alpha=0.15)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
        if key == "n_points":
            ax.set_xlabel("Exposure (days)")
    fig.suptitle(f"Layer 4b — Target forensics panel · {DIAG_CELL}", y=1.0)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{DIAG_CELL}_04b_forensics_panel.png", dpi=150)
    plt.close(fig)

    print("\n  Pre-failure window saved to parquet for traceability.")
    return pre_failure


# ==============================================================================
# LAYER 5 — STATISTICAL DIAGNOSTICS
# ==============================================================================
def layer5_statistical_diagnostics(
    daily: pd.DataFrame, reference_cells: List[str]
) -> Dict[str, pd.DataFrame]:
    print("\n" + "=" * 80)
    print("# LAYER 5 — STATISTICAL DIAGNOSTICS")
    print("=" * 80)

    results: Dict[str, pd.DataFrame] = {}

    # --- 5.1 Thermal dose ---
    print("\n  [5.1] Thermal dose per cell:")
    thermal_rows: List[Dict[str, Any]] = []
    for cell in [DIAG_CELL] + reference_cells:
        cs = str(cell)
        sub = daily[daily["cell_name"] == cell]
        temps = np.asarray(sub["Daily_Max_Temp_C"].to_numpy(), dtype=float)
        row: Dict[str, Any] = {"cell": cs}
        for thr in TEMP_THRESHOLDS:
            row[f"days_above_{int(thr)}C"] = int((temps > thr).sum())
            row[f"pct_above_{int(thr)}C"] = float(100.0 * (temps > thr).mean())
        row["thermal_dose_35"] = float(np.maximum(0.0, temps - 35.0).sum())
        row["thermal_dose_40"] = float(np.maximum(0.0, temps - 40.0).sum())
        row["temp_mean"] = float(temps.mean())
        row["temp_max"] = float(temps.max())
        thermal_rows.append(row)
    thermal_table = pd.DataFrame(thermal_rows).set_index("cell")
    print(thermal_table.round(2).to_string())
    results["thermal_dose"] = thermal_table

    # --- 5.2 Photo-annealing ---
    print("\n  [5.2] Photo-annealing (recovery days):")
    annealing_rows: List[Dict[str, Any]] = []
    for cell in [DIAG_CELL] + reference_cells:
        cs = str(cell)
        sub = daily[daily["cell_name"] == cell].sort_values("Exposure_Days")
        inc = np.asarray(sub["Daily_Damage_Increment"].to_numpy(), dtype=float)
        recovery = int((inc < -0.001).sum())
        degradation = int((inc > 0.001).sum())
        flat = int(len(inc) - recovery - degradation)
        annealing_rows.append({
            "cell": cs,
            "n_days": int(len(sub)),
            "recovery_days": recovery,
            "degradation_days": degradation,
            "flat_days": flat,
            "pct_recovery": float(100.0 * recovery / max(1, len(sub))),
            "max_recovery": float(inc.min()),
            "max_degradation": float(inc.max()),
        })
    annealing_table = pd.DataFrame(annealing_rows).set_index("cell")
    print(annealing_table.round(4).to_string())
    results["annealing"] = annealing_table

    # --- 5.3 Cross-correlation ---
    print("\n  [5.3] Cross-correlation T_max(t) vs dDamage(t+lag):")
    lags: List[int] = list(range(0, 11))
    corr_rows: List[Dict[str, Any]] = []
    for cell in [DIAG_CELL] + reference_cells:
        cs = str(cell)
        sub = daily[daily["cell_name"] == cell].sort_values("Exposure_Days").reset_index(drop=True)
        temp = np.asarray(sub["Daily_Max_Temp_C"].to_numpy(), dtype=float)
        dmg = np.asarray(sub["Daily_Damage_Increment"].to_numpy(), dtype=float)

        row: Dict[str, Any] = {"cell": cs}
        best_corr, best_lag = 0.0, 0
        for lag in lags:
            if lag >= len(dmg):
                row[f"lag{lag}"] = float("nan")
                continue
            t_slice = temp[: len(temp) - lag] if lag > 0 else temp
            d_slice = dmg[lag:]
            if (len(t_slice) < 5
                    or float(np.std(t_slice)) == 0.0
                    or float(np.std(d_slice)) == 0.0):
                row[f"lag{lag}"] = float("nan")
                continue
            corr = float(np.corrcoef(t_slice, d_slice)[0, 1])
            row[f"lag{lag}"] = corr
            if abs(corr) > abs(best_corr):
                best_corr, best_lag = corr, int(lag)
        row["best_lag"] = best_lag
        row["best_corr"] = float(best_corr)
        corr_rows.append(row)
    corr_table = pd.DataFrame(corr_rows).set_index("cell")
    print(corr_table[["best_lag", "best_corr"]].round(3).to_string())
    results["cross_correlation"] = corr_table

    # --- 5.4 Feature correlations with damage increment ---
    print("\n  [5.4] Correlation of features with dDamage:")
    corr_features: List[str] = [
        "Daily_Max_Temp_C",
        "Daily_Median_Humidity",
        "Daily_Irradiance_Dose",
        "Daily_Mean_POA",
        "Daily_Mean_Delta_Temp",
        "Daily_Mean_Delta_Hum",
        "Cumulative_Damage",
    ]
    available = [c for c in corr_features if c in daily.columns]
    corr_cols: Dict[str, pd.Series] = {}
    for cell in [DIAG_CELL] + reference_cells:
        cs = str(cell)
        sub = daily[daily["cell_name"] == cell].sort_values("Exposure_Days")
        corr_series = (
            sub[available + ["Daily_Damage_Increment"]]
            .corr()["Daily_Damage_Increment"]
            .drop("Daily_Damage_Increment")
        )
        corr_cols[cs] = corr_series
    corr_df = pd.DataFrame(corr_cols)
    print(corr_df.round(3).to_string())
    results["feature_correlations"] = corr_df

    # --- Figure 5a: correlation matrices ---
    cells_to_plot = [DIAG_CELL] + reference_cells[:3]
    n_panels = len(cells_to_plot)
    fig, axes_arr = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4.5))
    axes_list: List[Any] = list(np.atleast_1d(axes_arr))

    last_im: Any = None
    cols_to_use = available + ["Daily_Damage_Increment"]
    for ax, cell in zip(axes_list, cells_to_plot):
        cs = str(cell)
        sub = daily[daily["cell_name"] == cell].sort_values("Exposure_Days")
        cols_present = [c for c in cols_to_use if c in sub.columns]
        corr = sub[cols_present].corr()
        last_im = ax.imshow(
            np.asarray(corr.to_numpy(), dtype=float),
            cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="auto",
        )
        ax.set_xticks(range(len(cols_present)))
        ax.set_yticks(range(len(cols_present)))
        short = [
            c.replace("Daily_", "").replace("Mean_", "Mn_").replace("_", "\n")[:14]
            for c in cols_present
        ]
        ax.set_xticklabels(short, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(short, fontsize=7)
        ax.set_title(cs, fontsize=10)

    if last_im is not None and len(axes_list) > 0:
        fig.colorbar(last_im, ax=axes_list[0], orientation="vertical",
                     shrink=0.7, fraction=0.05)

    fig.suptitle("Layer 5a — Feature correlation matrices", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{DIAG_CELL}_05a_correlation_matrices.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Figure 5b: cross-correlation curves ---
    fig, ax = plt.subplots(figsize=(10, 5))
    for cell in corr_table.index:
        cs = str(cell)
        vals = [_f(corr_table.at[cs, f"lag{k}"]) for k in lags]
        is_target = cs == DIAG_CELL
        ax.plot(lags, vals,
                linewidth=2.2 if is_target else 1.2,
                alpha=1.0 if is_target else 0.6,
                marker="o", markersize=4, label=cs)
    ax.axhline(0.0, color="black", linewidth=0.5)
    ax.set_xlabel("Lag (days)")
    ax.set_ylabel("Corr(T_max(t), dDamage(t+lag))")
    ax.set_title(
        f"Layer 5b — Cross-correlation temperature to damage · target: {DIAG_CELL}"
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{DIAG_CELL}_05b_cross_correlation.png", dpi=150)
    plt.close(fig)

    return results


# ==============================================================================
# LAYER 6 — FAILURE MODE ASSESSMENT
# ==============================================================================
def layer6_failure_assessment(
    daily: pd.DataFrame,
    reference_cells: List[str],
) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print("# LAYER 6 — FAILURE MODE ASSESSMENT")
    print("=" * 80)

    target = daily[daily["cell_name"] == DIAG_CELL].sort_values("Exposure_Days")
    target_early = target[target["Exposure_Days"] <= EARLY_PHASE_CUTOFF]
    inc = np.asarray(target["Daily_Damage_Increment"].to_numpy(), dtype=float)

    # --- Target metrics ------------------------------------------------
    # pce_std is computed on the EARLY phase only, not the full lifecycle:
    # late-life collapse inflates the full-cycle std and hides the
    # "abnormally stable during operation" signature we want to detect.
    # thermal_dose_35 captures both intensity and duration of thermal stress,
    # unlike temp_mean which only captures intensity.
    metrics: Dict[str, float] = {
        "max_jump": float(inc.max()),
        "damage_rate_mean": float(inc.mean()),
        "damage_rate_std": float(inc.std()),
        "recovery_pct": float(100.0 * (inc < -0.001).sum() / max(1, len(inc))),
        "temp_mean": _f(target["Daily_Max_Temp_C"].mean()),
        "pce_early_mean": _f(target_early["PCE"].mean()),
        "pce_std": _f(target_early["PCE"].std()),
        "thermal_dose_35": float(
            np.maximum(0.0, target["Daily_Max_Temp_C"].to_numpy(dtype=float) - 35.0).sum()
        ),
    }

    # --- Peer metrics --------------------------------------------------
    peers = daily[daily["cell_name"].isin(reference_cells)]
    peers_early = peers[peers["Exposure_Days"] <= EARLY_PHASE_CUTOFF]

    peer_doses: List[float] = []
    for pc in reference_cells:
        sub_p = daily[daily["cell_name"] == pc]
        peer_doses.append(float(
            np.maximum(0.0, sub_p["Daily_Max_Temp_C"].to_numpy(dtype=float) - 35.0).sum()
        ))
    peer_dose_mean = float(np.mean(peer_doses)) if peer_doses else 0.0

    peer_metrics: Dict[str, float] = {
        "max_jump": float(peers.groupby("cell_name")["Daily_Damage_Increment"].max().mean()),
        "temp_mean": _f(peers["Daily_Max_Temp_C"].mean()),
        "pce_early_mean": _f(peers_early["PCE"].mean()),
        "pce_std": _f(peers_early["PCE"].std()),
        "thermal_dose_35": peer_dose_mean,
    }

    # --- Evidence collection ------------------------------------------
    evidence: List[Dict[str, str]] = []

    # (1) Sudden collapse: single-day damage jump above absolute threshold.
    if metrics["max_jump"] > THRESHOLD_SUDDEN_JUMP:
        evidence.append({
            "flag": "sudden_collapse",
            "desc": (
                f"Maximum single-day damage jump of {metrics['max_jump']:.3f} "
                f"(> {THRESHOLD_SUDDEN_JUMP:.2f} typically indicates structural collapse)."
            ),
            "severity": "high",
        })

    # (2) Thermal dose stress: cumulative excess above 35 C, higher than
    #     peers by a meaningful margin. Captures intensity AND duration.
    if (peer_metrics["thermal_dose_35"] > 0.0
            and metrics["thermal_dose_35"] > peer_metrics["thermal_dose_35"] * THRESHOLD_THERMAL_DOSE_RATIO):
        evidence.append({
            "flag": "thermal_dose_stress",
            "desc": (
                f"Cumulative thermal dose above 35 C {metrics['thermal_dose_35']:.0f} "
                f"vs peer mean {peer_metrics['thermal_dose_35']:.0f} "
                f"({metrics['thermal_dose_35'] / peer_metrics['thermal_dose_35']:.2f}x)."
            ),
            "severity": "medium",
        })

    # (3) Low initial PCE: cell starts with significantly lower performance.
    if (peer_metrics["pce_early_mean"] > 0.0
            and metrics["pce_early_mean"] < peer_metrics["pce_early_mean"] * THRESHOLD_LOW_PCE_RATIO):
        evidence.append({
            "flag": "low_initial_pce",
            "desc": (
                f"Early-phase PCE {metrics['pce_early_mean']:.2f} vs peer mean "
                f"{peer_metrics['pce_early_mean']:.2f} "
                f"({metrics['pce_early_mean'] / peer_metrics['pce_early_mean']:.2f}x)."
            ),
            "severity": "medium",
        })

    # (4) Low PCE variance: cell behaves unusually stably during early phase.
    if (peer_metrics["pce_std"] > 0.0
            and metrics["pce_std"] < peer_metrics["pce_std"] * THRESHOLD_LOW_PCE_STD_RATIO):
        evidence.append({
            "flag": "low_pce_variance",
            "desc": (
                f"Early-phase PCE std {metrics['pce_std']:.3f} vs peer mean "
                f"{peer_metrics['pce_std']:.3f} "
                f"({metrics['pce_std'] / peer_metrics['pce_std']:.2f}x)."
            ),
            "severity": "low",
        })

    # (5) Monotonic degradation: fewer recovery days than peers.
    if metrics["recovery_pct"] < THRESHOLD_LOW_RECOVERY_PCT:
        evidence.append({
            "flag": "monotonic_degradation",
            "desc": (
                f"Only {metrics['recovery_pct']:.1f}% of days show damage "
                f"recovery (below {THRESHOLD_LOW_RECOVERY_PCT:.0f}%)."
            ),
            "severity": "low",
        })

    flags = {e["flag"] for e in evidence}

    # --- Classification rules -----------------------------------------
    if "sudden_collapse" in flags and "thermal_dose_stress" in flags:
        mode = "Thermal-dose-driven sudden collapse"
        confidence = "high"
    elif "sudden_collapse" in flags and "low_initial_pce" in flags:
        mode = "Sudden collapse of an intrinsically weak cell"
        confidence = "medium"
    elif "sudden_collapse" in flags:
        mode = "Sudden structural collapse (mechanism unknown)"
        confidence = "medium"
    elif "low_initial_pce" in flags and "low_pce_variance" in flags:
        mode = "Intrinsically weak cell with low starting performance"
        confidence = "medium"
    elif not evidence:
        mode = "No anomaly detected with current diagnostic rules"
        confidence = "low"
    else:
        mode = "Anomalous behavior (unspecified)"
        confidence = "low"

    # --- Recommendation -----------------------------------------------
    if mode == "No anomaly detected with current diagnostic rules":
        recommendation = "Cell behaves within cohort norms. No remediation needed."
    elif "thermal-dose-driven" in mode.lower():
        recommendation = (
            "Investigate thermal management: ventilation, shading, or "
            "deployment-site heat load. Cumulative dose exceeds peers by a "
            "wide margin; the collapse may be preventable by reducing peak "
            "temperatures during the operational phase."
        )
    elif "sudden" in mode.lower() and "weak" in mode.lower():
        recommendation = (
            "Inspect hardware for manufacturing defects. The combination of "
            "low starting performance and sudden collapse is characteristic of "
            "an intrinsically weak sample rather than environmental-driven "
            "degradation. Consider excluding from the healthy reference cohort."
        )
    elif "sudden" in mode.lower():
        recommendation = (
            "Investigate hardware: check for delamination, contact damage, or "
            "encapsulation failure at the collapse day. If cohort size is "
            "sufficient, consider a separate model for the sudden-failure regime."
        )
    elif "weak" in mode.lower():
        recommendation = (
            "Consider excluding from the healthy reference cohort. The cell's "
            "low starting performance suggests a manufacturing defect rather "
            "than environmental degradation."
        )
    else:
        recommendation = (
            "Increase cohort size to disambiguate. Current evidence is "
            "insufficient to prescribe a specific remediation."
        )

    # --- Print summary -------------------------------------------------
    print(f"\n  Target cell: {DIAG_CELL}")
    print(f"  Failure mode: {mode}")
    print(f"  Confidence: {confidence}")
    print(f"\n  Evidence ({len(evidence)} signals):")
    for e in evidence:
        print(f"    [{e['severity']:>6}] {e['flag']:<22} {e['desc']}")
    if not evidence:
        print("    (none)")
    print("\n  Recommendation:")
    print(f"    {recommendation}")

    return {
        "cell": DIAG_CELL,
        "failure_mode": mode,
        "confidence": confidence,
        "n_evidence": len(evidence),
        "recommendation": recommendation,
        "evidence_flags": ",".join(sorted(flags)) if flags else "",
    }


# ==============================================================================
# MAIN
# ==============================================================================
def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    CELL_DIAG_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting cell diagnosis for target: {DIAG_CELL}")

    daily, t80 = load_and_prepare()
    available_cells = sorted(str(c) for c in daily["cell_name"].unique())
    if DIAG_CELL not in available_cells:
        logger.error(
            f"Target cell '{DIAG_CELL}' not found. Available cells: {available_cells}"
        )
        return

    reference_cells = select_reference_cells(daily)
    logger.info(f"Reference peers selected: {reference_cells}")

    # ----- Run all layers -------------------------------------------------
    cohort_table = layer1_cohort_overview(daily, t80)
    layer2_trajectory_comparison(daily)
    early_table = layer3_matched_peers(daily, reference_cells)
    pre_failure = layer4_target_forensics(daily, reference_cells)
    stats = layer5_statistical_diagnostics(daily, reference_cells)
    assessment = layer6_failure_assessment(daily, reference_cells)

    # ----- Persist summary parquet ----------------------------------------
    summary_df = pd.DataFrame([{
        "target": str(assessment["cell"]),
        "reference_cells": ",".join(reference_cells),
        "failure_mode": str(assessment["failure_mode"]),
        "confidence": str(assessment["confidence"]),
        "recommendation": str(assessment["recommendation"]),
    }])
    summary_df.to_parquet(OUT_PARQUET, index=False)
    logger.info(f"Summary written -> {OUT_PARQUET}")

    # ----- Persist detailed tables ----------------------------------------
    cohort_table.to_parquet(DETAILED_DIR / "cohort_overview.parquet")
    early_table.to_parquet(DETAILED_DIR / "early_phase_comparison.parquet")
    pre_failure.to_parquet(DETAILED_DIR / "pre_failure_window.parquet")
    stats["thermal_dose"].to_parquet(DETAILED_DIR / "thermal_dose.parquet")
    stats["annealing"].to_parquet(DETAILED_DIR / "annealing.parquet")
    stats["cross_correlation"].to_parquet(DETAILED_DIR / "cross_correlation.parquet")
    stats["feature_correlations"].to_parquet(DETAILED_DIR / "feature_correlations.parquet")
    logger.info(f"Detailed tables written -> {DETAILED_DIR}")

    print("\n" + "=" * 80)
    print(" DIAGNOSIS COMPLETE")
    print("=" * 80)
    print(f" Target:          {DIAG_CELL}")
    print(f" Failure mode:    {assessment['failure_mode']}")
    print(f" Confidence:      {assessment['confidence']}")
    print(f" Figures:         {FIG_DIR}/{DIAG_CELL}_*.png")
    print(f" Summary table:   {OUT_PARQUET}")
    print(f" Detailed tables: {DETAILED_DIR}/")
    print("=" * 80)


def _run() -> None:
    original_stdout = sys.stdout
    OUT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_LOG, "w", encoding="utf-8") as f:
        sys.stdout = _Tee(original_stdout, f)
        try:
            main()
        finally:
            sys.stdout = original_stdout
    logger.info(f"Full report saved -> {OUT_LOG}")


if __name__ == "__main__":
    _run()
"""
Module: src/audit_multiparametric_twin.py
Description: Read-only audit extending the Dual Digital Twin (PCE + pFF) with
two additional orthogonal detection channels — Jsc and Voc — to test whether
the extra signals anticipate the failure of cells that die early and are
currently only caught by the physical gate.

Motivation
----------
The production Dual Digital Twin tracks two channels: PCE_rel and pFF_rel.
A perovskite cell can degrade through three orthogonal physical mechanisms:
  - Voc drop  → recombination, absorber or contact degradation
  - Jsc drop  → absorption loss, delamination, active area loss
  - FF drop   → series resistance, shunts, contact degradation
The current twin captures the FF channel directly (via pFF) and the joint
effect via PCE, but has no direct visibility on Voc or Jsc. If a cell is
failing dominantly through one of the unobserved channels, the current twin
may not accumulate enough alerts to cross the gate before the physical T80
collapse fires.

This script does NOT modify any production artifact. It re-runs the LOOCV
protocol offline with 2 and 4 channels, on the same cohort and the same
burn-in window, and reports:
  - per-cell alert frequencies under both configurations
  - the per-channel breakdown for the 4-channel run
  - whether the healthy cluster remains clean
  - whether the pathological cluster remains separated
  - whether early-death cells (currently evicted only by the physical gate)
    would have been flagged by the extended twin

Design constraints
------------------
  - Same burn-in W* = BURN_IN_DAYS
  - Same feature vector FEATURES
  - Same OOF residual threshold policy (3-layer max)
  - Same alert-frequency gate τ_F = ALERT_FREQUENCY_THRESHOLD_PCT
  - Same LOOCV protocol (train on other cells' mature phase, evaluate on the
    holdout's early phase)
  - Targets normalized by each channel's initial reference (PCE_0, pFF_0,
    Jsc_0, Voc_0) — matches the production convention after the pFF fix.
"""

from __future__ import annotations

import importlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb

from src.config import (
    ALERT_FREQUENCY_THRESHOLD_PCT,
    BURN_IN_DAYS,
    DAYLIGHT_IRRADIANCE_MIN_W_M2,
    DIAGNOSTICS_SCREENING_DIR,
    FEATURES,
    FILE_MERGED_FEATURES,
    FILE_SCREENING_ARTIFACTS,
    FILE_T80_TRUTH,
    INITIAL_REF_FLOOR,
    MIN_PHYSICAL_MAE_PCE_ABSOLUTE,
    MIN_PHYSICAL_MAE_PFF_ABSOLUTE,
    T80_INITIAL_PEAK_DAYS,
    XGB_PCE_PARAMS,
    XGB_PFF_PARAMS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("audit.multiparametric")


# ==============================================================================
# CHANNEL REGISTRY
# ==============================================================================
# The two channels that the production twin currently uses, and the two that
# this audit proposes to add. The names are the public keys used everywhere
# downstream (JSON, console output, column suffixes).
CHANNEL_2 = ("pce", "pff")
CHANNEL_4 = ("pce", "pff", "jsc", "voc")

# Target column in the prepared dataframe per channel.
CHANNEL_TARGETS: Dict[str, str] = {
    "pce": "PCE_Relative",
    "pff": "pFF_Relative",
    "jsc": "Jsc_Relative",
    "voc": "Voc_Relative",
}

# Absolute floor per channel (last-resort guardrail of the 3-layer policy).
# Jsc and Voc use the same small floor as PCE and pFF; the floor is only
# active when the OOF residuals degenerate, which is not expected here.
CHANNEL_FLOORS: Dict[str, float] = {
    "pce": MIN_PHYSICAL_MAE_PCE_ABSOLUTE,
    "pff": MIN_PHYSICAL_MAE_PFF_ABSOLUTE,
    "jsc": 0.01,
    "voc": 0.01,
}

# XGBoost hyperparameters per channel. Jsc and Voc use shallow regressors
# consistent with the pFF head: the added channels carry less signal and
# we do not want to overfit them to the mature phase.
CHANNEL_PARAMS: Dict[str, dict] = {
    "pce": XGB_PCE_PARAMS,
    "pff": XGB_PFF_PARAMS,
    "jsc": dict(
        n_estimators=100, learning_rate=0.05, max_depth=3,
        subsample=0.8, random_state=42, n_jobs=-1,
    ),
    "voc": dict(
        n_estimators=100, learning_rate=0.05, max_depth=3,
        subsample=0.8, random_state=42, n_jobs=-1,
    ),
}


# ==============================================================================
# IMPORTS FROM MODULE 07 (production)
# ==============================================================================
# The screening module name starts with a digit; import via importlib.
_screening = importlib.import_module("src.07_jv_mppt_early_screening")
preprocess_telemetry_data = _screening.preprocess_telemetry_data
oof_abs_residuals = _screening.oof_abs_residuals
_threshold = _screening._threshold


# ==============================================================================
# INITIAL REFERENCE COMPUTATION
# ==============================================================================
def _compute_initial_references(
    df: pd.DataFrame,
    irradiance_threshold: float = DAYLIGHT_IRRADIANCE_MIN_W_M2,
) -> pd.DataFrame:
    """
    Compute Jsc_initial and Voc_initial per cell, matching the convention
    used by module 06 for PCE_initial and pFF_initial: the daily maximum
    within the first T80_INITIAL_PEAK_DAYS days of exposure, per cell.

    Returns a DataFrame indexed by cell_name with columns
    ['Jsc_initial', 'Voc_initial'].

    Raises if the source dataframe does not contain the Jsc/Voc columns,
    because in that case the audit cannot run at all and the user must
    re-execute module 05 to regenerate the merged features.
    """
    required = {"Jsc", "Voc"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(
            f"FILE_MERGED_FEATURES is missing columns {sorted(missing)}. "
            "Re-run src/05_merge_jv_mppt.py to regenerate the feature matrix "
            "with the curve-level Voc/Jsc/FF columns."
        )

    df_proc = df.copy()
    if df_proc.index.name == "Timestamp":
        df_proc = df_proc.reset_index()

    df_proc["Datetime"] = pd.to_datetime(df_proc["Timestamp"], utc=True)
    df_proc["Date_Day"] = df_proc["Datetime"].dt.date
    df_daylight = df_proc[df_proc["POA_Irradiance_W_m2"] > irradiance_threshold].copy()

    # Daily max per cell per channel
    daily = (
        df_daylight.groupby(["cell_name", "Date_Day"])
        .agg(Jsc_max=("Jsc", "max"), Voc_max=("Voc", "max"))
        .reset_index()
        .sort_values(by=["cell_name", "Date_Day"])
    )

    # First T80_INITIAL_PEAK_DAYS days per cell, then take the peak
    first_days = daily.groupby("cell_name").head(T80_INITIAL_PEAK_DAYS)

    idx_jsc = first_days.groupby("cell_name")["Jsc_max"].idxmax()
    idx_voc = first_days.groupby("cell_name")["Voc_max"].idxmax()

    jsc_init = (
        first_days.loc[idx_jsc, ["cell_name", "Jsc_max"]]
        .rename(columns={"Jsc_max": "Jsc_initial"})
        .set_index("cell_name")
    )
    voc_init = (
        first_days.loc[idx_voc, ["cell_name", "Voc_max"]]
        .rename(columns={"Voc_max": "Voc_initial"})
        .set_index("cell_name")
    )

    references = jsc_init.join(voc_init, how="outer")

    n_missing = references.isna().any(axis=1).sum()
    if n_missing > 0:
        logger.warning(
            f"{n_missing} cells lack a valid Jsc/Voc initial reference. "
            "They will be dropped from the audit."
        )

    return references.dropna()


# ==============================================================================
# DATA PREPARATION
# ==============================================================================
def _prepare_data(
    df: pd.DataFrame,
    t80_metrics: pd.DataFrame,
    burn_in_days: float = BURN_IN_DAYS,
) -> pd.DataFrame:
    """
    Prepare the analysis dataframe with all four channels normalized.

    Follows the same pattern as train_and_evaluate_censored_twin:
      1. Preprocess telemetry (daylight filter, exposure days)
      2. Merge with T80 metrics (initials + combined survival)
      3. Truncate at combined_survival_days
      4. Dropna on features and targets
      5. Normalize all four targets by their initial reference
    """
    df_dl = preprocess_telemetry_data(df)

    references = _compute_initial_references(df)
    t80_extended = t80_metrics.join(references, how="left")

    df_dl = df_dl.merge(
        t80_extended[
            ["PCE_initial", "pFF_initial", "Jsc_initial", "Voc_initial",
             "combined_survival_days"]
        ],
        left_on="cell_name", right_index=True, how="inner",
    )

    df_censored = df_dl[
        df_dl["Exposure_Days"] <= df_dl["combined_survival_days"]
    ].copy()

    needed = FEATURES + [
        "PCE", "pFF", "Jsc", "Voc",
        "PCE_initial", "pFF_initial", "Jsc_initial", "Voc_initial",
    ]
    df_censored = df_censored.dropna(subset=needed).reset_index(drop=True)

    for ch, initial_col in [
        ("pce", "PCE_initial"),
        ("pff", "pFF_initial"),
        ("jsc", "Jsc_initial"),
        ("voc", "Voc_initial"),
    ]:
        src = {"pce": "PCE", "pff": "pFF", "jsc": "Jsc", "voc": "Voc"}[ch]
        rel_col = CHANNEL_TARGETS[ch]
        df_censored[rel_col] = (
            df_censored[src] / df_censored[initial_col].clip(lower=INITIAL_REF_FLOOR)
        )

    logger.info(
        f"Prepared {len(df_censored)} rows from "
        f"{df_censored['cell_name'].nunique()} cells with 4 normalized channels."
    )
    return df_censored


# ==============================================================================
# CORE LOOCV ENGINE (channel-parameterized)
# ==============================================================================
def _run_single_fold(
    df: pd.DataFrame,
    holdout: str,
    train_cells: List[str],
    channels: Tuple[str, ...],
    burn_in_days: float = BURN_IN_DAYS,
) -> Optional[Dict]:
    """
    Run a single LOOCV fold: train one twin per channel on the mature phase
    of train_cells, compute OOF thresholds on the same training set, and
    evaluate on the holdout's early window (t <= burn_in_days).

    Returns a dict with frequency and per-channel breakdown, or None if the
    fold cannot be evaluated (no training data).
    """
    train_mask = (
        df["cell_name"].isin(train_cells)
        & (df["Exposure_Days"] > burn_in_days)
    )
    early_mask = (
        (df["cell_name"] == holdout)
        & (df["Exposure_Days"] <= burn_in_days)
    )

    X_train = df.loc[train_mask, FEATURES]
    X_early = df.loc[early_mask, FEATURES]

    if len(X_train) == 0:
        return None

    alerts_per_channel: Dict[str, np.ndarray] = {}
    thresholds: Dict[str, float] = {}

    for ch in channels:
        target = CHANNEL_TARGETS[ch]
        y_train = df.loc[train_mask, target]
        y_early = df.loc[early_mask, target]

        # Threshold from OOF residuals of the training set only (never the holdout).
        residuals = oof_abs_residuals(
            lambda p=CHANNEL_PARAMS[ch]: xgb.XGBRegressor(**p),
            X_train, y_train,
        )
        thr = _threshold(residuals, CHANNEL_FLOORS[ch])
        thresholds[ch] = float(thr)

        if len(X_early) == 0:
            alerts_per_channel[ch] = np.array([], dtype=bool)
            continue

        model = xgb.XGBRegressor(**CHANNEL_PARAMS[ch]).fit(X_train, y_train)
        underperf = model.predict(X_early) - np.asarray(y_early)
        alerts_per_channel[ch] = underperf > thr

    # Combined alert (OR of all channels)
    if len(X_early) > 0:
        combined = np.zeros(len(X_early), dtype=bool)
        for ch in channels:
            combined |= alerts_per_channel[ch]
        freq = 100.0 * combined.sum() / len(combined)
    else:
        freq = float("nan")

    per_channel_freq = {
        ch: (
            float(100.0 * alerts_per_channel[ch].sum() / len(alerts_per_channel[ch]))
            if len(alerts_per_channel[ch]) > 0 else float("nan")
        )
        for ch in channels
    }

    return {
        "holdout": holdout,
        "channels": list(channels),
        "n_train_points": int(len(X_train)),
        "n_early_points": int(len(X_early)),
        "freq_pct": float(freq),
        "per_channel_freq_pct": per_channel_freq,
        "thresholds": thresholds,
    }


# ==============================================================================
# FULL COMPARISON TABLE
# ==============================================================================
def _build_comparison_table(
    df: pd.DataFrame,
    t80_metrics: pd.DataFrame,
    training_cohort: List[str],
    burn_in_days: float = BURN_IN_DAYS,
) -> pd.DataFrame:
    """
    For every cell in t80_metrics, run the LOOCV fold with 2 and 4 channels
    using the same training cohort, and return a comparison table.

    Training cohort semantics:
      - If the holdout is IN the training cohort, it is excluded from training
        (standard LOOCV).
      - If the holdout is NOT in the training cohort (e.g. it died physically
        before burn-in), training uses the full cohort. This mimics what the
        pipeline would do if the cell were presented as a new holdout.
    """
    all_cells = t80_metrics.index.tolist()
    rows: List[Dict] = []

    for holdout in all_cells:
        train_cells = [c for c in training_cohort if c != holdout]
        if len(train_cells) == 0:
            logger.warning(f"No training cells available for holdout {holdout}. Skipping.")
            continue

        r2 = _run_single_fold(df, holdout, train_cells, CHANNEL_2, burn_in_days)
        r4 = _run_single_fold(df, holdout, train_cells, CHANNEL_4, burn_in_days)

        if r2 is None or r4 is None:
            logger.warning(f"Fold for {holdout} could not be evaluated. Skipping.")
            continue

        survival = float(str(t80_metrics.loc[holdout, "combined_survival_days"]))
        survival_pce = float(str(t80_metrics.loc[holdout, "survival_days_pce"]))
        survival_pff = float(str(t80_metrics.loc[holdout, "survival_days_pff"]))

        rows.append({
            "cell": holdout,
            "combined_survival_days": survival,
            "survival_days_pce": survival_pce,
            "survival_days_pff": survival_pff,
            "in_training_cohort": holdout in training_cohort,
            "gate_physical": survival <= burn_in_days,
            "freq_2ch_pct": r2["freq_pct"],
            "freq_4ch_pct": r4["freq_pct"],
            "delta_pct": r4["freq_pct"] - r2["freq_pct"],
            "n_early_points": r2["n_early_points"],
            "per_channel_2ch": r2["per_channel_freq_pct"],
            "per_channel_4ch": r4["per_channel_freq_pct"],
            "thresholds_4ch": r4["thresholds"],
        })

    return pd.DataFrame(rows)


# ==============================================================================
# VERDICT
# ==============================================================================
def _compute_verdict(table: pd.DataFrame, burn_in_days: float) -> Dict:
    """
    Analyze the comparison table and produce decision flags.

    Definitions:
      - 'healthy_cluster'     : cells in the training cohort (production cohort).
      - 'pathological_cluster': cells NOT in the training cohort whose survival
                                is short (they were excluded, but the twin sees them).
      - 'early_deaths'        : cells with combined_survival <= burn_in_days.

    Key questions:
      1. Does the healthy cluster stay clean (< 10% under 4ch)?
      2. Does the pathological cluster stay separated (> 25% under 4ch)?
      3. Do early-death cells cross the gate under 4ch when they did not under 2ch?
      4. Is the gap between clusters preserved?
    """
    healthy = table[table["in_training_cohort"]]
    early_deaths = table[table["gate_physical"]]

    healthy_max_2ch = float(healthy["freq_2ch_pct"].max()) if len(healthy) else float("nan")
    healthy_max_4ch = float(healthy["freq_4ch_pct"].max()) if len(healthy) else float("nan")

    patho_min_2ch = float(early_deaths["freq_2ch_pct"].min()) if len(early_deaths) else float("nan")
    patho_min_4ch = float(early_deaths["freq_4ch_pct"].min()) if len(early_deaths) else float("nan")

    # Cells that crossed τ_F only under 4ch (i.e. the extra channels rescued)
    rescued = table[
        (table["freq_2ch_pct"] <= ALERT_FREQUENCY_THRESHOLD_PCT)
        & (table["freq_4ch_pct"] > ALERT_FREQUENCY_THRESHOLD_PCT)
    ]["cell"].tolist()

    # Cells that crossed τ_F only under 2ch (i.e. the extra channels lost them)
    lost = table[
        (table["freq_2ch_pct"] > ALERT_FREQUENCY_THRESHOLD_PCT)
        & (table["freq_4ch_pct"] <= ALERT_FREQUENCY_THRESHOLD_PCT)
    ]["cell"].tolist()

    healthy_contaminated = healthy[healthy["freq_4ch_pct"] > ALERT_FREQUENCY_THRESHOLD_PCT]["cell"].tolist()

    gap_2ch = patho_min_2ch - healthy_max_2ch if not (np.isnan(healthy_max_2ch) or np.isnan(patho_min_2ch)) else float("nan")
    gap_4ch = patho_min_4ch - healthy_max_4ch if not (np.isnan(healthy_max_4ch) or np.isnan(patho_min_4ch)) else float("nan")

    return {
        "healthy_cluster_max_2ch_pct": healthy_max_2ch,
        "healthy_cluster_max_4ch_pct": healthy_max_4ch,
        "pathological_cluster_min_2ch_pct": patho_min_2ch,
        "pathological_cluster_min_4ch_pct": patho_min_4ch,
        "gap_2ch_pp": gap_2ch,
        "gap_4ch_pp": gap_4ch,
        "early_deaths_rescued": rescued,
        "early_deaths_lost": lost,
        "healthy_contaminated_4ch": healthy_contaminated,
        "safe_to_promote": (
            len(healthy_contaminated) == 0
            and (not np.isnan(gap_4ch)) and gap_4ch > 10.0
        ),
    }


# ==============================================================================
# MAIN
# ==============================================================================
def main() -> None:
    import joblib

    DIAGNOSTICS_SCREENING_DIR.mkdir(parents=True, exist_ok=True)

    if not FILE_MERGED_FEATURES.exists():
        logger.error(f"Missing {FILE_MERGED_FEATURES}. Run modules 01–05 first.")
        raise SystemExit(1)
    if not FILE_T80_TRUTH.exists():
        logger.error(f"Missing {FILE_T80_TRUTH}. Run module 06 first.")
        raise SystemExit(1)
    if not FILE_SCREENING_ARTIFACTS.exists():
        logger.error(f"Missing {FILE_SCREENING_ARTIFACTS}. Run module 07 first.")
        raise SystemExit(1)

    df_raw = pd.read_parquet(FILE_MERGED_FEATURES)
    t80_metrics = pd.read_parquet(FILE_T80_TRUTH)
    artifacts = joblib.load(FILE_SCREENING_ARTIFACTS)

    training_cohort = artifacts.get("healthy_cohort", [])
    logger.info(f"Training cohort from production artifact: {training_cohort}")

    if not training_cohort:
        logger.error("Empty training cohort in screening artifacts. Aborting.")
        raise SystemExit(1)

    # Prepare normalized data for all 4 channels
    df = _prepare_data(df_raw, t80_metrics, BURN_IN_DAYS)

    # Run the LOOCV for both configurations on every cell
    logger.info("Running 2-channel and 4-channel LOOCV across all cells...")
    table = _build_comparison_table(df, t80_metrics, training_cohort, BURN_IN_DAYS)

    if table.empty:
        logger.error("Empty comparison table. Aborting.")
        raise SystemExit(1)

    table = table.sort_values("freq_4ch_pct", ascending=False).reset_index(drop=True)

    # Verdict
    verdict = _compute_verdict(table, BURN_IN_DAYS)

    # Persist as JSON (drop non-serializable columns if any)
    output = {
        "burn_in_days": BURN_IN_DAYS,
        "training_cohort": training_cohort,
        "channel_2ch": list(CHANNEL_2),
        "channel_4ch": list(CHANNEL_4),
        "alert_frequency_threshold_pct": ALERT_FREQUENCY_THRESHOLD_PCT,
        "per_cell": table.to_dict(orient="records"),
        "verdict": verdict,
    }
    out_path = DIAGNOSTICS_SCREENING_DIR / "multiparametric_audit.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=float)
    logger.info(f"Audit written to {out_path}")

    # Console summary
    print("\n" + "=" * 88)
    print(" MULTIPARAMETRIC TWIN AUDIT — 2 channels (PCE, pFF) vs 4 channels (+ Jsc, Voc)")
    print("=" * 88)

    header = (
        f"{'cell':<12} {'t80(comb)':>10} {'gate':>5} "
        f"{'2ch %':>8} {'4ch %':>8} {'Δ':>8} "
        f"{'PCE':>6} {'pFF':>6} {'Jsc':>6} {'Voc':>6}"
    )
    print("\n" + header)
    print("-" * len(header))
    for _, row in table.iterrows():
        gate = "PHY" if row["gate_physical"] else "   "
        pc = row["per_channel_4ch"]
        print(
            f"{row['cell']:<12} {row['combined_survival_days']:>10.2f} {gate:>5} "
            f"{row['freq_2ch_pct']:>8.2f} {row['freq_4ch_pct']:>8.2f} "
            f"{row['delta_pct']:>+8.2f} "
            f"{pc.get('pce', float('nan')):>6.1f} "
            f"{pc.get('pff', float('nan')):>6.1f} "
            f"{pc.get('jsc', float('nan')):>6.1f} "
            f"{pc.get('voc', float('nan')):>6.1f}"
        )

    print("\n" + "=" * 88)
    print(" VERDICT")
    print("=" * 88)
    print(f"  Healthy cluster max:     2ch = {verdict['healthy_cluster_max_2ch_pct']:.2f} %"
          f"   4ch = {verdict['healthy_cluster_max_4ch_pct']:.2f} %")
    print(f"  Early-death cluster min: 2ch = {verdict['pathological_cluster_min_2ch_pct']:.2f} %"
          f"   4ch = {verdict['pathological_cluster_min_4ch_pct']:.2f} %")
    print(f"  Gap between clusters:    2ch = {verdict['gap_2ch_pp']:.2f} pp"
          f"   4ch = {verdict['gap_4ch_pp']:.2f} pp")
    print(f"  Healthy contaminated by 4ch: {verdict['healthy_contaminated_4ch'] or 'none'}")
    print(f"  Early deaths rescued by 4ch: {verdict['early_deaths_rescued'] or 'none'}")
    print(f"  Early deaths lost by 4ch:    {verdict['early_deaths_lost'] or 'none'}")
    print(f"\n  Safe to promote 4ch to production: "
          f"{'YES' if verdict['safe_to_promote'] else 'NO'}")
    print()


if __name__ == "__main__":
    main()
"""
Module: src/02_jv_filtering.py
Description: Physical filtering, artifact removal, and quality control pipeline for J-V curves.
Implements vectorized thermodynamic boundary checks, robust chronometric sorting,
and memory-safe physical parameter extraction (Voc, Jsc, FF, Pmpp).
"""

from __future__ import annotations

import gc
import json
import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from tqdm import tqdm

from src.config import (
    OPERATIONAL_HOUR_END,
    OPERATIONAL_HOUR_START,
    CELL_AREA_M2,
    DIR_PROCESSED,
    FILE_JV_FILTERED,
    FILE_FILTERING_META,
    DEPLOYMENT_TIMEZONE, 
)

# Professional MLOps logging configuration.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("Filtering")

# Active cell area in cm^2, reused for current-density conversion.
CELL_AREA_CM2 = float(CELL_AREA_M2 * 10_000.0)

# Spike threshold: fraction of the curve's current span above which a
# point-to-point jump is considered an outlier.
SPIKE_RELATIVE_THRESHOLD = 0.15


# ==============================================================================
# 1. PER-DEVICE FILTERING TOPOLOGY
# ==============================================================================
def _process_single_cell(
    df: pd.DataFrame,
    name: str,
    cell_id: int,
    v_span_thresh: int,
    freeze_thresh: int,
    i_span_thresh: float,
    corruption_tol: float = 0.90,
    spike_tol: int = 2,
    frozen_ratio_tol: float = 0.90,
    mean_mismatch_tol: float = 0.60,
) -> pd.DataFrame:
    """
    Apply a vectorized physical, temporal, and structural filtering topology to a
    single device's raw J-V telemetry.

    Args:
        df: Raw DataFrame containing J-V telemetry.
        name: Device identifier.
        cell_id: Numeric mapping for the device.
        v_span_thresh: Minimum required voltage span (mV).
        freeze_thresh: Rolling window size for hardware freeze detection.
        i_span_thresh: Minimum required current span (A).
        corruption_tol: Max allowable fraction of anomalous negative current readings.
        spike_tol: Maximum allowable outlier spikes per curve.
        frozen_ratio_tol: Maximum allowable frozen-point ratio per curve.
        mean_mismatch_tol: Maximum allowable relative forward/reverse mean mismatch.

    Returns:
        pd.DataFrame: Processed dataset with all QC boolean masks applied.
    """
    df_c = df.copy()

    # --- 0. PREPARATION & STRICT CHRONOLOGICAL SORTING ---
    # Enforcing UTC prevents future daylight saving time (DST) merge collisions.
    df_c["Timestamp"] = pd.to_datetime(df_c["Timestamp"], utc=True)
    df_c = df_c.sort_values("Timestamp").reset_index(drop=True)

    # Downcasting floats drastically reduces RAM footprint (~50% memory savings).
    df_c["Voltage_V"] = df_c["Voltage_V"].astype("float32")
    if "Current_A" in df_c.columns:
        df_c["Current_A"] = df_c["Current_A"].astype("float32")
    if "Power_mW" in df_c.columns:
        df_c["Power_mW"] = df_c["Power_mW"].astype("float32")

    # --- 1. CYCLE IDENTIFICATION ---
    # A new cycle strictly initiates upon detecting a 'Reverse' scan direction.
    is_reverse_start = (df_c["ScanDirection"] == "Reverse") & (
        df_c["ScanDirection"].shift(1) != "Reverse"
    )
    is_reverse_start.iloc[0] = True

    df_c["curve"] = is_reverse_start.cumsum()
    df_c["cell_name"] = name
    df_c["cell_id"] = cell_id
    df_c["is_reverse"] = (df_c["ScanDirection"] == "Reverse").astype("int8")

    # --- 2. VECTORIZED POINT-LEVEL METRICS ---
    # Daylight window is defined in local time (Europe/Madrid), not UTC.
    local_hours = df_c["Timestamp"].dt.tz_convert(DEPLOYMENT_TIMEZONE).dt.hour
    df_c["is_in_time"] = local_hours.between(OPERATIONAL_HOUR_START, OPERATIONAL_HOUR_END)

    # [PHYSICS FIX]: Negative Current Isolation.
    # Current is naturally negative when exceeding Voc (diode injection regime).
    # Flag as anomalous ONLY if it occurs at low voltages (< 0.5 V).
    df_c["is_anomalous_negative"] = (df_c["Current_A"] < 0) & (df_c["Voltage_V"] < 0.5)

    # Hardware freeze detection (rolling span within each curve).
    v_rolling = df_c.groupby("curve", sort=False)["Voltage_V"].rolling(
        freeze_thresh, min_periods=1
    )
    i_rolling = df_c.groupby("curve", sort=False)["Current_A"].rolling(
        freeze_thresh, min_periods=1
    )
    v_span_window = v_rolling.max().droplevel(0) - v_rolling.min().droplevel(0)
    i_span_window = i_rolling.max().droplevel(0) - i_rolling.min().droplevel(0)
    df_c["is_frozen_point"] = (v_span_window <= 1e-4) | (i_span_window <= 1e-6)

    # Spike detection preparation.
    df_c["d_current_abs"] = df_c.groupby("curve")["Current_A"].diff().abs().bfill()

    # --- 3. CONSOLIDATED CURVE-LEVEL AGGREGATION ---
    # A single massive aggregation eliminates extreme Pandas overhead.
    curve_stats = df_c.groupby("curve").agg(
        v_max=("Voltage_V", "max"),
        v_min=("Voltage_V", "min"),
        i_max=("Current_A", "max"),
        i_min=("Current_A", "min"),
        neg_ratio=("is_anomalous_negative", "mean"),
        in_time_window=("is_in_time", "all"),
        frozen_ratio=("is_frozen_point", "mean"),
    )

    curve_stats["v_span_mV"] = (curve_stats["v_max"] - curve_stats["v_min"]) * 1000
    curve_stats["i_span_A"] = curve_stats["i_max"] - curve_stats["i_min"]

    # Global boundary conditions.
    curve_stats["is_corrupted"] = curve_stats["neg_ratio"] > corruption_tol
    curve_stats["is_curve_frozen"] = curve_stats["frozen_ratio"] > frozen_ratio_tol
    curve_stats["is_low_voltage"] = curve_stats["v_span_mV"] <= v_span_thresh
    curve_stats["is_low_current"] = curve_stats["i_span_A"] <= i_span_thresh
    curve_stats["is_night_time"] = ~curve_stats["in_time_window"]

    # Map current span back to evaluate relative spikes.
    df_c["i_span_A"] = df_c["curve"].map(curve_stats["i_span_A"])
    df_c["is_outlier_point"] = (df_c["d_current_abs"] / df_c["i_span_A"].add(1e-9)) > SPIKE_RELATIVE_THRESHOLD
    curve_stats["is_spike_error"] = df_c.groupby("curve")["is_outlier_point"].sum() > spike_tol

    # --- 4. STRUCTURAL INTEGRITY (FORWARD VS REVERSE) ---
    # pivot_table always returns a DataFrame with MultiIndex columns, even when
    # only one scan direction is present. `unstack` would collapse to a Series
    # in that case, which breaks column renaming.
    dir_stats = df_c.pivot_table(
        index="curve",
        columns="is_reverse",
        values="Current_A",
        aggfunc=["count", "mean"],
        fill_value=0,
    )
    dir_stats.columns = [f"{stat}_{int(rev)}" for stat, rev in dir_stats.columns]

    # Failsafe extraction to prevent KeyError on unidirectional artifacts.
    zero_index = pd.Series(0, index=dir_stats.index)
    fwd_len = dir_stats.get("count_0", zero_index)
    rev_len = dir_stats.get("count_1", zero_index)
    fwd_mean = dir_stats.get("mean_0", zero_index)
    rev_mean = dir_stats.get("mean_1", zero_index)

    ratio = fwd_len / (rev_len + 1e-9)
    curve_stats["is_asymmetric"] = (fwd_len == 0) | (rev_len == 0) | (ratio < 0.85) | (ratio > 1.15)

    # [PHYSICS FIX]: Relaxed hysteresis discrepancy to 0.60 to preserve severely degraded cells.
    max_mean = np.maximum(fwd_mean, rev_mean)
    curve_stats["is_mean_mismatch"] = (fwd_mean - rev_mean).abs() / (max_mean + 1e-9) > mean_mismatch_tol

    # --- 5. FINALIZE MASK & MEMORY CLEANUP ---
    invalid_mask = (
        curve_stats["is_low_voltage"]
        | curve_stats["is_low_current"]
        | curve_stats["is_night_time"]
        | curve_stats["is_curve_frozen"]
        | curve_stats["is_corrupted"]
        | curve_stats["is_spike_error"]
        | curve_stats["is_asymmetric"]
        | curve_stats["is_mean_mismatch"]
    )
    curve_stats["is_curve_valid"] = (~invalid_mask).astype("int8")

    validity_cols = [
        "is_low_voltage", "is_low_current", "is_night_time", "is_curve_frozen",
        "is_corrupted", "is_spike_error", "is_asymmetric", "is_mean_mismatch",
        "is_curve_valid",
    ]

    df_c = df_c.merge(curve_stats[validity_cols], left_on="curve", right_index=True, how="left")

    # Point-level artifact dropping (removes noise without dropping the entire curve).
    df_c = df_c[~(df_c["is_anomalous_negative"] & (~df_c["is_corrupted"]))].copy()
    df_c = df_c[~(df_c["is_outlier_point"] & (~df_c["is_spike_error"]))].copy()

    # Flush temporary heavy arrays from RAM.
    df_c = df_c.drop(
        columns=[
            "is_in_time", "is_anomalous_negative", "is_frozen_point",
            "d_current_abs", "is_outlier_point", "i_span_A",
        ]
    )

    return df_c


# ==============================================================================
# 2. PHYSICS PARAMETER EXTRACTION
# ==============================================================================
def _interp_zero_crossing(x: np.ndarray, y: np.ndarray, positive_x_only: bool = False) -> float:
    """
    Return the x-value at which y crosses zero, using linear interpolation between
    the two bracketing samples. Falls back to the sample with minimum |y| if no
    sign change is detected.

    Args:
        x: Independent-variable array (sorted ascending).
        y: Dependent-variable array.
        positive_x_only: If True, ignore sign changes at x <= 0 (useful for Voc).
    """
    sign_y = np.sign(y)
    sign_changes = np.where(np.diff(sign_y) != 0)[0]

    if positive_x_only:
        sign_changes = sign_changes[x[sign_changes] > 0.1]

    if len(sign_changes) > 0:
        i = sign_changes[0]
        x0, x1 = x[i], x[i + 1]
        y0, y1 = y[i], y[i + 1]
        if y1 - y0 != 0:
            return float(x0 - y0 * (x1 - x0) / (y1 - y0))

    return float(x[np.argmin(np.abs(y))])


def extract_physics_parameters(df_curve: pd.DataFrame) -> dict:
    """
    Extract physical scalars (Voc, Jsc, FF, Pmpp) from a single cleaned J-V sweep.

    Returns a dictionary for fast DataFrame reconstruction.
    """
    df_curve = df_curve.sort_values(by="Voltage_V")

    V = df_curve["Voltage_V"].to_numpy(dtype=float)
    I_A = df_curve["Current_A"].to_numpy(dtype=float)
    J = (I_A * 1000.0) / CELL_AREA_CM2  # mA/cm^2
    P = V * J                            # mW/cm^2

    results = {
        "id_curve": int(df_curve["id_curve"].iloc[0]),
        "voc": np.nan, "jsc": np.nan, "ff": np.nan,
        "v_mpp": np.nan, "j_mpp": np.nan, "p_mpp": np.nan,
        "err_mpp": 0, "err_voc": 0, "err_jsc": 0, "err_ff": 0,
    }

    if len(V) < 10:
        results.update({"err_mpp": 1, "err_voc": 1, "err_jsc": 1, "err_ff": 1})
        return results

    # --- A. MPP EXTRACTION (Savitzky-Golay smoothing) ---
    try:
        window = int(min(11, len(V)))
        if window % 2 == 0:
            window -= 1

        P_smooth = (
            np.asarray(savgol_filter(P, window_length=window, polyorder=2))
            if window > 3
            else P
        )

        mpp_idx = int(np.argmax(P_smooth))
        results["p_mpp"] = float(P_smooth[mpp_idx])
        results["v_mpp"] = float(V[mpp_idx])
        results["j_mpp"] = float(J[mpp_idx])

        if results["p_mpp"] <= 0 or results["v_mpp"] <= 0 or results["j_mpp"] <= 0:
            results["err_mpp"] = 1
    except Exception:
        results["err_mpp"] = 1

    # --- B. Voc EXTRACTION (zero crossing of J at V > 0) ---
    try:
        nearest = np.argsort(np.abs(J))[:5]
        mean_v_near = float(np.mean(V[nearest]))

        results["voc"] = _interp_zero_crossing(V, J, positive_x_only=True)

        if mean_v_near != 0 and (results["voc"] / mean_v_near >= 1.5 or results["voc"] <= 0):
            results["voc"] = mean_v_near
    except Exception:
        results["err_voc"] = 1

    # --- C. Jsc EXTRACTION (zero crossing of V at J > 0) ---
    try:
        nearest = np.argsort(np.abs(V))[:5]
        mean_j_near = float(np.mean(J[nearest]))

        # Jsc is the current density at V = 0. The sweep starts at V = 0,
        # so the first sample is typically exact; interpolation handles the rest.
        if V.min() <= 0 <= V.max() and V.min() < V.max():
            results["jsc"] = _interp_zero_crossing(V, J)
        else:
            results["jsc"] = float(J[np.argmin(np.abs(V))])

        if mean_j_near != 0 and (results["jsc"] / mean_j_near >= 1.5 or results["jsc"] <= 0):
            results["jsc"] = mean_j_near
    except Exception:
        results["err_jsc"] = 1

    # --- D. FILL FACTOR (FF) CALCULATION ---
    v_mpp, j_mpp = results["v_mpp"], results["j_mpp"]
    voc, jsc = results["voc"], results["jsc"]

    if all(pd.notna(x) for x in (v_mpp, j_mpp, voc, jsc)) and voc > 0 and jsc > 0:
        results["ff"] = float((v_mpp * j_mpp) / (voc * jsc))
        if not (0.0 < results["ff"] < 1.0):
            results["ff"] = np.nan
            results["err_ff"] = 1
    else:
        results["err_ff"] = 1

    return results


def extract_all_physics_parameters(jv_dataset: pd.DataFrame, batch_size: int = 500) -> pd.DataFrame:
    """
    Extract physical parameters for every valid curve, iterating lazily to keep
    memory usage bounded regardless of dataset size.
    """
    logger.info("Extracting physical parameters in memory-safe batches...")
    gc.collect()

    valid_df = jv_dataset[jv_dataset["is_curve_valid"] == 1]
    total_curves = int(valid_df["id_curve"].nunique())
    logger.info(f"Total valid curves to process: {total_curves:,}")

    if total_curves == 0:
        return pd.DataFrame()

    extracted: list[dict] = []
    grouped = valid_df.groupby("id_curve", sort=False)

    for i, (_, group) in enumerate(tqdm(grouped, total=total_curves, desc="Extracting Physics")):
        extracted.append(extract_physics_parameters(group))
        if (i + 1) % batch_size == 0:
            gc.collect()

    return pd.DataFrame(extracted)


# ==============================================================================
# 3. CHRONOMETRIC DIAGNOSTICS
# ==============================================================================
def analyze_curve_timings(jv_df: pd.DataFrame) -> None:
    """
    Extract and log chronological metadata: sweep durations and hardware idle intervals.
    Daily resampling is performed in deployment-local time (Europe/Madrid).
    """
    logger.info("Computing chronometric metadata (sweep durations and sensor idle intervals)...")

    all_curves_stats = (
        jv_df.groupby(["cell_name", "id_curve"])
        .agg(
            t_min=("Timestamp", "min"),
            t_max=("Timestamp", "max"),
            is_valid=("is_curve_valid", "max"),
        )
        .sort_values("t_min")
    )

    global_durations: list[float] = []
    global_intervals: list[float] = []
    report_lines = ["\n=== TEMPORAL ANALYSIS PER DEVICE ==="]

    for cell in jv_df["cell_name"].unique():
        cell_stats = all_curves_stats.loc[cell]

        intervals = cell_stats["t_min"].diff().dt.total_seconds()
        intervals_clean = intervals[intervals < 3600]
        avg_interval = intervals_clean.mean()
        global_intervals.extend(intervals_clean.dropna().tolist())

        valid_stats = cell_stats[cell_stats["is_valid"] == 1]
        durations = (valid_stats["t_max"] - valid_stats["t_min"]).dt.total_seconds()
        avg_duration = durations.mean()
        global_durations.extend(durations.dropna().tolist())

        total_valid = len(valid_stats)

        # Peak daily cycle volume, computed in deployment-local time.
        valid_df = jv_df[(jv_df["cell_name"] == cell) & (jv_df["is_curve_valid"] == 1)]
        if not valid_df.empty:
            local_ts = valid_df["Timestamp"].dt.tz_convert(DEPLOYMENT_TIMEZONE)
            valid_local = valid_df.assign(_local=local_ts).set_index("_local")
            max_daily = int(valid_local.resample("D")["id_curve"].nunique().max())
        else:
            max_daily = 0

        report_lines.extend(
            [
                f"[{cell}]",
                f"  -> Valid cycles retained: {total_valid}",
                f"  -> Mean duration (Rev+Fwd): {avg_duration:.2f} s" if total_valid > 0 else "  -> Mean duration: N/A",
                f"  -> Mean idle interval: {avg_interval:.2f} s" if not np.isnan(avg_interval) else "  -> Mean idle interval: N/A",
                f"  -> Peak daily cycle volume: {max_daily}\n",
            ]
        )

    report_lines.extend(
        [
            "=" * 40,
            "=== AGGREGATED GLOBAL TEMPORAL METRICS ===",
            f"Mean Global Sweep Duration: {np.mean(global_durations):.2f} s" if global_durations else "Mean Global Sweep Duration: N/A",
            f"Mean Global Hardware Idle:  {np.mean(global_intervals):.2f} s" if global_intervals else "Mean Global Hardware Idle: N/A",
            "=" * 40,
        ]
    )

    logger.info("\n".join(report_lines))


# ==============================================================================
# 4. ORCHESTRATION
# ==============================================================================
def _load_and_filter_devices(device_dirs: list[str]) -> Optional[pd.DataFrame]:
    """Load each device's raw J-V parquet and run the per-device filtering topology."""
    all_cleaned_dfs: list[pd.DataFrame] = []

    for cell_id, name in enumerate(device_dirs):
        parquet_path = DIR_PROCESSED / name / f"{name}_jv.parquet"
        if not parquet_path.exists():
            logger.warning(f"Expected Parquet artifact missing for device: {name}")
            continue

        logger.info(f"Loading raw telemetry for device: {name}")
        df = pd.read_parquet(parquet_path)

        if df.empty:
            logger.warning(f"Parquet artifact for device {name} is empty.")
            continue

        logger.info(f"Executing filter topology on device: {name}")
        all_cleaned_dfs.append(
            _process_single_cell(
                df=df,
                name=name,
                cell_id=cell_id,
                v_span_thresh=50,
                freeze_thresh=15,
                i_span_thresh=0.0001,
                spike_tol=2,
            )
        )

        del df
        gc.collect()

    if not all_cleaned_dfs:
        return None

    logger.info("Consolidating processed device chunks...")
    consolidated = pd.concat(all_cleaned_dfs, axis=0, ignore_index=True)
    del all_cleaned_dfs
    gc.collect()
    return consolidated


def _log_quality_control_report(df: pd.DataFrame) -> None:
    """Aggregate and log per-device yield and global rejection frequencies."""
    logger.info("Aggregating global diagnostics report...")
    summary = df.groupby(["cell_name", "curve"]).first()

    try:
        val_counts = summary.groupby("cell_name")["is_curve_valid"].agg(["count", "sum"])
        val_counts.columns = ["Total Cycles Detected", "Valid Cycles Retained"]

        discard_cols = [
            "is_low_voltage", "is_low_current", "is_night_time",
            "is_curve_frozen", "is_corrupted", "is_spike_error",
            "is_asymmetric", "is_mean_mismatch",
        ]

        logger.info(
            f"\n=== QUALITY CONTROL METRICS ===\n"
            f"Aggregate cycles evaluated globally: {len(summary)}\n\n"
            f"Cycle Yield Distribution by Device:\n{val_counts}\n\n"
            f"Global Rejection Frequencies (Applied per cycle):\n"
            f"{summary[discard_cols].sum().to_string()}"
        )
    except Exception as e:
        logger.warning(f"Diagnostics aggregation failed: {e}")

    logger.info("\n--- PIPELINE RETENTION AUDIT ---")
    logger.info(f"Raw Vector Dimensions (Rows):\n{df.groupby('is_reverse')['id_curve'].count().to_string()}")
    logger.info(f"Gross Cycle Count:\n{df.groupby('is_reverse')['id_curve'].nunique().to_string()}")
    logger.info(
        f"Net Valid Cycle Count (Post-Filter):\n"
        f"{df[df['is_curve_valid'] == 1].groupby('is_reverse')['id_curve'].nunique().to_string()}"
    )


def main() -> None:
    FILE_JV_FILTERED.parent.mkdir(parents=True, exist_ok=True)
    FILE_FILTERING_META.parent.mkdir(parents=True, exist_ok=True)

    if not DIR_PROCESSED.exists():
        logger.error(f"Target directory missing: {DIR_PROCESSED}. Verify upstream execution.")
        raise FileNotFoundError(f"Directory {DIR_PROCESSED} not found.")

    device_dirs = sorted(d.name for d in DIR_PROCESSED.iterdir() if d.is_dir())
    logger.info(f"Discovered processed datasets for localized devices: {device_dirs}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        df_filtered = _load_and_filter_devices(device_dirs)

        if df_filtered is None:
            logger.error("Data ingestion failure: No valid Parquet artifacts were loaded into memory.")
            return

        df_filtered["id_curve"] = df_filtered.groupby(["cell_name", "curve"]).ngroup()

        _log_quality_control_report(df_filtered)
        analyze_curve_timings(df_filtered)

        # --- QUALITY ASSURANCE & METADATA EXPORT ---
        logger.info("\n--- QUALITY ASSURANCE & DATASET METADATA EXPORT ---")
        valid_only_df = df_filtered[df_filtered["is_curve_valid"] == 1]
        valid_counts_dict = valid_only_df.groupby("cell_name")["id_curve"].nunique().to_dict()

        with open(FILE_FILTERING_META, "w") as f:
            json.dump(valid_counts_dict, f, indent=4)
        logger.info(f"Dataset structural metadata successfully serialized to: {FILE_FILTERING_META}")

        # --- PHYSICS SCALAR EXTRACTION & MEMORY-SAFE MERGE ---
        logger.info("Initiating physics parameter extraction for valid curves...")
        df_physics = extract_all_physics_parameters(valid_only_df)

        if df_physics.empty:
            logger.warning("No physics parameters were extracted. Skipping merge step.")
        else:
            float_cols = df_physics.select_dtypes(include=["float64", "float"]).columns
            df_physics[float_cols] = df_physics[float_cols].astype("float32")
            int_cols = df_physics.select_dtypes(include=["int64", "int"]).columns
            df_physics[int_cols] = df_physics[int_cols].astype("int32")

            del valid_only_df
            gc.collect()

            logger.info("Merging extracted physics parameters into the main dataset...")
            df_filtered = df_filtered.merge(df_physics, on="id_curve", how="left")
            del df_physics
            gc.collect()

        # --- SERIALIZE ---
        logger.info(f"Serializing filtered dataset artifact to {FILE_JV_FILTERED}...")
        df_filtered.to_parquet(
            FILE_JV_FILTERED, engine="pyarrow", compression="snappy", index=False
        )
        logger.info("Data engineering pipeline successfully terminated. Artifact serialized.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Execution aborted due to an unhandled exception.")
        raise
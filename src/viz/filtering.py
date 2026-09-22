"""
Module: src/viz_filtering.py
Description: Automated generation of analytical diagnostic plots for pre- and post-filtering
data audits. Implements memory-efficient early aggregation to prevent Out-Of-Memory (OOM)
failures when processing high-density raw telemetry.
"""

import gc
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import (
    OPERATIONAL_HOUR_END,
    OPERATIONAL_HOUR_START,
    DIR_PROCESSED,
    FILE_JV_FILTERED,
    DEPLOYMENT_TIMEZONE,
    FIGURES_FILTERING_DIR, 
)

OUTPUT_DIR = FIGURES_FILTERING_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Professional MLOps logging configuration.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("Viz-Filtering")
sns.set_theme(style="whitegrid")

def plot_voltage_span_distribution(spans_df: pd.DataFrame, stage_name: str, filename: str) -> None:
    """
    Renders the distribution of J-V sweep voltage spans (ΔV) to audit
    hardware freezes and incomplete scan cycles.
    """
    logger.info(f"Rendering voltage span distribution for stage: {stage_name}...")
    visual_bins = 16

    plt.figure(figsize=(12, 7))
    device_names = spans_df["cell_name"].unique()
    plotted_count = 0

    for name in device_names:
        cell_data = spans_df.loc[spans_df["cell_name"] == name, "v_span_mV"]
        if isinstance(cell_data, pd.Series) and not cell_data.empty:
            sns.histplot(
                data=np.asarray(cell_data, dtype=float),
                bins=visual_bins,
                kde=True,
                alpha=0.3,
                label=name,
            )
            plotted_count += 1

    if plotted_count == 0:
        logger.warning(f"No valid data to plot for {stage_name}.")
        plt.close()
        return

    plt.xlabel(r"$\Delta V = V_{\mathrm{max}} - V_{\mathrm{min}}$ (mV)", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.title(
        f"Distribution of Voltage Spans per Cell ({stage_name})",
        fontsize=14,
        fontweight="bold",
    )
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = OUTPUT_DIR / filename
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Plot successfully saved to: {output_path.name}")
    plt.close()


def _load_pre_filter_spans() -> pd.DataFrame:
    """
    Scan each device's raw J-V parquet, identify scan cycles, restrict to the
    local-time daylight operational window, and aggregate to per-curve voltage
    spans. Memory-safe: each device is loaded, aggregated, and released before
    moving to the next one.
    """
    device_dirs = sorted(d.name for d in DIR_PROCESSED.iterdir() if d.is_dir())
    logger.info(f"Scanning processed devices for Pre-Filter visualization: {device_dirs}")

    span_frames: list[pd.DataFrame] = []

    for name in device_dirs:
        parquet_path = DIR_PROCESSED / name / f"{name}_jv.parquet"
        if not parquet_path.exists():
            logger.warning(f"Expected Parquet artifact missing for device: {name}")
            continue

        # Load minimal columns to mitigate RAM footprint.
        df = pd.read_parquet(
            parquet_path,
            columns=["ScanDirection", "Voltage_V", "Timestamp"],
        )

        if df.empty:
            logger.warning(f"Parquet artifact for device {name} is empty.")
            continue

        # Enforce strict chronological sorting before evaluating directional shifts.
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True)
        df = df.sort_values("Timestamp")

        is_reverse_start = (df["ScanDirection"] == "Reverse") & (
            df["ScanDirection"].shift(1) != "Reverse"
        )
        is_reverse_start.iloc[0] = True
        df["curve"] = is_reverse_start.cumsum()

        # Restrict to daylight operational hours, evaluated in local time.
        local_hours = df["Timestamp"].dt.tz_convert(DEPLOYMENT_TIMEZONE).dt.hour
        df = df[local_hours.between(OPERATIONAL_HOUR_START, OPERATIONAL_HOUR_END)]

        # Early aggregation: reduces millions of rows to ~1000 per device.
        spans = df.groupby("curve")["Voltage_V"].agg(lambda x: x.max() - x.min()) * 1000

        cell_df = spans.reset_index(name="v_span_mV")
        cell_df["cell_name"] = name
        span_frames.append(cell_df)

        # Explicit garbage collection flush.
        del df
        gc.collect()

    if not span_frames:
        return pd.DataFrame()

    return pd.concat(span_frames, ignore_index=True)


def _load_post_filter_spans() -> pd.DataFrame:
    """
    Load only the valid curves from the filtered dataset and aggregate their
    per-curve voltage spans.
    """
    jv_clean = pd.read_parquet(
        FILE_JV_FILTERED,
        columns=["cell_name", "id_curve", "Voltage_V"],
        filters=[("is_curve_valid", "==", 1)],
    )

    clean_spans = (
        jv_clean.groupby(["cell_name", "id_curve"])["Voltage_V"]
        .agg(lambda x: x.max() - x.min())
        * 1000
    )
    return clean_spans.reset_index(name="v_span_mV")


def main() -> None:
    if not DIR_PROCESSED.exists():
        logger.error(f"Directory {DIR_PROCESSED} not found. Ensure upstream processing is complete.")
        raise SystemExit(1)

    # -------------------------------------------------------------------------
    # 1. PRE-FILTER VISUALIZATION (memory-optimized via early aggregation)
    # -------------------------------------------------------------------------
    pre_filter_spans = _load_pre_filter_spans()

    if not pre_filter_spans.empty:
        plot_voltage_span_distribution(
            spans_df=pre_filter_spans,
            stage_name="Pre-Filter (Raw)",
            filename="voltage_span_pre_filter.png",
        )
        del pre_filter_spans
        gc.collect()

    # -------------------------------------------------------------------------
    # 2. POST-FILTER VISUALIZATION
    # -------------------------------------------------------------------------
    if FILE_JV_FILTERED.exists():
        logger.info("Loading post-filter dataset...")
        clean_spans_df = _load_post_filter_spans()

        plot_voltage_span_distribution(
            spans_df=clean_spans_df,
            stage_name="Post-Filter (Clean)",
            filename="voltage_span_post_filter.png",
        )
    else:
        logger.warning(f"Filtered dataset not found at {FILE_JV_FILTERED}. Skipping post-filter plot.")

    logger.info("Filtering diagnostic visualizations complete.")


if __name__ == "__main__":
    main()
"""
Module: src/audit_filter_suite.py
Description: Consolidated audit suite for the J-V filter (module 02).

Runs a sequence of independent audits against the filtered dataset and
prints a single ordered report. Each audit is self-contained and can be
called individually.

Audits
------
1. Physical truth agreement   — confusion matrix vs. physics labels
2. False positives by rule    — which rule rejects valid curves
3. False negatives by rule    — which rule accepts broken curves
4. Threshold sweep            — SPIKE_MAX_POINTS trade-off curve
5. Filter-vs-clustering       — does module 03 need the filter upstream?

Physical truth definition
-------------------------
A curve is labelled with one of four physical states, based only on the
raw J-V sweep (V, I), not on any filter column:

    accept       : Voc > 0.5 V AND Jsc > 2 mA/cm² (functioning cell)
    reject_bad   : Voc < 0.4 V OR  Jsc < 1 mA/cm² (broken measurement)
    reject_night : local hour outside [6, 22]
    gray         : everything else (ambiguous, excluded from rates)

These thresholds are conventional, not derived from data. They are the
current best-effort target for the audit; the audit's purpose is to expose
disagreement, not to certify correctness.

Usage
-----
    python -m src.audit_filter_suite              # run all audits
    python -m src.audit_filter_suite --audit 1    # run only audit 1
    python -m src.audit_filter_suite --list       # list available audits
    python -m src.audit_filter_suite --rebuild-truth
                                                  # force rebuild of the
                                                  # cached physical truth table
"""

from __future__ import annotations

import argparse
import gc
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import FILE_JV_FILTERED, RANDOM_STATE


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
CELL_AREA_CM2 = 0.64
TZ = "Europe/Madrid"
HOUR_START = 6
HOUR_END = 22

VOC_ACCEPT_MIN = 0.5
JSC_ACCEPT_MIN = 2.0
VOC_REJECT_MAX = 0.4
JSC_REJECT_MAX = 1.0

AUDIT_TABLE_PATH = "data/filtered/outdoor/02_filtering_audit.parquet"
TRUTH_CACHE_PATH = "data/filtered/outdoor/02_physical_truth.parquet"
BASELINE_PATH = "/tmp/02_baseline.parquet"

SPIKE_K_SWEEP = (2, 3, 4, 5, 6, 8, 10, 15, 20, 30)


# ===========================================================================
# SHARED UTILITIES
# ===========================================================================
def raw_voc_jsc(g: pd.DataFrame) -> pd.Series:
    """
    Extract raw Voc and Jsc from a single J-V sweep, without using any
    filter-derived column. Shared by every audit that needs physical truth.

    Compatible with pandas < 2.2 (no include_groups argument).
    """
    V = g["Voltage_V"].to_numpy(dtype=float)
    I = g["Current_A"].to_numpy(dtype=float)
    if len(V) < 5:
        return pd.Series({"voc": np.nan, "jsc": np.nan})

    idx = np.argsort(V)
    V, I = V[idx], I[idx]

    # Voc via linear interpolation at the first I=0 crossing with V > 0.1
    sign = np.sign(I)
    ch = np.where(np.diff(sign) != 0)[0]
    ch = ch[V[ch] > 0.1]
    if len(ch):
        i = ch[0]
        v0, v1 = V[i], V[i + 1]
        i0, i1 = I[i], I[i + 1]
        voc = float(v0 - i0 * (v1 - v0) / (i1 - i0)) if i1 != i0 else float(v0)
    else:
        voc = float(V[np.argmin(np.abs(I))])

    # Jsc = current at V = 0
    jsc_A = (
        float(np.interp(0.0, V, I))
        if V.min() <= 0 <= V.max()
        else float(I[np.argmin(np.abs(V))])
    )
    jsc = jsc_A * 1000.0 / CELL_AREA_CM2

    return pd.Series({"voc": voc, "jsc": jsc})


def load_filtered(columns: list[str]) -> pd.DataFrame:
    """Read the filtered dataset with only the requested columns."""
    return pd.read_parquet(FILE_JV_FILTERED, columns=columns)


def build_physical_truth(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach a physical truth label to every (cell_name, curve) in df.
    df must contain: cell_name, curve, Timestamp, Voltage_V, Current_A.
    """
    df = df.copy()
    df["hour_local"] = df["Timestamp"].dt.tz_convert(TZ).dt.hour

    # Only pass the columns raw_voc_jsc needs. Avoids pandas version issues
    # with include_groups and keeps the apply payload small.
    params = (
        df[["cell_name", "curve", "Voltage_V", "Current_A"]]
        .groupby(["cell_name", "curve"], sort=False)
        .apply(raw_voc_jsc)
        .reset_index()
    )
    hours = (
        df.groupby(["cell_name", "curve"])["hour_local"].first().reset_index()
    )
    params = params.merge(hours, on=["cell_name", "curve"])

    params["truth"] = np.where(
        ~params["hour_local"].between(HOUR_START, HOUR_END),
        "reject_night",
        np.where(
            (params["voc"] > VOC_ACCEPT_MIN) & (params["jsc"] > JSC_ACCEPT_MIN),
            "accept",
            np.where(
                (params["voc"] < VOC_REJECT_MAX) | (params["jsc"] < JSC_REJECT_MAX),
                "reject_bad",
                "gray",
            ),
        ),
    )
    return params[["cell_name", "curve", "truth", "voc", "jsc"]]


def _header(title: str) -> None:
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


# ===========================================================================
# AUDIT 1 — Physical truth agreement
# ===========================================================================
def audit_physical_truth(merged: pd.DataFrame | None = None) -> dict:
    """Confusion matrix between the filter's is_curve_valid and the physical truth."""
    _header("AUDIT 1 — Physical truth agreement")

    if merged is None:
        print("Rebuilding truth + audit join...")
        df = load_filtered(
            ["cell_name", "curve", "Timestamp", "Voltage_V", "Current_A"]
        )
        truth = build_physical_truth(df)
        del df
        gc.collect()
        audit = pd.read_parquet(AUDIT_TABLE_PATH)
        merged = audit.merge(truth, on=["cell_name", "curve"], how="left")
        del audit
        gc.collect()

    print("\nTruth distribution:")
    print(merged["truth"].value_counts().to_string())

    ct = pd.crosstab(merged["truth"], merged["is_curve_valid"])
    print("\nConfusion matrix (truth × filter):")
    print(ct.to_string())

    det = merged[merged["truth"] != "gray"].copy()
    accept_total = (det["truth"] == "accept").sum()
    reject_total = det["truth"].isin(["reject_bad", "reject_night"]).sum()

    tp_accept = ((det["truth"] == "accept") & (det["is_curve_valid"] == 1)).sum()
    tp_reject = (
        (det["truth"].isin(["reject_bad", "reject_night"]))
        & (det["is_curve_valid"] == 0)
    ).sum()

    fn_rate = 1 - tp_accept / max(accept_total, 1)
    fp_rate = 1 - tp_reject / max(reject_total, 1)
    agreement = (tp_accept + tp_reject) / len(det)

    print(f"\nMetrics on deterministic subset (n={len(det):,}):")
    print(f"  Agreement            : {agreement:.2%}")
    print(f"  FN rate (valid lost) : {fn_rate:.2%}   "
          f"({accept_total - tp_accept:,} of {accept_total:,})")
    print(f"  FP rate (broken kept): {fp_rate:.2%}   "
          f"({reject_total - tp_reject:,} of {reject_total:,})")

    return {
        "agreement": agreement,
        "fn_rate": fn_rate,
        "fp_rate": fp_rate,
        "merged": merged,
    }


# ===========================================================================
# AUDIT 2 — False positives by rule
# ===========================================================================
def audit_fp_by_rule(merged: pd.DataFrame | None = None) -> dict:
    """Which rejection rule is responsible for false positives?"""
    _header("AUDIT 2 — False positives by rejection rule")

    if merged is None:
        print("Rebuilding truth + audit join...")
        df = load_filtered(["cell_name", "curve", "Timestamp", "Voltage_V", "Current_A"])
        truth = build_physical_truth(df)
        del df
        gc.collect()
        audit = pd.read_parquet(AUDIT_TABLE_PATH)
        merged = audit.merge(truth, on=["cell_name", "curve"], how="left")
        del audit
        gc.collect()

    fp = merged[(merged["truth"] == "accept") & (merged["is_curve_valid"] == 0)].copy()
    print(f"\nTotal FP (truth=accept, filter=reject): {len(fp):,}")

    if len(fp) == 0:
        print("No FP to analyze.")
        return {"fp": fp, "rules": {}}

    rules = ["rej_night", "rej_unphysical", "rej_vspan_low", "rej_snr_low", "rej_spike"]
    rule_counts = {}
    print("\nRule fired (a curve may fire several):")
    for r in rules:
        n = int(fp[r].sum())
        rule_counts[r] = n
        print(f"  {r:20s} {n:>6,}  ({n / len(fp):>6.2%})")

    fp["_rule_combo"] = fp[rules].apply(
        lambda row: "+".join([r.replace("rej_", "") for r in rules if row[r]]) or "NONE",
        axis=1,
    )
    print("\nTop rule combinations:")
    print(fp["_rule_combo"].value_counts().head(10).to_string())

    print("\nFP per cell:")
    print(fp.groupby("cell_name").size().sort_values(ascending=False).to_string())

    only_spike = fp[
        fp["rej_spike"]
        & ~fp["rej_night"]
        & ~fp["rej_unphysical"]
        & ~fp["rej_vspan_low"]
        & ~fp["rej_snr_low"]
    ]
    if len(only_spike):
        print(f"\nFP rejected ONLY by spike: {len(only_spike):,}")
        print("spike_count distribution:")
        print(only_spike["spike_count"].describe().round(2).to_string())
        for k in (3, 4, 5):
            print(f"  spike_count == {k}: {(only_spike['spike_count'] == k).sum()}")
        print(f"  spike_count >  5: {(only_spike['spike_count'] > 5).sum()}")

    return {"fp": fp, "rules": rule_counts}


# ===========================================================================
# AUDIT 3 — False negatives by rule
# ===========================================================================
def audit_fn_by_rule(merged: pd.DataFrame | None = None) -> dict:
    """Which feature is shared by broken curves that the filter accepts?"""
    _header("AUDIT 3 — False negatives by feature distribution")

    if merged is None:
        df = load_filtered(["cell_name", "curve", "Timestamp", "Voltage_V", "Current_A"])
        truth = build_physical_truth(df)
        del df
        gc.collect()
        audit = pd.read_parquet(AUDIT_TABLE_PATH)
        merged = audit.merge(truth, on=["cell_name", "curve"], how="left")
        del audit
        gc.collect()

    fn = merged[(merged["truth"] == "reject_bad") & (merged["is_curve_valid"] == 1)].copy()
    print(f"\nTotal FN (truth=reject_bad, filter=accept): {len(fn):,}")

    if len(fn) == 0:
        print("No FN to analyze.")
        return {"fn": fn}

    print("\nFeature distributions among accepted broken curves:")
    cols = ["v_span_ratio", "snr_i", "spike_count", "hysteresis_index", "voc", "jsc"]
    cols = [c for c in cols if c in fn.columns]
    print(fn[cols].describe().round(4).to_string())

    print("\nFN per cell:")
    print(fn.groupby("cell_name").size().sort_values(ascending=False).to_string())

    print("\nFN rate per cell (accepted broken / total broken):")
    per_cell = (
        merged[merged["truth"] == "reject_bad"]
        .groupby("cell_name")["is_curve_valid"]
        .agg(total="count", accepted="sum")
    )
    per_cell["fn_rate"] = per_cell["accepted"] / per_cell["total"]
    print(per_cell.round(3).to_string())

    return {"fn": fn, "per_cell": per_cell}


# ===========================================================================
# AUDIT 4 — SPIKE_MAX_POINTS sweep
# ===========================================================================
def audit_threshold_sweep(merged: pd.DataFrame | None = None) -> dict:
    """Re-apply the decision layer with varying SPIKE_MAX_POINTS."""
    _header("AUDIT 4 — SPIKE_MAX_POINTS threshold sweep")

    if merged is None:
        df = load_filtered(["cell_name", "curve", "Timestamp", "Voltage_V", "Current_A"])
        truth = build_physical_truth(df)
        del df
        gc.collect()
        audit = pd.read_parquet(AUDIT_TABLE_PATH)
        merged = audit.merge(truth, on=["cell_name", "curve"], how="left")
        del audit
        gc.collect()

    SNR_MIN = 3.0
    V_SPAN_RATIO_MIN = 0.10

    n_accept_truth = int((merged["truth"] == "accept").sum())

    print(f" {'k':>3} | {'Accepted':>9} | {'AcceptKept':>10} | {'RejBadLeak':>10} | "
          f"{'Loss%':>7} | {'Leak%':>7}")
    print("-" * 76)

    rows = []
    for k in SPIKE_K_SWEEP:
        accepted = (
            ~merged["rej_night"]
            & ~merged["rej_unphysical"]
            & (merged["v_span_ratio"] >= V_SPAN_RATIO_MIN)
            & (merged["snr_i"] >= SNR_MIN)
            & (merged["spike_count"] <= k)
        )
        n_acc = int(accepted.sum())
        accept_kept = int((accepted & (merged["truth"] == "accept")).sum())
        rejbad_leak = int((accepted & (merged["truth"] == "reject_bad")).sum())
        accept_lost = n_accept_truth - accept_kept
        loss_pct = 100.0 * accept_lost / max(n_accept_truth, 1)
        leak_pct = 100.0 * rejbad_leak / max(n_acc, 1)
        rows.append((k, n_acc, accept_kept, rejbad_leak, loss_pct, leak_pct))
        print(f" {k:>3} | {n_acc:>9,} | {accept_kept:>10,} | {rejbad_leak:>10,} | "
              f"{loss_pct:>6.2f}% | {leak_pct:>6.2f}%")

    print("\nMarginal trade-off:")
    print(f" {'k':>3} | {'ΔAcceptKept':>12} | {'ΔRejBadLeak':>12} | "
          f"{'ΔLoss%':>9} | {'ΔLeak%':>9}")
    print("-" * 76)
    prev = None
    for row in rows:
        k, _, accept_kept, rejbad_leak, loss_pct, leak_pct = row
        if prev is None:
            print(f" {k:>3} | {'—':>12} | {'—':>12} | {'—':>9} | {'—':>9}")
        else:
            print(f" {k:>3} | {accept_kept - prev[2]:>+12,} | "
                  f"{rejbad_leak - prev[3]:>+12,} | "
                  f"{loss_pct - prev[4]:>+8.2f}% | {leak_pct - prev[5]:>+8.2f}%")
        prev = row

    return {"sweep": rows}


# ===========================================================================
# AUDIT 5 — Filter vs clustering necessity
# ===========================================================================
def audit_filter_vs_clustering(merged: pd.DataFrame | None = None) -> None:
    """Delegate to the standalone module (kept lightweight here)."""
    _header("AUDIT 5 — Filter vs clustering necessity")
    print("Running src.audit_filter_vs_clustering as a subprocess...")
    print("(see that module's docstring for the design)")
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "src.audits.filter_vs_clustering"],
        capture_output=True, text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)


# ===========================================================================
# ORCHESTRATOR
# ===========================================================================
AUDITS = {
    1: ("Physical truth agreement", audit_physical_truth),
    2: ("False positives by rule", audit_fp_by_rule),
    3: ("False negatives by feature", audit_fn_by_rule),
    4: ("Threshold sweep", audit_threshold_sweep),
    5: ("Filter vs clustering", audit_filter_vs_clustering),
}


def _load_or_build_merged(
    cache_path: str = TRUTH_CACHE_PATH,
) -> pd.DataFrame:
    """
    Return a DataFrame with one row per curve, joining the audit table with
    the physical truth label. Cached to disk so consecutive runs of the suite
    do not recompute the truth (which takes ~5-8 min on 28M rows).
    Delete the cache file or pass --rebuild-truth to force a rebuild.
    """
    cache = Path(cache_path)
    if cache.exists():
        print(f"Loading cached physical truth from {cache}...")
        merged = pd.read_parquet(cache)
        print(f"  → {len(merged):,} curves")
        return merged

    print("Building physical truth (first run only, ~5-8 min)...")
    df = load_filtered(
        ["cell_name", "curve", "Timestamp", "Voltage_V", "Current_A"]
    )
    truth = build_physical_truth(df)
    del df
    gc.collect()

    print("Joining with audit table...")
    audit = pd.read_parquet(AUDIT_TABLE_PATH)
    if "neg_ratio" not in audit.columns:
        audit["neg_ratio"] = 0.0
    merged = audit.merge(truth, on=["cell_name", "curve"], how="left")
    del audit
    gc.collect()

    cache.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(cache, index=False, compression="snappy")
    print(f"  → cached to {cache}")
    return merged


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    parser = argparse.ArgumentParser(description="J-V filter audit suite")
    parser.add_argument(
        "--audit", type=int, default=None,
        help="Run only this audit number (default: all)",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available audits and exit",
    )
    parser.add_argument(
        "--rebuild-truth", action="store_true",
        help="Force rebuild of the cached physical truth table",
    )
    args = parser.parse_args()

    if args.list:
        for i, (name, _) in AUDITS.items():
            print(f"  {i}: {name}")
        return

    cache = Path(TRUTH_CACHE_PATH)
    if args.rebuild_truth and cache.exists():
        print(f"Removing cache {cache}...")
        cache.unlink()

    merged_cache: pd.DataFrame | None = None

    to_run = [args.audit] if args.audit else list(AUDITS.keys())
    for i in to_run:
        if i not in AUDITS:
            print(f"Unknown audit: {i}")
            continue
        name, fn = AUDITS[i]
        try:
            if i == 5:
                # Audit 5 delegates to a subprocess; no merged needed.
                fn()
            else:
                # All other audits benefit from the shared cached join.
                if merged_cache is None:
                    merged_cache = _load_or_build_merged()
                fn(merged=merged_cache)
        except Exception as e:
            print(f"\nAudit {i} failed: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
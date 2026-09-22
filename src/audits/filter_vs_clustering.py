"""
Module: src/audit_filter_vs_clustering.py
Description: Experiment — does the clustering module (03) already separate
curves that the strict filter rejects?

Hypothesis under test
---------------------
If the clustering pipeline (PCA + IsolationForest + K-Medoids + thermodynamic
medoid validation) already isolates the curves that the strict filter rejects,
then the strict filter is redundant and can be replaced by a minimal filter
(solo invariantes físicos).

Design
------
1. Rebuild the physical truth label per curve (accept / reject_bad / gray / night)
2. Sample ~12k curves stratified by (cell, truth)
3. Normalize each curve the same way module 03 does
4. Train PCA + IsolationForest + K-Medoids on the FULL sample (no pre-filtering)
5. Analyze cluster composition: do clusters contain mostly accept or mostly reject_bad?
6. Compute ARI between cluster labels and truth labels
7. Compare with: clustering on STRICT-FILTERED subset only

Metrics reported
----------------
- Cluster purity (max fraction of accept or reject_bad in each cluster)
- Adjusted Rand Index (ARI) between clusters and truth
- Fraction of reject_bad curves that land in "bad" clusters
- Silhouette score on truth labels
- Comparison: strict-filter subset vs full sample

Memory strategy
---------------
Load V/I only for the sampled curves. Peak RAM < 3 GB.
"""

from __future__ import annotations

import gc
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import KMeans

from src.config import FILE_JV_FILTERED, RANDOM_STATE


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SAMPLES_PER_GROUP = 500        # curves per (cell × truth_label)
N_POINTS = 50                  # points per normalized sweep (matches module 03)
KMEDOIDS_TRAIN_SIZE = 5000     # subsample for K-Medoids training
N_CLUSTERS = 5                 # from module 03 default (or auto-selected)
ISO_CONTAMINATION = 0.03

CELL_AREA_CM2 = 0.64
TZ = "Europe/Madrid"

# Physical truth thresholds (must match audit_physical_truth.py)
VOC_ACCEPT_MIN = 0.5
JSC_ACCEPT_MIN = 2.0
VOC_REJECT_MAX = 0.4
JSC_REJECT_MAX = 1.0
HOUR_START = 6
HOUR_END = 22


# ---------------------------------------------------------------------------
# 1. Physical truth (recomputed; must match audit_physical_truth.py)
# ---------------------------------------------------------------------------
def raw_voc_jsc(g: pd.DataFrame) -> pd.Series:
    V = g["Voltage_V"].to_numpy(dtype=float)
    I = g["Current_A"].to_numpy(dtype=float)
    if len(V) < 5:
        return pd.Series({"voc": np.nan, "jsc": np.nan})

    idx = np.argsort(V)
    V, I = V[idx], I[idx]

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

    jsc_A = (
        float(np.interp(0.0, V, I))
        if V.min() <= 0 <= V.max()
        else float(I[np.argmin(np.abs(V))])
    )
    jsc = jsc_A * 1000.0 / CELL_AREA_CM2
    return pd.Series({"voc": voc, "jsc": jsc})


def build_truth(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute physical truth for each (cell, curve)."""
    df = df.copy()
    df["hour_local"] = df["Timestamp"].dt.tz_convert(TZ).dt.hour

    params = (
        df.groupby(["cell_name", "curve"], sort=False)
        .apply(raw_voc_jsc, include_groups=False)
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
    return params[["cell_name", "curve", "truth"]]


# ---------------------------------------------------------------------------
# 2. Curve normalization (copy of module 03 logic; kept inline for
#    self-containment)
# ---------------------------------------------------------------------------
def normalize_curve(jv: pd.DataFrame, n_points: int = N_POINTS) -> np.ndarray:
    V = jv["Voltage_V"].to_numpy(dtype=float)
    J = jv["Current_A"].to_numpy(dtype=float)

    if len(V) < 10:
        return np.full(n_points * 2, np.nan)

    v_min, v_max = V.min(), V.max()
    j_min, j_max = J.min(), J.max()
    v_range, j_range = v_max - v_min, j_max - j_min

    if v_range == 0 or j_range == 0:
        return np.full(n_points * 2, np.nan)

    idx_turn = int(np.argmax(np.abs(V - V[0])))
    if idx_turn < 2 or idx_turn > len(V) - 3:
        return np.full(n_points * 2, np.nan)

    V_rev, J_rev = V[: idx_turn + 1], J[: idx_turn + 1]
    V_fwd, J_fwd = V[idx_turn:], J[idx_turn:]

    def interpolate_sweep(v_sweep, j_sweep):
        v_norm = (v_sweep - v_min) / v_range
        j_norm = (j_sweep - j_min) / j_range
        v_unq, unq_inv = np.unique(v_norm, return_inverse=True)
        j_unq = np.bincount(unq_inv, weights=j_norm) / np.bincount(unq_inv)
        if len(v_unq) < 2:
            return np.full(n_points, np.nan)
        return np.interp(
            np.linspace(0, 1, n_points), v_unq, j_unq,
            left=j_unq[0], right=j_unq[-1],
        )

    try:
        return np.concatenate([
            interpolate_sweep(V_rev, J_rev),
            interpolate_sweep(V_fwd, J_fwd),
        ])
    except Exception:
        return np.full(n_points * 2, np.nan)


# ---------------------------------------------------------------------------
# 3. Cluster analysis helpers
# ---------------------------------------------------------------------------
def cluster_composition(
    labels: np.ndarray, truth: pd.Series, label_name: str = "Cluster"
) -> pd.DataFrame:
    """Per-cluster composition (accept / reject_bad / gray fractions)."""
    df = pd.DataFrame({"cluster": labels, "truth": truth.values})
    df = df[df["cluster"] >= 0]  # exclude noise (-1)

    comp = pd.crosstab(df["cluster"], df["truth"], normalize="index")
    counts = df.groupby("cluster").size().rename("n")
    comp = comp.join(counts)

    # Purity = max fraction of accept or reject_bad in each cluster
    accept_col = comp.get("accept", pd.Series(0, index=comp.index))
    reject_col = comp.get("reject_bad", pd.Series(0, index=comp.index))
    comp["purity"] = np.maximum(accept_col, reject_col)
    comp["dominant"] = np.where(accept_col > reject_col, "accept", "reject_bad")

    print(f"\n=== {label_name} composition ===")
    print(comp.round(3).to_string())
    return comp


def run_pipeline(
    X: np.ndarray,
    truth: pd.Series,
    tag: str,
    n_clusters: int = N_CLUSTERS,
    contamination: float = ISO_CONTAMINATION,
) -> dict:
    """
    Run PCA → IsolationForest → K-Medoids on X and return labels + metrics.
    """
    print(f"\n{'=' * 70}")
    print(f" {tag}")
    print(f" n_curves = {len(X):,}")
    print(f"{'=' * 70}")

    # 1. PCA
    pca = PCA(n_components=0.90, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X)
    print(f" PCA: {X_pca.shape[1]} components (90% variance)")

    # 2. IsolationForest
    iso = IsolationForest(
        contamination=contamination, random_state=RANDOM_STATE, n_estimators=200
    )
    iso_labels = iso.fit_predict(X_pca)
    inlier_mask = iso_labels == 1
    print(f" IsolationForest: {inlier_mask.sum():,} inliers "
          f"({100 * inlier_mask.mean():.1f}%)")

    # 3. K-Medoids on a training subsample of inliers
    inlier_idx = np.where(inlier_mask)[0]
    if len(inlier_idx) > KMEDOIDS_TRAIN_SIZE:
        rng = np.random.RandomState(RANDOM_STATE)
        train_idx = rng.choice(inlier_idx, KMEDOIDS_TRAIN_SIZE, replace=False)
    else:
        train_idx = inlier_idx

    kmedoids = KMedoids(
        n_clusters=n_clusters, random_state=RANDOM_STATE, method="pam"
    )
    kmedoids.fit(X_pca[train_idx])
    print(f" K-Medoids: trained on {len(train_idx):,} curves, "
          f"k = {n_clusters}")

    # 4. Predict all
    cluster_labels = np.full(len(X), -1, dtype=int)
    cluster_labels[inlier_mask] = kmedoids.predict(X_pca[inlier_mask])

    # 5. Composition
    comp = cluster_composition(cluster_labels, truth, tag)

    # 6. ARI vs truth (only on curves that got a cluster label)
    valid_mask = cluster_labels >= 0
    truth_bin = truth.values[valid_mask]
    # Map truth to binary: accept=0, reject_bad=1, exclude gray/night
    keep = np.isin(truth_bin, ["accept", "reject_bad"])
    if keep.sum() > 10:
        truth_bin_clean = (truth_bin[keep] == "reject_bad").astype(int)
        ari = adjusted_rand_score(truth_bin_clean, cluster_labels[valid_mask][keep])
    else:
        ari = np.nan

    # 7. Silhouette on truth binary (only on labeled, non-gray points)
    if keep.sum() > 10:
        sil = silhouette_score(
            X_pca[valid_mask][keep],
            truth_bin_clean,
            sample_size=min(5000, keep.sum()),
            random_state=RANDOM_STATE,
        )
    else:
        sil = np.nan

    print(f"\n ARI(clusters, truth) : {ari:.4f}")
    print(f" Silhouette(truth)    : {sil:.4f}")
    print(f" 'reject_bad' in dominant cluster(s):")

    # Count: fraction of reject_bad that landed in a "reject_bad"-dominant cluster
    dom = comp["dominant"]
    reject_dominant_clusters = dom[dom == "reject_bad"].index.tolist()
    accept_dominant_clusters = dom[dom == "accept"].index.tolist()

    rb_mask = truth.values == "reject_bad"
    rb_in_reject_cluster = np.isin(cluster_labels, reject_dominant_clusters) & rb_mask
    accept_mask = truth.values == "accept"
    accept_in_accept_cluster = np.isin(cluster_labels, accept_dominant_clusters) & accept_mask

    print(f"   reject_bad in reject-dominant clusters : "
          f"{rb_in_reject_cluster.sum():,} / {rb_mask.sum():,} "
          f"({100 * rb_in_reject_cluster.sum() / max(rb_mask.sum(), 1):.1f}%)")
    print(f"   accept     in accept-dominant clusters : "
          f"{accept_in_accept_cluster.sum():,} / {accept_mask.sum():,} "
          f"({100 * accept_in_accept_cluster.sum() / max(accept_mask.sum(), 1):.1f}%)")

    return {
        "cluster_labels": cluster_labels,
        "comp": comp,
        "ari": ari,
        "sil": sil,
        "reject_dominant_clusters": reject_dominant_clusters,
        "accept_dominant_clusters": accept_dominant_clusters,
    }


# ---------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------
def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # --- Load audit + truth, sample keys ---
    print("Loading audit table and recomputing physical truth...")
    audit = pd.read_parquet(
        "data/filtered/outdoor/02_filtering_audit.parquet",
        columns=["cell_name", "curve", "is_curve_valid"],
    )

    # Need Timestamp + V/I to compute truth. Load from filtered parquet.
    print("Loading Timestamp for truth computation...")
    ts_df = pd.read_parquet(
        FILE_JV_FILTERED,
        columns=["cell_name", "curve", "Timestamp", "Voltage_V", "Current_A"],
    )
    truth = build_truth(ts_df)
    del ts_df
    gc.collect()

    meta = audit.merge(truth, on=["cell_name", "curve"], how="left")
    print(f"\nTruth distribution (full dataset):")
    print(meta["truth"].value_counts().to_string())

    # --- Sample stratified by (cell, truth) ---
    print(f"\nSampling {SAMPLES_PER_GROUP} curves per (cell, truth)...")
    rng = np.random.RandomState(RANDOM_STATE)
    sampled_keys: set[str] = set()
    for (cell, tr), g in meta.groupby(["cell_name", "truth"]):
        if tr == "reject_night":
            continue  # skip night, no useful shape
        n = min(len(g), SAMPLES_PER_GROUP)
        keys = g.sample(n, random_state=rng)[["cell_name", "curve"]].values
        for k in keys:
            sampled_keys.add(f"{k[0]}::{int(k[1])}")

    print(f"  → {len(sampled_keys):,} curves sampled")

    # --- Load V/I only for sampled curves ---
    print("Loading V/I for sampled curves...")
    full = pd.read_parquet(
        FILE_JV_FILTERED,
        columns=["cell_name", "curve", "Voltage_V", "Current_A"],
    )
    key_series = full["cell_name"].astype(str) + "::" + full["curve"].astype(str)
    sub = full[key_series.isin(sampled_keys)].copy()
    del full, key_series
    gc.collect()
    print(f"  → {len(sub):,} rows loaded ({sub.groupby(['cell_name','curve']).ngroups} curves)")

    # --- Normalize ---
    print("\nNormalizing curves...")
    normalized = []
    keys = []
    for (cell, curve), g in sub.groupby(["cell_name", "curve"], sort=False):
        vec = normalize_curve(g, n_points=N_POINTS)
        if not np.isnan(vec).any():
            normalized.append(vec)
            keys.append((cell, int(curve)))

    X = np.stack(normalized)
    key_df = pd.DataFrame(keys, columns=["cell_name", "curve"])
    key_df = key_df.merge(meta, on=["cell_name", "curve"], how="left")
    print(f"  → {len(X):,} valid curves after normalization")
    print(f"  → Truth distribution in normalized sample:")
    print(key_df["truth"].value_counts().to_string())

    truth_series = key_df["truth"]

    # --- EXPERIMENT 1: Cluster on FULL sample (no pre-filter) ---
    result_full = run_pipeline(X, truth_series, "FULL SAMPLE (minimal filter, no pre-rejection)")

    # --- EXPERIMENT 2: Cluster only on strict-filtered subset ---
    strict_mask = key_df["is_curve_valid"] == 1
    if strict_mask.sum() > 100:
        X_strict = X[strict_mask.values]
        truth_strict = truth_series[strict_mask].reset_index(drop=True)
        result_strict = run_pipeline(
            X_strict, truth_strict, "STRICT-FILTERED SUBSET (baseline pipeline)"
        )
    else:
        result_strict = None

    # --- Summary ---
    print("\n" + "=" * 70)
    print(" SUMMARY")
    print("=" * 70)
    print(f" Full sample    : ARI={result_full['ari']:.4f}  Sil={result_full['sil']:.4f}")
    if result_strict is not None:
        print(f" Strict subset  : ARI={result_strict['ari']:.4f}  Sil={result_strict['sil']:.4f}")
    else:
        print(" Strict subset  : not enough curves")

    print("\n Interpretation:")
    print("  ARI > 0.5 → clustering separates accept vs reject_bad on its own")
    print("              → strict filter is redundant")
    print("  ARI < 0.2 → clustering cannot distinguish; strict filter adds value")
    print("  Sil > 0.2 → truth labels form well-separated groups in PCA space")
    print("  Sil < 0.1 → truth labels overlap in PCA space (fuzzy boundary)")


if __name__ == "__main__":
    main()
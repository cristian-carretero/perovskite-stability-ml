"""
Module: src/audit_if_diagnosis.py
Description: Diagnóstico del IsolationForest de module 03.

Objetivo: ¿qué son los puntos blancos que marca el IsolationForest?
¿Son células válidas con morfología poco común, o son curvas inválidas
que module 02 dejó escapar?

Nota: los módulos del pipeline están numerados (01_, 02_, 03_) y no pueden
importarse con `import src.03_...`. Este script usa importlib para cargar
`normalize_curve` desde `03_jv_clustering.py`.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

from src.config import FILE_JV_LABELED, RANDOM_STATE


# ---------------------------------------------------------------------------
# Import the clustering module's normalize_curve via importlib
# ---------------------------------------------------------------------------
def _load_clustering_module():
    here = Path(__file__).parent
    spec = importlib.util.spec_from_file_location(
        "jv_clustering", here / "03_jv_clustering.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_clustering = _load_clustering_module()
normalize_curve = _clustering.normalize_curve

N_POINTS = 50          # matches module 03's default
N_SAMPLE = 30_000      # curves to sample for the diagnosis


def main() -> None:
    print("Cargando curvas clustered...")
    df = pd.read_parquet(
        FILE_JV_LABELED,
        columns=[
            "cell_name", "curve",
            "Voltage_V", "Current_A",
            "voc", "jsc",
        ],
    )

    # Merge per-curve quality scores from the module 02 audit table.
    # (03_jv_clustered.parquet does not carry these columns; they live in
    # the filtering audit and are propagated to 05_merged.)
    from src.config import DIR_FILTERED
    audit_path = DIR_FILTERED / "02_filtering_audit.parquet"
    print(f"Cargando scores de calidad desde {audit_path.name}...")
    audit = pd.read_parquet(
        audit_path,
        columns=["cell_name", "curve", "hysteresis_index", "spike_count"],
    )
    df = df.merge(audit, on=["cell_name", "curve"], how="left")

    n_matched = int(df["hysteresis_index"].notna().sum())
    n_total = len(df)
    print(f"  → {n_matched:,}/{n_total:,} puntos con score ({100*n_matched/n_total:.1f}%)")

    # Rellenar NaN por si alguna curva quedó sin score (con 0.0 defensivo)
    df["hysteresis_index"] = df["hysteresis_index"].fillna(0.0)
    df["spike_count"] = df["spike_count"].fillna(0.0)

    key_cols = ["cell_name", "curve"]
    curves = df[key_cols].drop_duplicates()
    rng = np.random.RandomState(RANDOM_STATE)
    sampled = curves.sample(min(N_SAMPLE, len(curves)), random_state=rng)
    df = df.merge(sampled, on=key_cols, how="inner")
    print(f"Curvas muestreadas: {df.groupby(key_cols).ngroups:,}")

    # ---- Normalize each curve ----
    print("Normalizando curvas...")
    normed: list[np.ndarray] = []
    meta: list[dict] = []
    for (cell, curve), g in df.groupby(key_cols, sort=False):
        vec = normalize_curve(g, n_points=N_POINTS)
        if np.isnan(vec).any():
            continue
        normed.append(vec)
        meta.append({
            "cell_name": cell,
            "curve": int(curve),
            "hysteresis_index": float(g["hysteresis_index"].iloc[0]),
            "spike_count": float(g["spike_count"].iloc[0]),
            "voc": float(g["voc"].iloc[0]),
            "jsc": float(g["jsc"].iloc[0]),
        })

    X = np.stack(normed)
    meta_df = pd.DataFrame(meta)
    print(f"Curvas normalizadas: {len(X):,}")

    # ---- PCA + IsolationForest (same conditions as module 03) ----
    pca = PCA(n_components=0.90, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X)

    iso = IsolationForest(
        contamination=0.03, random_state=RANDOM_STATE, n_estimators=200
    )
    labels = iso.fit_predict(X_pca)
    meta_df["if_label"] = labels
    meta_df["if_score"] = iso.score_samples(X_pca)

    # ---- Compare inliers vs outliers ----
    inliers = meta_df[meta_df["if_label"] == 1]
    outliers = meta_df[meta_df["if_label"] == -1]

    print("\n" + "=" * 70)
    print(" COMPARATIVA: inliers vs outliers detectados por IsolationForest")
    print("=" * 70)
    for col in ["hysteresis_index", "spike_count", "voc", "jsc"]:
        print(f"\n{col}:")
        print(
            f"  Inliers : median={inliers[col].median():>8.3f}  "
            f"p10={inliers[col].quantile(0.10):>8.3f}  "
            f"p90={inliers[col].quantile(0.90):>8.3f}"
        )
        print(
            f"  Outliers: median={outliers[col].median():>8.3f}  "
            f"p10={outliers[col].quantile(0.10):>8.3f}  "
            f"p90={outliers[col].quantile(0.90):>8.3f}"
        )

    # ---- Distribution of outliers per cell ----
    print("\n" + "=" * 70)
    print(" DISTRIBUCIÓN DE OUTLIERS POR CÉLULA")
    print("=" * 70)
    dist = meta_df.groupby("cell_name").agg(
        total=("if_label", "count"),
        outliers=("if_label", lambda x: int((x == -1).sum())),
    )
    dist["outlier_pct"] = (100 * dist["outliers"] / dist["total"]).round(2)
    print(dist.sort_values("outlier_pct", ascending=False).to_string())

    # ---- Score distribution: is there a natural cutoff? ----
    print("\n" + "=" * 70)
    print(" SCORE DISTRIBUTION (¿dónde está el corte natural?)")
    print("=" * 70)
    scores = np.sort(meta_df["if_score"].values)
    for p in [0.5, 1, 2, 3, 5, 10, 20, 50]:
        idx = int(p / 100 * len(scores))
        print(f"  p{p:>4.1f}   score = {scores[idx]:>8.4f}")

    meta_df.to_parquet("/tmp/if_diagnosis.parquet", index=False)
    print("\n→ Guardado en /tmp/if_diagnosis.parquet")


if __name__ == "__main__":
    main()
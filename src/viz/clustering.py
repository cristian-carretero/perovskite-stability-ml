"""
Module: src/viz_clustering.py
Description: Advanced Machine Learning visualizations (PCA, K-Medoids).
Loads pre-trained artifacts to decouple execution from the training pipeline.
Implements extreme memory optimizations and discrete semantic colormapping
to handle structurally pruned (anomalous) clusters gracefully.
"""

import logging
import math
from pathlib import Path

import joblib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from sklearn.decomposition import PCA

from src.config import (
    PCA_TARGET_EXPLAINED_VARIANCE,
    FILE_JV_LABELED,
    FILE_CLUSTERING_ARTIFACTS,
    FIGURES_CLUSTERING_DIR,
)

OUTPUT_DIR = FIGURES_CLUSTERING_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Professional MLOps logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("Viz-Clustering")
sns.set_theme(style="whitegrid")

def plot_pca_variance_and_loadings(
    pca_full: PCA,
    pca: PCA,
    n_points: int = 50,
    filename: str = "01_pca_variance_loadings.png",
) -> None:
    """Plots the cumulative explained variance and the feature loadings for the principal components."""
    fig, (ax_var, ax_load) = plt.subplots(1, 2, figsize=(16, 6))

    # Cumulative Variance Plot
    max_comps = min(10, len(pca_full.explained_variance_ratio_))
    cum_var = np.cumsum(pca_full.explained_variance_ratio_)[: max_comps + 1]

    ax_var.plot(range(1, len(cum_var) + 1), cum_var, marker="o", lw=3, ms=8, color="teal")
    ax_var.axhline(
        y=PCA_TARGET_EXPLAINED_VARIANCE,
        color="r",
        linestyle="--",
        lw=2,
        label=f"{PCA_TARGET_EXPLAINED_VARIANCE:.0%} Explained Variance",
    )
    ax_var.set(
        xlabel="Number of Components",
        ylabel="Cumulative Explained Variance",
        title="PCA: Cumulative Explained Variance",
        xlim=(0.5, max_comps + 1.5),
    )
    ax_var.grid(True, linestyle="--", alpha=0.5)
    ax_var.legend(fontsize=12)

    # PCA Loadings Plot
    v_norm = np.linspace(0, 1, n_points)
    cmap_loadings = plt.cm.tab10

    for i in range(pca.n_components_):
        pc_rev = pca.components_[i, :n_points]
        pc_fwd = pca.components_[i, n_points:]
        c = cmap_loadings(i % 10)
        ax_load.plot(v_norm, pc_rev, "-", lw=3, color=c, label=f"PC{i+1} (Rev)")
        ax_load.plot(v_norm, pc_fwd, "--", lw=3, color=c, alpha=0.6, label=f"PC{i+1} (Fwd)")

    ax_load.axhline(0, color="black", lw=1, linestyle="--")
    ax_load.set(
        xlabel="Normalized Voltage",
        ylabel="Loading Weight",
        title=f"PCA Loadings ({pca.n_components_} Components)",
    )
    ax_load.grid(True, linestyle="--", alpha=0.3)
    ax_load.legend(fontsize=10, loc="best", ncol=(2 if pca.n_components_ > 2 else 1))

    plt.tight_layout()
    output_path = OUTPUT_DIR / filename
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Plot saved: {output_path.name}")
    plt.close()


def plot_cluster_evaluation(
    cluster_range: list,
    inertia: list,
    silhouette: list,
    n_clusters: int,
    filename: str = "02_cluster_evaluation.png",
) -> None:
    """Plots the K-selection metrics (Inertia vs. Silhouette) to visualize optimal topological clustering."""
    fig, ax_metrics = plt.subplots(figsize=(10, 6))

    color_wcss = "tab:blue"
    ax_metrics.set_xlabel("Number of Clusters (k)", fontsize=14)
    ax_metrics.set_ylabel("Inertia (WCSS)", color=color_wcss, fontsize=14)
    lns1 = ax_metrics.plot(cluster_range, inertia, marker="o", color=color_wcss, lw=3, ms=10, label="Inertia")
    ax_metrics.tick_params(axis="y", labelcolor=color_wcss)

    ax_sil = ax_metrics.twinx()
    color_sil = "tab:red"
    ax_sil.set_ylabel("Silhouette Score", color=color_sil, fontsize=14)
    lns2 = ax_sil.plot(cluster_range, silhouette, marker="s", color=color_sil, lw=3, ms=10, label="Silhouette")
    ax_sil.tick_params(axis="y", labelcolor=color_sil)
    ax_sil.set_ylim(0, 1)

    ax_metrics.axvline(x=n_clusters, color="k", linestyle="--", alpha=0.5, label=f"Chosen k={n_clusters}")

    lns = lns1 + lns2 + [ax_metrics.get_lines()[-1]]
    labs = [str(line.get_label()) for line in lns]

    ax_metrics.legend(lns, labs, loc="upper right", fontsize=12)
    ax_metrics.set_title("Optimal Clusters Validation", fontsize=16, fontweight="bold")
    ax_metrics.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    output_path = OUTPUT_DIR / filename
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Plot saved: {output_path.name}")
    plt.close()


def plot_cluster_median_curves(
    X_real: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
    n_points: int = 50,
    filename: str = "03_cluster_median_curves.png",
) -> None:
    """Renders the physical thermodynamic fingerprint (median & IQR) for each identified morphological cluster."""
    cols = min(n_clusters, 3)
    rows = math.ceil(n_clusters / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(16, 5 * rows), dpi=150)

    axes_flat = np.array(axes).flatten() if n_clusters > 1 else [axes]
    colors = plt.cm.viridis(np.linspace(0, 1, n_clusters))
    v_norm = np.linspace(0, 1, n_points)

    for i in range(n_clusters):
        ax = axes_flat[i]
        mask = labels == i
        n_samples = int(np.sum(mask))
        if n_samples == 0:
            continue

        median_c = np.median(X_real[mask], axis=0)
        p25_c = np.percentile(X_real[mask], 25, axis=0)
        p75_c = np.percentile(X_real[mask], 75, axis=0)

        # Reverse sweep
        ax.plot(v_norm, median_c[:n_points], color=colors[i], linewidth=3.5)
        ax.fill_between(v_norm, p25_c[:n_points], p75_c[:n_points], color=colors[i], alpha=0.25)

        # Forward sweep
        ax.plot(v_norm, median_c[n_points:], linestyle="--", color=colors[i], linewidth=3.5, alpha=0.9)
        ax.fill_between(v_norm, p25_c[n_points:], p75_c[n_points:], color=colors[i], alpha=0.15)

        ax.set(
            xlim=(-0.05, 1.05),
            ylim=(-0.05, 1.05),
            xlabel=r"$V_{\mathrm{norm}}$",
            ylabel=r"$J_{\mathrm{norm}}$",
        )
        ax.grid(True, linestyle=":", alpha=0.7)
        ax.text(
            0.95,
            0.95,
            f"Type {i}\n(n={n_samples})",
            transform=ax.transAxes,
            va="top",
            ha="right",
            bbox=dict(facecolor="white", alpha=0.9, edgecolor="gray"),
        )

    for j in range(n_clusters, len(axes_flat)):
        fig.delaxes(axes_flat[j])

    plt.tight_layout()
    output_path = OUTPUT_DIR / filename
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Plot saved: {output_path.name}")
    plt.close()


def plot_comparison_heatmaps(
    jv_labeled: pd.DataFrame,
    n_clusters: int,
    filename: str = "04_comparison_heatmaps.png",
) -> None:
    """
    Generates temporal heatmaps mapping the dominant daily/hourly topological states for each device.
    Employs discrete colormapping and rapid vectorized mode aggregation to prevent memory crashes.
    """
    df_curves = (
        jv_labeled[["id_curve", "cell_id", "cell_name", "Timestamp", "label_curve"]]
        .drop_duplicates(subset=["id_curve"])
        .copy()
    )
    unique_cells = df_curves[["cell_id", "cell_name"]].drop_duplicates().sort_values("cell_id")

    cols = min(len(unique_cells), 3)
    rows = math.ceil(len(unique_cells) / cols)

    if rows == 0 or cols == 0:
        return

    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 7 * rows), dpi=150)
    plt.subplots_adjust(hspace=0.4)
    axes_flat = np.array(axes).flatten() if len(unique_cells) > 1 else [axes]

    # Discrete colormap: neutral gray for anomalies (-1), Viridis for physical states
    color_denominator = max(n_clusters - 1, 1)
    color_list = ["#808080"] + [plt.cm.viridis(i / color_denominator) for i in range(n_clusters)]
    cmap_discrete = mcolors.ListedColormap(color_list)
    bounds = np.arange(-1.5, n_clusters + 0.5, 1)
    norm = mcolors.BoundaryNorm(bounds, cmap_discrete.N)

    for i, (_, row) in enumerate(unique_cells.iterrows()):
        ax = axes_flat[i]
        df = df_curves[df_curves["cell_id"] == row["cell_id"]].copy()
        df["date"] = pd.to_datetime(df["Timestamp"]).dt.date
        df["hour"] = pd.to_datetime(df["Timestamp"]).dt.hour

        # Vectorized mode extraction (avoids O(n) python lambda bottleneck)
        df_mode = df.groupby(["date", "hour", "label_curve"]).size().reset_index(name="count")
        df_mode = df_mode.sort_values("count", ascending=False).drop_duplicates(["date", "hour"])
        heatmap_data = df_mode.pivot(index="date", columns="hour", values="label_curve")

        sns.heatmap(heatmap_data, cmap=cmap_discrete, norm=norm, ax=ax, cbar=False)
        ax.set_title(f"Device: {row['cell_name']}", fontweight="bold")
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Date")

    for j in range(len(unique_cells), len(axes_flat)):
        fig.delaxes(axes_flat[j])

    output_path = OUTPUT_DIR / filename
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Plot saved: {output_path.name}")
    plt.close()

def plot_2d_pca(
    jv_labeled: pd.DataFrame,
    curves_normalized: pd.Series,
    pca_model: PCA,
    n_clusters: int,
    filename: str = "05_pca_2d.png",
) -> None:
    """
    Generates a static 2D scatter plot (PNG) of the PCA projection mapped to
    morphological clusters.

    Requires at least 2 principal components in the analysis PCA. If the
    pipeline selected only 1, the plot is skipped with a warning rather than
    fitting an auxiliary PCA, because a 2D projection must reflect the ACTUAL
    analysis space.
    """
    if pca_model.n_components_ < 2:
        logger.warning(
            f"Analysis PCA has {pca_model.n_components_} component(s); "
            f"cannot generate a 2D plot. Skipping."
        )
        return

    df_labels = jv_labeled[["cell_id", "curve", "label_curve"]].drop_duplicates()
    X = np.stack(curves_normalized.tolist())

    X_pca = pca_model.transform(X)[:, :2]
    var_pct = np.sum(pca_model.explained_variance_ratio_[:2]) * 100

    df_pca = curves_normalized.index.to_frame(index=False)
    df_pca[["PC1", "PC2"]] = X_pca
    df_2d = df_pca.merge(df_labels, on=["cell_id", "curve"], how="inner")
    df_2d["label_curve"] = df_2d["label_curve"].astype(int)

    fig, ax = plt.subplots(figsize=(12, 9), dpi=150)

    colors = plt.cm.viridis(np.linspace(0, 1, n_clusters))

    pruned = df_2d[df_2d["label_curve"] == -1]
    if not pruned.empty:
        ax.scatter(
            pruned["PC1"], pruned["PC2"],
            s=8, c="#808080", alpha=0.15, edgecolors="none",
            label=f"Pruned (n={len(pruned)})",
        )

    for i in range(n_clusters):
        sub = df_2d[df_2d["label_curve"] == i]
        if sub.empty:
            continue
        ax.scatter(
            sub["PC1"], sub["PC2"],
            s=8, c=[colors[i]], alpha=0.55, edgecolors="none",
            label=f"Cluster {i} (n={len(sub)})",
        )

    ax.set_xlabel("PC1", fontsize=13)
    ax.set_ylabel("PC2", fontsize=13)
    ax.set_title(
        f"2D PCA Morphological Space ({var_pct:.1f}% Total Variance)",
        fontsize=15, fontweight="bold",
    )
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best", fontsize=10, markerscale=2.5, framealpha=0.9)

    plt.tight_layout()
    output_path = OUTPUT_DIR / filename
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Plot saved: {output_path.name}")
    plt.close()

def plot_3d_pca_interactive(
    jv_labeled: pd.DataFrame,
    curves_normalized: pd.Series,
    pca_model: PCA,
    n_clusters: int,
    filename: str = "06_pca_3d_interactive.html",
) -> None:
    """
    Generates an interactive 3D scatter plot of the PCA projection mapped to
    morphological clusters.

    Note: the clustering pipeline may select fewer than 3 principal components
    when the dataset is clean enough (e.g. 2 components reach 90% variance).
    In that case, a dedicated 3-component PCA is fit ONLY for visualization,
    so the 3D space is populated without altering the analysis model.
    """
    df_labels = jv_labeled[["cell_id", "curve", "label_curve"]].drop_duplicates()
    X = np.stack(curves_normalized.tolist())

    # If the analysis PCA has >= 3 components, use it directly.
    # Otherwise, fit an auxiliary 3-component PCA for visualization.
    if pca_model.n_components_ >= 3:
        pca_viz = pca_model
        X_pca = pca_model.transform(X)[:, :3]
        var_pct = np.sum(pca_model.explained_variance_ratio_[:3]) * 100
    else:
        logger.info(
            f"Analysis PCA has {pca_model.n_components_} components; "
            f"fitting an auxiliary 3-component PCA for the 3D plot."
        )
        pca_viz = PCA(n_components=3, random_state=42)
        X_pca = pca_viz.fit_transform(X)
        var_pct = np.sum(pca_viz.explained_variance_ratio_[:3]) * 100

    df_pca = curves_normalized.index.to_frame(index=False)
    df_pca[["PC1", "PC2", "PC3"]] = X_pca
    df_3d = df_pca.merge(df_labels, on=["cell_id", "curve"], how="inner")
    df_3d["Cluster"] = df_3d["label_curve"].astype(int).astype(str)

    colors = plt.cm.viridis(np.linspace(0, 1, n_clusters))
    color_map = {
        str(i): f"rgba({int(c[0]*255)}, {int(c[1]*255)}, {int(c[2]*255)}, {c[3]})"
        for i, c in enumerate(colors)
    }

    # Explicit assignment for pruned/rejected anomalous curves in 3D space
    color_map["-1"] = "rgba(128, 128, 128, 0.15)"

    fig = px.scatter_3d(
        df_3d,
        x="PC1",
        y="PC2",
        z="PC3",
        color="Cluster",
        color_discrete_map=color_map,
        opacity=0.75,
        title=(
            f"3D PCA Morphological Space "
            f"({var_pct:.1f}% Total Variance)"
        ),
    )
    fig.update_traces(marker=dict(size=3.5))

    output_path = OUTPUT_DIR / filename
    fig.write_html(output_path, include_plotlyjs=True)
    logger.info(f"Interactive 3D plot saved: {output_path.name}")


if __name__ == "__main__":
    if not (FILE_JV_LABELED.exists() and FILE_CLUSTERING_ARTIFACTS.exists()):
        logger.error(
            "Required datasets or artifacts missing. "
            "Ensure src.03_jv_clustering executed successfully."
        )
        raise SystemExit(1)

    logger.info("Loading pre-trained dataset topology and models...")
    jv_labeled = pd.read_parquet(FILE_JV_LABELED)
    artifacts = joblib.load(FILE_CLUSTERING_ARTIFACTS)

    N_POINTS = 50
    NUM_CLUSTERS = artifacts["n_clusters"]

    logger.info("Generating Machine Learning diagnostic visualizations...")

    plot_pca_variance_and_loadings(
        pca_full=artifacts["pca_full_model"],
        pca=artifacts["pca_model"],
        n_points=N_POINTS,
    )

    plot_cluster_evaluation(
        cluster_range=artifacts["k_range"],
        inertia=artifacts["inertia"],
        silhouette=artifacts["silhouette"],
        n_clusters=NUM_CLUSTERS,
    )

    plot_cluster_median_curves(
        X_real=artifacts["X_real"],
        labels=artifacts["labels"],
        n_clusters=NUM_CLUSTERS,
        n_points=N_POINTS,
    )

    plot_comparison_heatmaps(
        jv_labeled=jv_labeled,
        n_clusters=NUM_CLUSTERS,
    )

    plot_2d_pca(
        jv_labeled=jv_labeled,
        curves_normalized=artifacts["curves_normalized"],
        pca_model=artifacts["pca_model"],
        n_clusters=NUM_CLUSTERS,
    )

    plot_3d_pca_interactive(
        jv_labeled=jv_labeled,
        curves_normalized=artifacts["curves_normalized"],
        pca_model=artifacts["pca_model"],
        n_clusters=NUM_CLUSTERS,
    )

    logger.info("All unsupervised clustering visualizations generated successfully!")
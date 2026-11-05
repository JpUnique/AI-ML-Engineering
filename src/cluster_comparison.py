from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLOTS_DIR = PROJECT_ROOT / "plots"


def compare_cluster_counts(
    scaled_features,
    min_clusters: int = 2,
    max_clusters: int = 7,
) -> pd.DataFrame:
    """Compare K-Means performance across different cluster counts."""

    results = []

    for k in range(min_clusters, max_clusters + 1):

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10,
        )

        labels = model.fit_predict(scaled_features)

        inertia = model.inertia_

        score = silhouette_score(
            scaled_features,
            labels,
        )

        results.append({
            "n_clusters": k,
            "inertia": inertia,
            "silhouette_score": score,
        })

        print(
            f"K={k} | "
            f"Inertia={inertia:.2f} | "
            f"Silhouette={score:.4f}"
        )

    comparison = pd.DataFrame(results)

    return comparison


def plot_cluster_comparison(
    comparison: pd.DataFrame,
):
    """Save the elbow curve and silhouette score plots."""

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Elbow method
    axes[0].plot(
        comparison["n_clusters"],
        comparison["inertia"],
        marker="o",
    )

    axes[0].set_title("Elbow Method")
    axes[0].set_xlabel("Number of clusters (K)")
    axes[0].set_ylabel("Inertia")

    # Silhouette score
    axes[1].plot(
        comparison["n_clusters"],
        comparison["silhouette_score"],
        marker="o",
    )

    axes[1].set_title("Silhouette Score")
    axes[1].set_xlabel("Number of clusters (K)")
    axes[1].set_ylabel("Silhouette Score")

    fig.tight_layout()

    output_path = PLOTS_DIR / "cluster_comparison.png"

    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"Cluster comparison plots saved to: {output_path}")
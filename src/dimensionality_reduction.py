from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLOTS_DIR = PROJECT_ROOT / "plots"


def run_pca(scaled_features, n_components: int = 2):
    """Reduce customer feature dimensions using PCA."""

    pca = PCA(
        n_components=n_components,
        random_state=42,
    )

    reduced_features = pca.fit_transform(scaled_features)

    explained_variance = pca.explained_variance_ratio_
    total_variance = explained_variance.sum()

    print("\nPCA completed.")

    for index, variance in enumerate(
        explained_variance,
        start=1,
    ):
        print(
            f"Principal Component {index}: "
            f"{variance:.4f} explained variance"
        )

    print(
        f"Total explained variance: {total_variance:.4f}"
    )

    return pca, reduced_features


def plot_pca(
    reduced_features,
    labels=None,
):
    """Visualize customer data using two principal components."""

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 6))

    scatter = ax.scatter(
        reduced_features[:, 0],
        reduced_features[:, 1],
        c=labels if labels is not None else None,
        cmap="viridis",
        alpha=0.7,
    )

    if labels is not None:
        fig.colorbar(scatter, ax=ax, label="Cluster")

    ax.set_title("Customer Data Visualized Using PCA")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")

    fig.tight_layout()

    output_path = PLOTS_DIR / "pca_visualization.png"

    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"PCA visualization saved to: {output_path}")

import pandas as pd

from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score


def run_dbscan(
    scaled_features,
    eps: float = 0.8,
    min_samples: int = 5,
):
    """Run DBSCAN clustering."""

    model = DBSCAN(
        eps=eps,
        min_samples=min_samples,
    )

    labels = model.fit_predict(scaled_features)

    unique_labels = set(labels)

    cluster_labels = unique_labels - {-1}

    n_clusters = len(cluster_labels)

    n_noise = (labels == -1).sum()

    print("\nDBSCAN experiment completed.")
    print(f"Clusters discovered: {n_clusters}")
    print(f"Noise points detected: {n_noise}")

    # Silhouette score requires at least two clusters,
    # and fewer clusters than observations.
    if 2 <= n_clusters < len(labels):
        score = silhouette_score(
            scaled_features,
            labels,
        )

        print(f"Silhouette score: {score:.4f}")
    else:
        print(
            "Silhouette score unavailable: "
            "DBSCAN did not produce at least two valid clusters."
        )

    return model, labels


def analyse_dbscan_results(
    customers: pd.DataFrame,
    labels,
) -> pd.DataFrame:
    """Attach DBSCAN cluster assignments to customer records."""

    results = customers.copy()

    results["dbscan_cluster"] = labels

    print("\nDBSCAN cluster distribution:")
    print(results["dbscan_cluster"].value_counts().sort_index())

    return results
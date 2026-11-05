
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src.preprocessing import prepare_features


def train_customer_clusters(customers):

    X_scaled, scaler = prepare_features(customers)

    model = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )

    model.fit(X_scaled)

    labels = model.labels_

    score = silhouette_score(
        X_scaled,
        labels
    )

    customers = customers.copy()
    customers["cluster"] = labels

    print("Silhouette Score:", round(score, 4))

    print("\nCluster distribution:")
    print(customers["cluster"].value_counts())

    return customers, model, scaler

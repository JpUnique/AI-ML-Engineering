import pandas as pd

from sklearn.ensemble import IsolationForest


def detect_anomalies(
    customers: pd.DataFrame,
    scaled_features,
    contamination: float = 0.05,
):
    """Detect unusual customer records using Isolation Forest."""

    model = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=100,
    )

    predictions = model.fit_predict(scaled_features)

    results = customers.copy()

    # Isolation Forest:
    # 1 = Normal observation
    # -1 = Anomaly

    results["anomaly"] = predictions

    results["anomaly_status"] = results["anomaly"].map({
        1: "Normal",
        -1: "Anomaly",
    })

    anomaly_count = (predictions == -1).sum()

    normal_count = (predictions == 1).sum()

    print("\nAnomaly detection completed.")
    print(f"Normal customers: {normal_count}")
    print(f"Anomalous customers: {anomaly_count}")

    print("\nAnomalous customer records:")
    print(
        results[results["anomaly"] == -1].head(10)
    )

    return model, results
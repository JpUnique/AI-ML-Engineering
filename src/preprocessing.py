
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "customers.csv"


FEATURE_COLUMNS = [
    "monthly_spending",
    "transactions_per_month",
    "account_age_months",
]


def load_customer_data():
    """Load customer records from the CSV file."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}. "
            "Run generate_data.py first."
        )

    customers = pd.read_csv(DATA_PATH)

    return customers


def prepare_features(customers):
    """
    Extract customer features and standardize them.

    Returns:
        X_scaled: Standardized feature matrix.
        scaler: Fitted StandardScaler.
    """

    features = customers[FEATURE_COLUMNS]

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(features)

    return X_scaled, scaler


def preprocess_customer_data():
    """Load and preprocess customer data."""

    customers = load_customer_data()

    if customers[FEATURE_COLUMNS].isnull().any().any():
        raise ValueError("Customer dataset contains missing values.")

    X_scaled, scaler = prepare_features(customers)

    print("\nPreprocessing completed.")
    print(f"Customers: {len(customers)}")
    print(f"Features: {X_scaled.shape[1]}")

    return customers, X_scaled, scaler


from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_PATH = DATA_DIR / "customers.csv"


def generate_customers(n=600, random_state=42):
    rng = np.random.default_rng(random_state)

    # Group 1: frequent, high-spending customers
    premium = pd.DataFrame({
        "monthly_spending": rng.normal(500000, 80000, n // 3),
        "transactions_per_month": rng.normal(40, 6, n // 3),
        "account_age_months": rng.normal(48, 10, n // 3),
    })

    # Group 2: moderate-spending customers
    regular = pd.DataFrame({
        "monthly_spending": rng.normal(200000, 50000, n // 3),
        "transactions_per_month": rng.normal(20, 4, n // 3),
        "account_age_months": rng.normal(24, 6, n // 3),
    })

    # Group 3: low-spending, occasional customers
    occasional = pd.DataFrame({
        "monthly_spending": rng.normal(40000, 15000, n // 3),
        "transactions_per_month": rng.normal(5, 2, n // 3),
        "account_age_months": rng.normal(8, 3, n // 3),
    })

    customers = pd.concat(
        [premium, regular, occasional],
        ignore_index=True
    )

    customers = customers.clip(lower=0)

    customers.insert(
        0,
        "customer_id",
        range(1, len(customers) + 1)
    )

    return customers.sample(
        frac=1,
        random_state=random_state
    ).reset_index(drop=True)


def save_customer_data(n=600, random_state=42):
    """Generate customer data and save it to the data directory."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    customers = generate_customers(
        n=n,
        random_state=random_state
    )

    customers.to_csv(DATA_PATH, index=False)

    print(f"Generated {len(customers)} customers.")
    print(f"Dataset saved to: {DATA_PATH}")

    return customers


if __name__ == "__main__":
    save_customer_data()

"""
generate_data.py
----------------
Generates a synthetic credit-scoring dataset with realistic correlations
between financial features and the binary target 'creditworthy'.

The dataset is saved to  data/credit_data.csv  for downstream consumption.
"""

import numpy as np
import pandas as pd
import os

def generate_credit_dataset(n_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    Create a synthetic credit dataset.

    Features
    --------
    income              : Annual income in USD (15 000 – 250 000)
    age                 : Applicant age (18 – 70)
    debt_ratio          : Total debt / total assets (0 – 1)
    num_credit_lines    : Number of open credit lines (1 – 20)
    payment_history     : Score 0 – 100 (higher = better)
    employment_years    : Years of continuous employment (0 – 40)
    loan_amount         : Requested loan amount (1 000 – 150 000)
    credit_utilization  : Credit-card utilization ratio (0 – 1)
    num_late_payments   : Count of late payments in last 2 yrs (0 – 20)
    has_mortgage        : 1 if applicant has a mortgage, 0 otherwise

    Target
    ------
    creditworthy : 1 = creditworthy, 0 = not creditworthy
    """
    rng = np.random.RandomState(seed)

    # --- Feature generation ---------------------------------------------------
    income             = rng.normal(60000, 25000, n_samples).clip(15000, 250000)
    age                = rng.randint(18, 71, n_samples)
    debt_ratio         = rng.beta(2, 5, n_samples)
    num_credit_lines   = rng.randint(1, 21, n_samples)
    payment_history    = rng.normal(70, 20, n_samples).clip(0, 100)
    employment_years   = rng.randint(0, 41, n_samples)
    loan_amount        = rng.normal(30000, 20000, n_samples).clip(1000, 150000)
    credit_utilization = rng.beta(2, 3, n_samples)
    num_late_payments  = rng.poisson(2, n_samples).clip(0, 20)
    has_mortgage       = rng.binomial(1, 0.35, n_samples)

    # --- Weighted composite score (simulates a bureau model) ------------------
    score = (
        0.30 * (payment_history / 100)      +   # most important factor
        0.20 * (1 - debt_ratio)              +
        0.15 * (income / 250000)             +
        0.10 * (employment_years / 40)       +
        0.10 * (1 - credit_utilization)      +
        0.10 * (1 - num_late_payments / 20)  +
        0.05 * (age / 70)
    )
    # Add Gaussian noise so the boundary is not perfectly separable
    score += rng.normal(0, 0.06, n_samples)

    creditworthy = (score > 0.55).astype(int)

    # --- Build DataFrame ------------------------------------------------------
    df = pd.DataFrame({
        "income":             np.round(income, 2),
        "age":                age,
        "debt_ratio":         np.round(debt_ratio, 4),
        "num_credit_lines":   num_credit_lines,
        "payment_history":    np.round(payment_history, 2),
        "employment_years":   employment_years,
        "loan_amount":        np.round(loan_amount, 2),
        "credit_utilization": np.round(credit_utilization, 4),
        "num_late_payments":  num_late_payments,
        "has_mortgage":       has_mortgage,
        "creditworthy":       creditworthy,
    })
    return df


def main():
    """Generate the dataset and persist to CSV."""
    df = generate_credit_dataset(n_samples=1000)

    # Save next to this script
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "credit_data.csv")
    df.to_csv(out_path, index=False)

    print(f"[✓] Dataset generated: {out_path}")
    print(f"    Shape : {df.shape}")
    print(f"    Target distribution:\n{df['creditworthy'].value_counts().to_string()}")


if __name__ == "__main__":
    main()

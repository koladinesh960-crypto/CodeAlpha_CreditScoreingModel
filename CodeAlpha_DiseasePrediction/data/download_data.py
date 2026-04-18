"""
download_data.py
----------------
Downloads the UCI Heart Disease (Cleveland) dataset and saves it as heart.csv.

Dataset source: UCI Machine Learning Repository
URL: https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/
"""

import os
import pandas as pd


# Column names for the Cleveland dataset
COLUMN_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target",
]

# Direct URL to the processed Cleveland data
DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)


def download_heart_dataset(output_path: str = None) -> pd.DataFrame:
    """
    Download and clean the UCI Heart Disease dataset.

    The original target has values 0–4.  We binarize it:
      0 → 0 (no disease)
      1–4 → 1 (disease present)

    Returns
    -------
    df : pd.DataFrame   –  cleaned dataset with 'target' column
    """
    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "heart.csv"
        )

    print(f"[▶] Downloading Heart Disease dataset from UCI ...")

    try:
        df = pd.read_csv(DATA_URL, header=None, names=COLUMN_NAMES, na_values="?")
    except Exception as e:
        print(f"[!] Download failed: {e}")
        print("[!] Generating a synthetic fallback dataset ...")
        df = _generate_fallback()

    # Drop rows with missing values (typically only a few)
    rows_before = len(df)
    df = df.dropna().reset_index(drop=True)
    print(f"    Dropped {rows_before - len(df)} rows with missing values")

    # Binarize target: 0 = no disease, 1+ = disease present
    df["target"] = (df["target"] > 0).astype(int)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[✓] Dataset saved: {output_path}")
    print(f"    Shape: {df.shape}")
    print(f"    Target distribution:\n{df['target'].value_counts().to_string()}")

    return df


def _generate_fallback(n: int = 303, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic heart-disease-like dataset as fallback."""
    import numpy as np
    rng = np.random.RandomState(seed)

    df = pd.DataFrame({
        "age":      rng.randint(29, 78, n),
        "sex":      rng.binomial(1, 0.68, n),
        "cp":       rng.randint(0, 4, n),
        "trestbps": rng.normal(131, 18, n).clip(90, 200).astype(int),
        "chol":     rng.normal(246, 52, n).clip(100, 600).astype(int),
        "fbs":      rng.binomial(1, 0.15, n),
        "restecg":  rng.choice([0, 1, 2], n, p=[0.49, 0.49, 0.02]),
        "thalach":  rng.normal(150, 23, n).clip(70, 210).astype(int),
        "exang":    rng.binomial(1, 0.33, n),
        "oldpeak":  rng.exponential(1.0, n).clip(0, 6.2).round(1),
        "slope":    rng.choice([0, 1, 2], n, p=[0.07, 0.47, 0.46]),
        "ca":       rng.choice([0, 1, 2, 3], n, p=[0.58, 0.22, 0.13, 0.07]),
        "thal":     rng.choice([0, 1, 2, 3], n, p=[0.01, 0.06, 0.55, 0.38]),
        "target":   rng.binomial(1, 0.46, n),
    })
    return df


if __name__ == "__main__":
    download_heart_dataset()

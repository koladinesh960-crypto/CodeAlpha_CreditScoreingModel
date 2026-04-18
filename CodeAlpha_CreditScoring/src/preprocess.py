"""
preprocess.py
-------------
Handles data loading, cleaning, feature engineering, and train/test splitting
for the credit-scoring pipeline.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


def load_data(data_path: str = None) -> pd.DataFrame:
    """
    Load the credit-scoring CSV.
    If no path is given, look for  data/credit_data.csv  relative to project root.
    """
    if data_path is None:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_path = os.path.join(root, "data", "credit_data.csv")

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. "
            "Run  python data/generate_data.py  first."
        )

    df = pd.read_csv(data_path)
    print(f"[✓] Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def preprocess(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Preprocess the credit dataset:
      1. Separate features (X) and target (y).
      2. Train / test split.
      3. Standard-scale numerical features.

    Returns
    -------
    X_train, X_test : np.ndarray   – scaled feature matrices
    y_train, y_test : np.ndarray   – binary labels
    scaler          : StandardScaler – fitted scaler (saved for inference)
    feature_names   : list[str]
    """
    target_col = "creditworthy"
    feature_cols = [c for c in df.columns if c != target_col]

    X = df[feature_cols].values
    y = df[target_col].values

    # Stratified split preserves class ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Standard scaling (zero-mean, unit-variance)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    print(f"[✓] Preprocessing complete — Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test, scaler, feature_cols


def save_scaler(scaler: StandardScaler, path: str):
    """Persist the fitted scaler for inference."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(scaler, path)
    print(f"[✓] Scaler saved to {path}")

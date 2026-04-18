"""
preprocess.py
-------------
Loads, cleans, and splits the Heart Disease dataset for model training.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


# Feature descriptions for interpretability
FEATURE_INFO = {
    "age":      "Age in years",
    "sex":      "Sex (1 = male, 0 = female)",
    "cp":       "Chest pain type (0–3)",
    "trestbps": "Resting blood pressure (mm Hg)",
    "chol":     "Serum cholesterol (mg/dL)",
    "fbs":      "Fasting blood sugar > 120 mg/dL (1 = true)",
    "restecg":  "Resting ECG results (0–2)",
    "thalach":  "Maximum heart rate achieved",
    "exang":    "Exercise-induced angina (1 = yes)",
    "oldpeak":  "ST depression induced by exercise",
    "slope":    "Slope of peak exercise ST segment",
    "ca":       "Number of major vessels colored by fluoroscopy (0–3)",
    "thal":     "Thalassemia (0 = normal, 1 = fixed defect, 2 = reversible defect)",
}


def load_data(data_path: str = None) -> pd.DataFrame:
    """Load heart disease CSV."""
    if data_path is None:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_path = os.path.join(root, "data", "heart.csv")

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. "
            "Run  python data/download_data.py  first."
        )

    df = pd.read_csv(data_path)
    print(f"[✓] Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def preprocess(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Preprocess the heart disease dataset:
      1. Separate features and target.
      2. Train/test split (stratified).
      3. Standard-scale features.

    Returns
    -------
    X_train, X_test, y_train, y_test, scaler, feature_names
    """
    target_col = "target"
    feature_cols = [c for c in df.columns if c != target_col]

    X = df[feature_cols].values
    y = df[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    print(f"[✓] Preprocessing complete — Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test, scaler, feature_cols


def save_scaler(scaler, path: str):
    """Save fitted scaler."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(scaler, path)
    print(f"[✓] Scaler saved → {path}")

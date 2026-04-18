"""
predict.py
----------
Utility for making single-sample credit-worthiness predictions
using a saved model and scaler.
"""

import os
import joblib
import numpy as np


# Feature order must match training data
FEATURE_NAMES = [
    "income", "age", "debt_ratio", "num_credit_lines",
    "payment_history", "employment_years", "loan_amount",
    "credit_utilization", "num_late_payments", "has_mortgage",
]


def load_artifacts(models_dir: str = None):
    """Load the best model (Random Forest) and the scaler."""
    if models_dir is None:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        models_dir = os.path.join(root, "models")

    model  = joblib.load(os.path.join(models_dir, "random_forest.pkl"))
    scaler = joblib.load(os.path.join(models_dir, "scaler.pkl"))
    return model, scaler


def predict_single(features: dict, model=None, scaler=None):
    """
    Predict creditworthiness for a single applicant.

    Parameters
    ----------
    features : dict  –  Keys must match FEATURE_NAMES.

    Returns
    -------
    prediction  : int    – 0 or 1
    probability : float  – P(creditworthy)
    """
    if model is None or scaler is None:
        model, scaler = load_artifacts()

    # Build feature vector in the correct order
    x = np.array([[features[f] for f in FEATURE_NAMES]])
    x_scaled = scaler.transform(x)

    prediction  = int(model.predict(x_scaled)[0])
    probability = float(model.predict_proba(x_scaled)[0][1])

    return prediction, probability


# ---------------------------------------------------------------------------
# Quick CLI test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sample = {
        "income": 75000,
        "age": 35,
        "debt_ratio": 0.25,
        "num_credit_lines": 5,
        "payment_history": 85,
        "employment_years": 10,
        "loan_amount": 20000,
        "credit_utilization": 0.30,
        "num_late_payments": 1,
        "has_mortgage": 1,
    }
    pred, prob = predict_single(sample)
    label = "Creditworthy" if pred == 1 else "Not Creditworthy"
    print(f"Prediction : {label}")
    print(f"Probability: {prob:.2%}")

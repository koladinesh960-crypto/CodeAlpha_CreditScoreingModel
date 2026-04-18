"""
predict.py
----------
Single-patient disease prediction utility.
"""

import os
import sys
import joblib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

FEATURE_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]


def load_artifacts(models_dir: str = None):
    """Load the best model and scaler."""
    if models_dir is None:
        models_dir = os.path.join(ROOT, "models")

    # Prefer XGBoost > Random Forest > Logistic Regression
    for fname in ["xgboost.pkl", "random_forest.pkl", "logistic_regression.pkl"]:
        path = os.path.join(models_dir, fname)
        if os.path.exists(path):
            model = joblib.load(path)
            break
    else:
        raise FileNotFoundError("No trained model found in " + models_dir)

    scaler = joblib.load(os.path.join(models_dir, "scaler.pkl"))
    return model, scaler


def predict_patient(features: dict, model=None, scaler=None):
    """
    Predict heart disease risk for a single patient.

    Returns
    -------
    prediction  : int    – 0 (no disease) or 1 (disease)
    probability : float  – P(disease)
    """
    if model is None or scaler is None:
        model, scaler = load_artifacts()

    x = np.array([[features[f] for f in FEATURE_NAMES]])
    x_scaled = scaler.transform(x)

    prediction  = int(model.predict(x_scaled)[0])
    probability = float(model.predict_proba(x_scaled)[0][1])

    return prediction, probability


if __name__ == "__main__":
    sample = {
        "age": 55, "sex": 1, "cp": 2, "trestbps": 140, "chol": 250,
        "fbs": 0, "restecg": 0, "thalach": 150, "exang": 0,
        "oldpeak": 1.5, "slope": 1, "ca": 0, "thal": 2,
    }
    pred, prob = predict_patient(sample)
    label = "Disease Detected" if pred == 1 else "No Disease"
    print(f"Prediction : {label}")
    print(f"Probability: {prob:.2%}")

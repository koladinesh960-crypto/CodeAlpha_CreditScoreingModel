"""
train.py
--------
Trains four classification models on the Heart Disease dataset:
  1. Logistic Regression
  2. Support Vector Machine (SVM)
  3. Random Forest
  4. XGBoost

All models are evaluated with 5-fold cross-validation.
"""

import os
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


def train_all_models(X_train: np.ndarray, y_train: np.ndarray, random_state: int = 42):
    """
    Train and return all four classifiers.

    Returns
    -------
    models : dict  –  {name: fitted_model}
    cv_results : dict  –  {name: {"mean": float, "std": float}}
    """
    model_specs = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=random_state
        ),
        "SVM": SVC(
            kernel="rbf", probability=True, random_state=random_state
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=random_state, n_jobs=-1
        ),
    }

    # XGBoost (optional — graceful fallback if not installed)
    try:
        from xgboost import XGBClassifier
        model_specs["XGBoost"] = XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            use_label_encoder=False, eval_metric="logloss",
            random_state=random_state, verbosity=0,
        )
    except ImportError:
        print("[!] XGBoost not installed — skipping. Install with: pip install xgboost")

    models = {}
    cv_results = {}

    for name, model in model_specs.items():
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
        model.fit(X_train, y_train)
        models[name] = model
        cv_results[name] = {"mean": cv_scores.mean(), "std": cv_scores.std()}
        print(f"[✓] {name:25s} — CV Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

    return models, cv_results


def save_models(models: dict, output_dir: str):
    """Save all trained models."""
    os.makedirs(output_dir, exist_ok=True)
    for name, model in models.items():
        fname = name.lower().replace(" ", "_") + ".pkl"
        path = os.path.join(output_dir, fname)
        joblib.dump(model, path)
        print(f"[✓] Saved {name} → {path}")

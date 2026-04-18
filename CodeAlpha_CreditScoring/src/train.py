"""
train.py
--------
Trains Logistic Regression and Random Forest classifiers on the
preprocessed credit-scoring data, with cross-validation.
"""

import os
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


def train_models(X_train: np.ndarray, y_train: np.ndarray, random_state: int = 42):
    """
    Train two classifiers and report 5-fold cross-validation accuracy.

    Returns
    -------
    models : dict  –  {"Logistic Regression": fitted_lr, "Random Forest": fitted_rf}
    """
    models = {}

    # ---- 1. Logistic Regression ----
    lr = LogisticRegression(max_iter=1000, random_state=random_state)
    lr_cv = cross_val_score(lr, X_train, y_train, cv=5, scoring="accuracy")
    lr.fit(X_train, y_train)
    models["Logistic Regression"] = lr
    print(f"[✓] Logistic Regression  — CV Accuracy: {lr_cv.mean():.4f} (±{lr_cv.std():.4f})")

    # ---- 2. Random Forest ----
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        random_state=random_state,
        n_jobs=-1,
    )
    rf_cv = cross_val_score(rf, X_train, y_train, cv=5, scoring="accuracy")
    rf.fit(X_train, y_train)
    models["Random Forest"] = rf
    print(f"[✓] Random Forest        — CV Accuracy: {rf_cv.mean():.4f} (±{rf_cv.std():.4f})")

    return models


def save_models(models: dict, output_dir: str):
    """Save all trained models to disk as .pkl files."""
    os.makedirs(output_dir, exist_ok=True)
    for name, model in models.items():
        fname = name.lower().replace(" ", "_") + ".pkl"
        path = os.path.join(output_dir, fname)
        joblib.dump(model, path)
        print(f"[✓] Saved {name} → {path}")

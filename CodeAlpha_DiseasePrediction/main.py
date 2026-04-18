"""
main.py
-------
End-to-end pipeline for Disease Prediction from Medical Data.
Downloads data → preprocesses → trains 4 models → evaluates → saves.

Usage:
    python main.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from data.download_data import download_heart_dataset
from src.preprocess     import load_data, preprocess, save_scaler
from src.train          import train_all_models, save_models
from src.evaluate       import evaluate_all


def main():
    print("=" * 60)
    print("  Disease Prediction from Medical Data — Full Pipeline")
    print("=" * 60)

    data_dir   = os.path.join(ROOT, "data")
    model_dir  = os.path.join(ROOT, "models")
    output_dir = os.path.join(ROOT, "outputs")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # --- Step 1: Download / load data -----------------------------------------
    print("\n▶ Step 1 — Downloading Heart Disease dataset ...")
    csv_path = os.path.join(data_dir, "heart.csv")
    if not os.path.exists(csv_path):
        download_heart_dataset(csv_path)
    df = load_data(csv_path)

    # --- Step 2: Preprocess ---------------------------------------------------
    print("\n▶ Step 2 — Preprocessing ...")
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess(df)
    save_scaler(scaler, os.path.join(model_dir, "scaler.pkl"))

    # --- Step 3: Train all models ---------------------------------------------
    print("\n▶ Step 3 — Training models ...")
    models, cv_results = train_all_models(X_train, y_train)
    save_models(models, model_dir)

    # --- Step 4: Evaluate -----------------------------------------------------
    print("\n▶ Step 4 — Evaluating models ...")
    all_metrics = evaluate_all(models, X_test, y_test, output_dir)

    # --- Summary --------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  ✅  Pipeline complete!")
    print(f"  Models → {model_dir}")
    print(f"  Plots  → {output_dir}")
    print()

    # Print best model
    best_name = max(all_metrics, key=lambda k: all_metrics[k]["ROC-AUC"])
    best_auc  = all_metrics[best_name]["ROC-AUC"]
    print(f"  🏆 Best model: {best_name} (ROC-AUC = {best_auc:.4f})")
    print()
    print("  Run the Flask app:  python app/app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()

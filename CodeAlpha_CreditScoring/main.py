"""
main.py
-------
End-to-end pipeline for the Credit Scoring Model.
Generates data → preprocesses → trains → evaluates → saves artefacts.

Usage:
    python main.py
"""

import os
import sys

# Ensure project root is on the path
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from data.generate_data import generate_credit_dataset
from src.preprocess    import load_data, preprocess, save_scaler
from src.train         import train_models, save_models
from src.evaluate      import (
    evaluate_model,
    plot_confusion_matrix,
    plot_roc_curves,
    plot_feature_importance,
)


def main():
    print("=" * 60)
    print("  Credit Scoring Model — Full Pipeline")
    print("=" * 60)

    # --- Paths ----------------------------------------------------------------
    data_dir   = os.path.join(ROOT, "data")
    model_dir  = os.path.join(ROOT, "models")
    output_dir = os.path.join(ROOT, "outputs")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # --- Step 1: Generate synthetic data --------------------------------------
    print("\n▶ Step 1 — Generating synthetic credit dataset ...")
    df = generate_credit_dataset(n_samples=1000)
    csv_path = os.path.join(data_dir, "credit_data.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Saved to {csv_path}")

    # --- Step 2: Preprocess ---------------------------------------------------
    print("\n▶ Step 2 — Preprocessing ...")
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess(df)
    save_scaler(scaler, os.path.join(model_dir, "scaler.pkl"))

    # --- Step 3: Train --------------------------------------------------------
    print("\n▶ Step 3 — Training models ...")
    models = train_models(X_train, y_train)
    save_models(models, model_dir)

    # --- Step 4: Evaluate -----------------------------------------------------
    print("\n▶ Step 4 — Evaluating models ...")
    roc_data = {}
    for name, model in models.items():
        metrics, y_pred, y_prob = evaluate_model(model, X_test, y_test, name)
        plot_confusion_matrix(y_test, y_pred, name, output_dir)
        roc_data[name] = y_prob

    plot_roc_curves(roc_data, y_test, output_dir)

    # Feature importance (Random Forest only)
    if "Random Forest" in models:
        plot_feature_importance(models["Random Forest"], feature_names, output_dir)

    # --- Done -----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  ✅  Pipeline complete!")
    print(f"  Models  → {model_dir}")
    print(f"  Plots   → {output_dir}")
    print("  Run the Flask app:  python app/app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()

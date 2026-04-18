"""
main.py
-------
End-to-end pipeline for Handwritten Character Recognition.
Loads data → builds CNN → trains → evaluates → saves model & plots.

Usage:
    python main.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.load_data import load_emnist_letters, preprocess_images
from src.model     import build_cnn
from src.train     import train_model, plot_training_curves
from src.evaluate  import evaluate_model


def main():
    print("=" * 60)
    print("  Handwritten Character Recognition — Full Pipeline")
    print("=" * 60)

    model_dir  = os.path.join(ROOT, "models")
    output_dir = os.path.join(ROOT, "outputs")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # --- Step 1: Load data ----------------------------------------------------
    print("\n▶ Step 1 — Loading dataset ...")
    (X_train, y_train), (X_test, y_test), num_classes, label_map = load_emnist_letters()

    # --- Step 2: Preprocess ---------------------------------------------------
    print("\n▶ Step 2 — Preprocessing images ...")
    X_train, X_test = preprocess_images(X_train, X_test)

    # --- Step 3: Build model --------------------------------------------------
    print("\n▶ Step 3 — Building CNN ...")
    model = build_cnn(input_shape=(28, 28, 1), num_classes=num_classes)

    # --- Step 4: Train --------------------------------------------------------
    print("\n▶ Step 4 — Training ...")
    model_path = os.path.join(model_dir, "char_recognition_cnn.h5")
    history = train_model(
        model, X_train, y_train, X_test, y_test,
        model_save_path=model_path,
        epochs=20,
        batch_size=128,
    )
    plot_training_curves(history, output_dir)

    # --- Step 5: Evaluate -----------------------------------------------------
    print("\n▶ Step 5 — Evaluating ...")
    evaluate_model(model, X_test, y_test, label_map, output_dir)

    print("\n" + "=" * 60)
    print("  ✅  Pipeline complete!")
    print(f"  Model → {model_path}")
    print(f"  Plots → {output_dir}")
    print("  Run the Flask app:  python app/app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()

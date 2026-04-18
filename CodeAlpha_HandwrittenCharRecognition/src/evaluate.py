"""
evaluate.py
-----------
Evaluates the trained CNN on the test set and produces a confusion matrix.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


def evaluate_model(model, X_test, y_test, label_map: dict, output_dir: str):
    """
    Evaluate the model and generate:
      - Classification report (printed)
      - Confusion matrix heatmap (saved to output_dir)

    Parameters
    ----------
    model     : keras Model
    X_test    : np.ndarray – test images
    y_test    : np.ndarray – true labels
    label_map : dict       – {int: str}  e.g. {0: 'A', 1: 'B', ...}
    output_dir: str
    """
    os.makedirs(output_dir, exist_ok=True)

    # Predictions
    y_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    # Overall accuracy
    accuracy = np.mean(y_pred == y_test)
    print(f"\n{'='*50}")
    print(f"  Test Accuracy: {accuracy:.4f}")
    print(f"{'='*50}")

    # Classification report
    labels = sorted(label_map.keys())
    target_names = [label_map[i] for i in labels]
    report = classification_report(y_test, y_pred, labels=labels,
                                   target_names=target_names, zero_division=0)
    print(report)

    # Save report to file
    with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
        f.write(f"Test Accuracy: {accuracy:.4f}\n\n")
        f.write(report)

    # Confusion matrix (for datasets with ≤ 26 classes)
    if len(labels) <= 26:
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        fig, ax = plt.subplots(figsize=(14, 12))
        sns.heatmap(cm, annot=(len(labels) <= 15), fmt="d", cmap="Blues",
                    xticklabels=target_names, yticklabels=target_names, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix — Handwritten Character Recognition")
        plt.tight_layout()
        path = os.path.join(output_dir, "confusion_matrix.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"[✓] Saved confusion matrix → {path}")

    return accuracy

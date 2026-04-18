"""
evaluate.py
-----------
Evaluates trained models on the test set and generates publication-quality
plots: confusion matrix, ROC curve, and feature-importance bar chart.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")                       # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve,
)


def evaluate_model(model, X_test, y_test, model_name: str = "Model"):
    """
    Print a classification report and return a metrics dict.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall":    recall_score(y_test, y_pred),
        "f1":        f1_score(y_test, y_pred),
        "roc_auc":   roc_auc_score(y_test, y_prob),
    }

    print(f"\n{'='*50}")
    print(f"  {model_name} — Evaluation Results")
    print(f"{'='*50}")
    for k, v in metrics.items():
        print(f"  {k:>10s}: {v:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Not CW', 'Creditworthy'])}")

    return metrics, y_pred, y_prob


def plot_confusion_matrix(y_test, y_pred, model_name: str, output_dir: str):
    """Save a confusion-matrix heatmap."""
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Not CW", "Creditworthy"],
                yticklabels=["Not CW", "Creditworthy"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    path = os.path.join(output_dir, f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[✓] Saved {path}")


def plot_roc_curves(results: dict, y_test, output_dir: str):
    """
    Plot overlaid ROC curves for all models.

    Parameters
    ----------
    results : dict  –  {model_name: y_prob}
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, y_prob in results.items():
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random Baseline")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Comparison")
    ax.legend(loc="lower right")
    plt.tight_layout()
    path = os.path.join(output_dir, "roc_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[✓] Saved {path}")


def plot_feature_importance(model, feature_names: list, output_dir: str):
    """
    Bar chart of Random Forest feature importances.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(
        range(len(importances)),
        importances[indices],
        color=sns.color_palette("viridis", len(importances)),
    )
    ax.set_yticks(range(len(importances)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title("Random Forest — Feature Importance")
    plt.tight_layout()
    path = os.path.join(output_dir, "feature_importance.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[✓] Saved {path}")

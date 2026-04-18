"""
evaluate.py
-----------
Evaluates all trained models on the test set and generates comparison plots,
confusion matrices, and ROC curves.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix, roc_curve,
)


def evaluate_all(models: dict, X_test, y_test, output_dir: str):
    """
    Evaluate every model and generate:
      - Printed classification reports
      - Confusion matrices
      - Combined ROC curve
      - Model comparison bar chart

    Returns
    -------
    all_metrics : dict  –  {name: {metric: value}}
    """
    os.makedirs(output_dir, exist_ok=True)
    all_metrics = {}
    roc_data = {}

    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = {
            "Accuracy":  accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall":    recall_score(y_test, y_pred),
            "F1-Score":  f1_score(y_test, y_pred),
            "ROC-AUC":   roc_auc_score(y_test, y_prob),
        }
        all_metrics[name] = metrics
        roc_data[name] = y_prob

        print(f"\n{'='*50}")
        print(f"  {name}")
        print(f"{'='*50}")
        for k, v in metrics.items():
            print(f"  {k:>10s}: {v:.4f}")
        print(classification_report(y_test, y_pred,
              target_names=["No Disease", "Disease"]))

        # Individual confusion matrix
        _plot_cm(y_test, y_pred, name, output_dir)

    # Combined ROC curves
    _plot_roc(roc_data, y_test, output_dir)

    # Model comparison bar chart
    _plot_comparison(all_metrics, output_dir)

    return all_metrics


def _plot_cm(y_test, y_pred, model_name, output_dir):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges",
                xticklabels=["No Disease", "Disease"],
                yticklabels=["No Disease", "Disease"], ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    path = os.path.join(output_dir, f"cm_{model_name.lower().replace(' ', '_')}.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"[✓] Saved {path}")


def _plot_roc(roc_data, y_test, output_dir):
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, y_prob in roc_data.items():
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0,1],[0,1], "k--", alpha=0.4, label="Random")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("ROC Curve Comparison — All Models")
    ax.legend(loc="lower right")
    plt.tight_layout()
    path = os.path.join(output_dir, "roc_curves.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"[✓] Saved {path}")


def _plot_comparison(all_metrics, output_dir):
    """Bar chart comparing all models across metrics."""
    model_names = list(all_metrics.keys())
    metric_names = list(next(iter(all_metrics.values())).keys())

    x = np.arange(len(model_names))
    width = 0.15
    fig, ax = plt.subplots(figsize=(10, 5))

    colors = sns.color_palette("Set2", len(metric_names))
    for i, metric in enumerate(metric_names):
        values = [all_metrics[m][metric] for m in model_names]
        ax.bar(x + i * width, values, width, label=metric, color=colors[i])

    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — All Metrics")
    ax.set_xticks(x + width * (len(metric_names) - 1) / 2)
    ax.set_xticklabels(model_names, rotation=15, ha="right")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_ylim(0.5, 1.05)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "model_comparison.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"[✓] Saved {path}")

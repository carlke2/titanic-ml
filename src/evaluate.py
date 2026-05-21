"""
evaluate.py
-----------
Model evaluation, metrics, and visualization functions.
All plots are saved to the reports/ directory.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score as sk_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


# ---------------------------------------------------------------------------
# Ensure the reports directory exists
# ---------------------------------------------------------------------------
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")


def _save_fig(fig, filename: str) -> str:
    """Save a matplotlib figure to the reports/ directory."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    path = os.path.join(REPORTS_DIR, filename)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    print(f"[plot] Saved -> {path}")
    return path


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def accuracy_score(y_true, y_pred) -> float:
    """Compute and print classification accuracy.

    Parameters
    ----------
    y_true : array-like  Ground truth labels.
    y_pred : array-like  Predicted labels.

    Returns
    -------
    float  Accuracy between 0.0 and 1.0.
    """
    acc = sk_accuracy_score(y_true, y_pred)
    print(f"  Accuracy : {acc:.4f}  ({acc*100:.2f}%)")
    return acc


def print_classification_report(y_true, y_pred) -> None:
    """Print precision, recall, F1-score, and support per class.

    Parameters
    ----------
    y_true : array-like  Ground truth labels.
    y_pred : array-like  Predicted labels.
    """
    report = classification_report(y_true, y_pred, target_names=["Did not survive", "Survived"])
    print("\n  Classification Report:")
    print("  " + report.replace("\n", "\n  "))


def roc_auc(y_true, y_prob, save: bool = True) -> float:
    """Compute ROC-AUC and optionally plot and save the ROC curve.

    Parameters
    ----------
    y_true : array-like  Ground truth binary labels.
    y_prob : array-like  Predicted probabilities for class 1.
    save   : bool        Save the plot to reports/ (default True).

    Returns
    -------
    float  AUC score.
    """
    auc = roc_auc_score(y_true, y_prob)
    fpr, tpr, _ = roc_curve(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color="#6C63FF", lw=2, label=f"ROC curve (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], color="#aaa", linestyle="--", lw=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve — Titanic Survival Classifier", fontsize=14)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save:
        _save_fig(fig, "roc_curve.png")
    plt.show()
    print(f"  ROC-AUC  : {auc:.4f}")
    return auc


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_confusion_matrix(y_true, y_pred, save: bool = True) -> None:
    """Display a seaborn confusion matrix heatmap.

    Parameters
    ----------
    y_true : array-like  Ground truth labels.
    y_pred : array-like  Predicted labels.
    save   : bool        Save the plot to reports/ (default True).
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Did not survive", "Survived"],
        yticklabels=["Did not survive", "Survived"],
        ax=ax,
        linewidths=0.5,
        annot_kws={"size": 14},
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title("Confusion Matrix — Titanic Classifier", fontsize=14)
    plt.tight_layout()

    if save:
        _save_fig(fig, "confusion_matrix.png")
    plt.show()


def plot_feature_importance(model, feature_names: list, top_n: int = 15, save: bool = True) -> None:
    """Plot a horizontal bar chart of feature importances.

    Works with tree-based models that expose feature_importances_.
    Skips gracefully for models without this attribute (e.g. LogisticRegression).

    Parameters
    ----------
    model         : fitted sklearn estimator
    feature_names : list of str  Column names used during training.
    top_n         : int  Show only the top N features (default 15).
    save          : bool  Save the plot to reports/ (default True).
    """
    # Unwrap Pipeline if needed
    clf = model
    if hasattr(model, "named_steps"):
        clf = model.named_steps.get("clf", model)

    if not hasattr(clf, "feature_importances_"):
        print("  [skip] Model does not expose feature_importances_. Skipping plot.")
        return

    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]

    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.4)))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))
    ax.barh(top_features[::-1], top_importances[::-1], color=colors[::-1])
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title(f"Top {top_n} Feature Importances", fontsize=14)
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()

    if save:
        _save_fig(fig, "feature_importance.png")
    plt.show()


def compare_models(results: dict, save: bool = True) -> None:
    """Bar chart comparing cross-validation accuracy across multiple models.

    Parameters
    ----------
    results : dict  {model_name: {'mean': float, 'std': float}}
    save    : bool  Save the plot to reports/ (default True).
    """
    names = list(results.keys())
    means = [results[n]["mean"] for n in names]
    stds = [results[n]["std"] for n in names]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#6C63FF", "#FF6584", "#43B089", "#F4A261"]
    bars = ax.bar(names, means, yerr=stds, color=colors[:len(names)],
                  capsize=6, edgecolor="white", linewidth=1.2)

    for bar, mean in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.004,
            f"{mean:.4f}",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )

    ax.set_ylim(0.7, 1.0)
    ax.set_ylabel("CV Accuracy (5-fold)", fontsize=12)
    ax.set_title("Model Comparison — Cross-Validation Accuracy", fontsize=14)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    if save:
        _save_fig(fig, "model_comparison.png")
    plt.show()

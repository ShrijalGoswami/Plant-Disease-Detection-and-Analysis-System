"""
Evaluation metrics for the classifier.

Accuracy alone can hide problems when some classes have far more images
than others, which is the case in PlantVillage. Precision, recall and F1
are therefore reported per class and as macro averages (every class
counts equally). The confusion matrix shows which diseases get mixed up.
"""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader

from src.utils import format_class_name

matplotlib.use("Agg")  # no display needed, we only save figures to disk
import matplotlib.pyplot as plt  # noqa: E402


@torch.no_grad()
def collect_predictions(model: torch.nn.Module, loader: DataLoader, device: torch.device):
    """Run the model over a loader and return (true_labels, predicted_labels)."""
    model.eval()
    all_true, all_pred = [], []
    for images, labels in loader:
        outputs = model(images.to(device))
        all_pred.extend(outputs.argmax(dim=1).cpu().tolist())
        all_true.extend(labels.tolist())
    return np.array(all_true), np.array(all_pred)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> dict:
    """Accuracy, macro and weighted precision/recall/F1, plus a per class table."""
    labels = list(range(len(class_names)))
    macro = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    weighted = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    per_class = classification_report(
        y_true, y_pred, labels=labels, target_names=class_names, output_dict=True, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro[0]),
        "macro_recall": float(macro[1]),
        "macro_f1": float(macro[2]),
        "weighted_precision": float(weighted[0]),
        "weighted_recall": float(weighted[1]),
        "weighted_f1": float(weighted[2]),
        "num_samples": int(len(y_true)),
        "per_class": {name: per_class[name] for name in class_names},
    }


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    plot_path: Path,
    csv_path: Path,
) -> None:
    """Save the confusion matrix both as a CSV and as a heatmap image."""
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    readable = [format_class_name(name) for name in class_names]
    pd.DataFrame(matrix, index=readable, columns=readable).to_csv(csv_path)

    # Normalise each row so the colour shows the recall of every class even
    # when the classes have very different sizes.
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = matrix / np.maximum(row_sums, 1)

    size = max(8, 0.35 * len(class_names))
    fig, ax = plt.subplots(figsize=(size, size))
    image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Fraction of true class")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(readable, rotation=90, fontsize=7)
    ax.set_yticklabels(readable, fontsize=7)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Confusion matrix on the test split (row normalised)")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

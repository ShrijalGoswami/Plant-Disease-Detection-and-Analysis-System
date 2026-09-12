"""
Step 3 of training: measure the saved model on the held out test split.

Usage (from the project root):
    python training/evaluate.py

The test images were never used for training or for choosing the best
epoch, so these numbers are the ones reported in the project report.
Outputs: models/metrics.json, models/confusion_matrix.png and
models/confusion_matrix.csv.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from torch.utils.data import DataLoader  # noqa: E402

from config import (  # noqa: E402
    BATCH_SIZE,
    CONFUSION_MATRIX_CSV,
    CONFUSION_MATRIX_PLOT,
    METRICS_PATH,
    NUM_WORKERS,
)
from src.dataset import LeafDataset  # noqa: E402
from src.evaluation import collect_predictions, compute_metrics, save_confusion_matrix  # noqa: E402
from src.model import load_trained_model  # noqa: E402
from src.utils import get_device, save_json  # noqa: E402


def main() -> None:
    device = get_device()
    model, class_names = load_trained_model(device=device)
    test_set = LeafDataset("test", augment=False)
    loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    y_true, y_pred = collect_predictions(model, loader, device)
    metrics = compute_metrics(y_true, y_pred, class_names)
    save_json(metrics, METRICS_PATH)
    save_confusion_matrix(y_true, y_pred, class_names, CONFUSION_MATRIX_PLOT, CONFUSION_MATRIX_CSV)

    print(f"Test images      : {metrics['num_samples']}")
    print(f"Accuracy         : {metrics['accuracy']:.4f}")
    print(f"Macro precision  : {metrics['macro_precision']:.4f}")
    print(f"Macro recall     : {metrics['macro_recall']:.4f}")
    print(f"Macro F1         : {metrics['macro_f1']:.4f}")
    print(f"Saved metrics to {METRICS_PATH}")
    print(f"Saved confusion matrix to {CONFUSION_MATRIX_PLOT}")


if __name__ == "__main__":
    main()

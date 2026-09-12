"""
Step 2 of training: fine tune MobileNetV2 on the PlantVillage split.

Usage (from the project root):
    python training/train.py [--epochs N] [--batch-size N] [--lr X]

After every epoch the model is scored on the validation split. The weights
with the best validation accuracy are kept, which protects against
overfitting in later epochs. Progress is printed and also written to
models/training.log.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from config import (  # noqa: E402
    BATCH_SIZE,
    CLASS_NAMES_PATH,
    EPOCHS,
    HISTORY_PATH,
    LEARNING_RATE,
    MODELS_DIR,
    NUM_WORKERS,
    RANDOM_SEED,
    TRAINING_LOG_PATH,
    WEIGHTS_PATH,
)
from src.dataset import LeafDataset  # noqa: E402
from src.model import build_model, save_model  # noqa: E402
from src.utils import get_device, save_json, set_seed  # noqa: E402


def setup_logging() -> logging.Logger:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[logging.FileHandler(TRAINING_LOG_PATH), logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger("train")


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    """One pass over a loader. Returns (mean loss, accuracy)."""
    model.train(train)
    total_loss, correct, seen = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            seen += images.size(0)
    return total_loss / seen, correct / seen


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine tune MobileNetV2 on leaf images")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--workers", type=int, default=NUM_WORKERS)
    args = parser.parse_args()

    log = setup_logging()
    set_seed(RANDOM_SEED)
    device = get_device()
    log.info("Using device: %s", device)

    train_set = LeafDataset("train", augment=True)
    val_set = LeafDataset("val", augment=False)
    class_names = sorted(train_set.table["class_name"].unique())
    log.info("Train images: %d, validation images: %d, classes: %d",
             len(train_set), len(val_set), len(class_names))

    pin = device.type == "cuda"
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=pin, persistent_workers=args.workers > 0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=pin, persistent_workers=args.workers > 0)

    model = build_model(num_classes=len(class_names), pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    # A small learning rate keeps the useful pretrained features intact while
    # the new classifier head and the later layers adapt to leaf images.
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = []
    best_val_acc = 0.0
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        elapsed = time.time() - epoch_start
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                        "val_loss": val_loss, "val_acc": val_acc, "seconds": elapsed})
        log.info("Epoch %d/%d  train loss %.4f acc %.4f | val loss %.4f acc %.4f | %.0fs",
                 epoch, args.epochs, train_loss, train_acc, val_loss, val_acc, elapsed)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(model, WEIGHTS_PATH)
            log.info("  saved new best model (val acc %.4f)", val_acc)

    save_json(class_names, CLASS_NAMES_PATH)
    save_json({"epochs": history, "best_val_acc": best_val_acc,
               "total_seconds": time.time() - start, "device": str(device),
               "batch_size": args.batch_size, "learning_rate": args.lr}, HISTORY_PATH)
    log.info("Training finished in %.1f minutes. Best val acc: %.4f",
             (time.time() - start) / 60, best_val_acc)


if __name__ == "__main__":
    main()

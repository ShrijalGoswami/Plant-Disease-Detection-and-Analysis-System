"""Small helper functions shared by the other modules."""

import json
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch. The split becomes exactly repeatable;
    GPU training is seeded but not bit for bit deterministic."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Use the GPU when one is available, otherwise fall back to the CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def split_class_name(class_name: str) -> tuple[str, str]:
    """
    Turn a PlantVillage folder name into a readable (crop, condition) pair.

    Folder names look like "Tomato___Late_blight" or "Apple___healthy".
    The three underscores separate the crop from the condition, and single
    underscores inside each part stand for spaces.
    """
    if "___" in class_name:
        crop, condition = class_name.split("___", 1)
    else:
        crop, condition = class_name, ""
    crop = crop.replace("_", " ").strip()
    condition = condition.replace("_", " ").strip()
    return crop, condition


def is_healthy_class(class_name: str) -> bool:
    """A class is healthy when its condition part is the word healthy."""
    _, condition = split_class_name(class_name)
    return condition.lower() == "healthy"


def format_class_name(class_name: str) -> str:
    """Readable label for the UI, e.g. 'Tomato - Late blight'."""
    crop, condition = split_class_name(class_name)
    return f"{crop} - {condition}" if condition else crop


def save_json(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

"""
Model definition and loading.

The classifier is MobileNetV2 with ImageNet pretrained weights. Only the
final fully connected layer is replaced so that it outputs one score per
plant disease class. The whole network is then fine tuned on the leaf
images.

Why MobileNetV2:
* it is small (about 2.2 million parameters for 38 classes) so it trains
  in minutes on a laptop GPU and predicts in well under a second on a CPU
* the ImageNet features (edges, textures, colour blobs) transfer well to
  leaf images, so far fewer training images are needed than training from
  scratch
* it is a plain sequential CNN that is easy to explain layer by layer
"""

from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

from config import CLASS_NAMES_PATH, WEIGHTS_PATH
from src.utils import load_json


def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    """Create MobileNetV2 with a new classification head."""
    weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    model = mobilenet_v2(weights=weights)
    # model.classifier is [Dropout, Linear(1280, 1000)]. Keep the dropout
    # and swap the last layer for one with the right number of outputs.
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def save_model(model: nn.Module, path: Path = WEIGHTS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_trained_model(
    weights_path: Path = WEIGHTS_PATH,
    class_names_path: Path = CLASS_NAMES_PATH,
    device: torch.device | None = None,
) -> tuple[nn.Module, list[str]]:
    """
    Load the fine tuned weights and the class name list saved by training.

    Raises FileNotFoundError with a clear message when the model has not
    been trained yet, so the app can show a helpful error instead of a
    stack trace.
    """
    if not weights_path.exists() or not class_names_path.exists():
        raise FileNotFoundError(
            "Trained model not found. Run `python training/train.py` first "
            f"(expected {weights_path.name} and {class_names_path.name} in {weights_path.parent})."
        )
    class_names = load_json(class_names_path)
    device = device or torch.device("cpu")
    model = build_model(num_classes=len(class_names), pretrained=False)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, class_names

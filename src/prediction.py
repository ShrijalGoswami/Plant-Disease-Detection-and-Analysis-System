"""
Module 2: Disease detection.

Takes a preprocessed image tensor, runs the trained network and turns the
raw output scores into something a user can read: the predicted class,
a confidence value between 0 and 1, whether the leaf is healthy, and the
top few alternative classes.
"""

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from src.utils import format_class_name, is_healthy_class, split_class_name


@dataclass
class PredictionResult:
    class_name: str            # raw dataset label, e.g. "Tomato___Late_blight"
    display_name: str          # readable label, e.g. "Tomato - Late blight"
    crop: str                  # e.g. "Tomato"
    condition: str             # e.g. "Late blight" or "healthy"
    confidence: float          # softmax probability of the predicted class
    is_healthy: bool
    top_k: list[tuple[str, float]] = field(default_factory=list)  # (display_name, probability)


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    class_names: list[str],
    device: torch.device | None = None,
    top_k: int = 3,
) -> PredictionResult:
    """
    Classify one preprocessed image.

    `image_tensor` must have shape (1, 3, 224, 224) or (3, 224, 224).
    The network outputs one raw score (logit) per class. Softmax converts
    them into probabilities that sum to 1, and the largest one is used as
    the confidence of the prediction.
    """
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
    if image_tensor.dim() != 4 or image_tensor.shape[0] != 1:
        raise ValueError(f"Expected a single image tensor, got shape {tuple(image_tensor.shape)}")

    device = device or next(model.parameters()).device
    model.eval()
    logits = model(image_tensor.to(device))
    probabilities = F.softmax(logits, dim=1)[0]

    top_k = min(top_k, len(class_names))
    top_probs, top_indices = torch.topk(probabilities, k=top_k)

    best_index = int(top_indices[0])
    class_name = class_names[best_index]
    crop, condition = split_class_name(class_name)

    return PredictionResult(
        class_name=class_name,
        display_name=format_class_name(class_name),
        crop=crop,
        condition=condition,
        confidence=float(top_probs[0]),
        is_healthy=is_healthy_class(class_name),
        top_k=[
            (format_class_name(class_names[int(i)]), float(p))
            for p, i in zip(top_probs, top_indices)
        ],
    )

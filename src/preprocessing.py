"""
Module 1: Image processing.

This module handles everything that happens to an image before it reaches
the classifier:

1. validate the file (extension and size)
2. decode the bytes into a pixel array
3. resize to the fixed input size expected by the network
4. scale pixel values to [0, 1] and normalise with the ImageNet statistics
5. rearrange the array into the (channels, height, width) tensor layout

The same functions are used by the training data loader and by the
Streamlit app, so the model always sees images prepared in exactly the
same way.
"""

from pathlib import Path

import cv2
import numpy as np
import torch

from config import ALLOWED_EXTENSIONS, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, MAX_UPLOAD_MB


class ImageValidationError(ValueError):
    """Raised when an uploaded file cannot be used as a leaf image."""


def validate_upload(file_name: str, file_size_bytes: int) -> None:
    """Check the extension and size before trying to decode anything."""
    extension = Path(file_name).suffix.lower().lstrip(".")
    if extension not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise ImageValidationError(
            f"Unsupported file type '.{extension}'. Please upload one of: {allowed}."
        )
    if file_size_bytes == 0:
        raise ImageValidationError("The uploaded file is empty.")
    if file_size_bytes > MAX_UPLOAD_MB * 1024 * 1024:
        raise ImageValidationError(f"The file is larger than {MAX_UPLOAD_MB} MB.")


def decode_image(file_bytes: bytes) -> np.ndarray:
    """
    Decode raw bytes into an RGB uint8 array of shape (H, W, 3).

    OpenCV returns BGR by default, so the channels are reordered. Grayscale
    and RGBA images are converted to three channel RGB so that every image
    reaches the model with the same shape.
    """
    buffer = np.frombuffer(file_bytes, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ImageValidationError(
            "The file could not be decoded as an image. It may be corrupted."
        )
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def load_image(path: str | Path) -> np.ndarray:
    """Read an image from disk as RGB. Used by the dataset loader."""
    with open(path, "rb") as f:
        return decode_image(f.read())


def resize_image(image: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    """
    Resize to a square of side `size`.

    INTER_AREA gives a cleaner result when shrinking photos, which is the
    usual case because phone photos are much larger than 224 pixels.
    """
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Scale uint8 pixels to [0, 1] and then subtract the ImageNet channel mean
    and divide by the channel standard deviation.

    The pretrained MobileNetV2 weights were learned on inputs prepared this
    way, so skipping this step would hurt accuracy.
    """
    scaled = image.astype(np.float32) / 255.0
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    return (scaled - mean) / std


def to_tensor(image: np.ndarray) -> torch.Tensor:
    """Convert a (H, W, C) float array into a (C, H, W) float tensor."""
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))


def preprocess_image(image: np.ndarray) -> torch.Tensor:
    """Full pipeline: RGB array -> resized, normalised (3, 224, 224) tensor."""
    resized = resize_image(image)
    normalized = normalize_image(resized)
    return to_tensor(normalized)


def preprocess_upload(file_name: str, file_bytes: bytes):
    """
    Convenience wrapper used by the app.

    Returns the original RGB image, the resized image (for display) and the
    model ready tensor with a batch dimension of 1.
    """
    validate_upload(file_name, len(file_bytes))
    original = decode_image(file_bytes)
    resized = resize_image(original)
    tensor = to_tensor(normalize_image(resized)).unsqueeze(0)
    return original, resized, tensor

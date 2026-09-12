"""Shared fixtures: small synthetic images so tests do not need the dataset."""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def leaf_like_image() -> np.ndarray:
    """A 300x400 RGB image with a green blob, roughly like a leaf photo."""
    image = np.full((300, 400, 3), 220, dtype=np.uint8)
    cv2.ellipse(image, (200, 150), (120, 70), 30, 0, 360, (40, 150, 60), -1)
    return image


@pytest.fixture
def jpeg_bytes(leaf_like_image) -> bytes:
    ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(leaf_like_image, cv2.COLOR_RGB2BGR))
    assert ok
    return encoded.tobytes()


@pytest.fixture
def png_bytes(leaf_like_image) -> bytes:
    ok, encoded = cv2.imencode(".png", cv2.cvtColor(leaf_like_image, cv2.COLOR_RGB2BGR))
    assert ok
    return encoded.tobytes()

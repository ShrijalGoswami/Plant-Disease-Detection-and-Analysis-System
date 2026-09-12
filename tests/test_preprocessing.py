"""Tests for Module 1: validation, decoding, resizing and normalisation."""

import numpy as np
import pytest
import torch

from config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, MAX_UPLOAD_MB
from src.preprocessing import (
    ImageValidationError,
    decode_image,
    normalize_image,
    preprocess_image,
    preprocess_upload,
    resize_image,
    validate_upload,
)


def test_valid_jpg_and_png_are_accepted():
    validate_upload("leaf.jpg", 1024)
    validate_upload("leaf.JPEG", 1024)
    validate_upload("leaf.png", 1024)


def test_unsupported_extension_is_rejected():
    with pytest.raises(ImageValidationError, match="Unsupported file type"):
        validate_upload("notes.txt", 1024)
    with pytest.raises(ImageValidationError):
        validate_upload("leaf.gif", 1024)


def test_empty_and_oversized_files_are_rejected():
    with pytest.raises(ImageValidationError, match="empty"):
        validate_upload("leaf.jpg", 0)
    with pytest.raises(ImageValidationError, match="larger"):
        validate_upload("leaf.jpg", (MAX_UPLOAD_MB + 1) * 1024 * 1024)


def test_corrupted_bytes_raise_a_clear_error():
    with pytest.raises(ImageValidationError, match="corrupted"):
        decode_image(b"this is definitely not an image")


def test_decode_returns_rgb_uint8(jpeg_bytes, leaf_like_image):
    image = decode_image(jpeg_bytes)
    assert image.dtype == np.uint8
    assert image.shape == leaf_like_image.shape
    # The blob was drawn green in RGB. If channels were still BGR the red
    # and blue values would be swapped.
    centre = image[150, 200]
    assert centre[1] > centre[0] and centre[1] > centre[2]


def test_grayscale_png_is_expanded_to_three_channels():
    import cv2

    gray = np.random.randint(0, 255, (50, 60), dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", gray)
    assert ok
    image = decode_image(encoded.tobytes())
    assert image.shape == (50, 60, 3)


def test_resize_produces_square_output(leaf_like_image):
    resized = resize_image(leaf_like_image)
    assert resized.shape == (IMAGE_SIZE, IMAGE_SIZE, 3)
    assert resized.dtype == np.uint8


def test_normalisation_matches_imagenet_formula():
    image = np.full((4, 4, 3), 255, dtype=np.uint8)
    normalized = normalize_image(image)
    expected = (1.0 - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
    assert normalized.dtype == np.float32
    assert np.allclose(normalized[0, 0], expected, atol=1e-5)


def test_preprocess_image_tensor_layout(leaf_like_image):
    tensor = preprocess_image(leaf_like_image)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
    assert tensor.dtype == torch.float32


def test_preprocess_upload_end_to_end(png_bytes):
    original, resized, tensor = preprocess_upload("leaf.png", png_bytes)
    assert original.ndim == 3
    assert resized.shape == (IMAGE_SIZE, IMAGE_SIZE, 3)
    assert tensor.shape == (1, 3, IMAGE_SIZE, IMAGE_SIZE)


def test_preprocess_upload_rejects_bad_extension(png_bytes):
    with pytest.raises(ImageValidationError):
        preprocess_upload("leaf.bmp", png_bytes)

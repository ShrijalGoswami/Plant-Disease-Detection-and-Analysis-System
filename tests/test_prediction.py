"""Tests for Module 2: model construction, loading and prediction output."""

import math

import pytest
import torch

from config import CLASS_NAMES_PATH, IMAGE_SIZE, WEIGHTS_PATH
from src.model import build_model, load_trained_model
from src.prediction import PredictionResult, predict
from src.preprocessing import preprocess_image

FAKE_CLASSES = ["Apple___Apple_scab", "Apple___healthy", "Tomato___Late_blight", "Tomato___healthy"]


@pytest.fixture(scope="module")
def untrained_model():
    """Random head on a MobileNetV2 body. Enough to test shapes and ranges."""
    torch.manual_seed(0)
    model = build_model(num_classes=len(FAKE_CLASSES), pretrained=False)
    model.eval()
    return model


def test_build_model_output_size(untrained_model):
    dummy = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    with torch.no_grad():
        logits = untrained_model(dummy)
    assert logits.shape == (1, len(FAKE_CLASSES))


def test_predict_returns_valid_result(untrained_model, leaf_like_image):
    tensor = preprocess_image(leaf_like_image)
    result = predict(untrained_model, tensor, FAKE_CLASSES, device=torch.device("cpu"))
    assert isinstance(result, PredictionResult)
    assert result.class_name in FAKE_CLASSES
    assert 0.0 <= result.confidence <= 1.0
    assert len(result.top_k) == 3
    # The top-k probabilities are sorted from highest to lowest.
    probs = [p for _, p in result.top_k]
    assert probs == sorted(probs, reverse=True)
    assert math.isclose(probs[0], result.confidence)


def test_healthy_flag_follows_class_name(untrained_model, leaf_like_image):
    tensor = preprocess_image(leaf_like_image)
    result = predict(untrained_model, tensor, FAKE_CLASSES, device=torch.device("cpu"))
    assert result.is_healthy == result.class_name.endswith("healthy")


def test_predict_rejects_batches(untrained_model):
    batch = torch.zeros(2, 3, IMAGE_SIZE, IMAGE_SIZE)
    with pytest.raises(ValueError):
        predict(untrained_model, batch, FAKE_CLASSES, device=torch.device("cpu"))


def test_missing_weights_give_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="train.py"):
        load_trained_model(weights_path=tmp_path / "none.pt", class_names_path=tmp_path / "none.json")


@pytest.mark.skipif(
    not (WEIGHTS_PATH.exists() and CLASS_NAMES_PATH.exists()),
    reason="trained weights not present, run training/train.py first",
)
def test_trained_model_loads_and_predicts(leaf_like_image):
    model, class_names = load_trained_model(device=torch.device("cpu"))
    assert len(class_names) > 1
    result = predict(model, preprocess_image(leaf_like_image), class_names, device=torch.device("cpu"))
    assert result.class_name in class_names
    assert 0.0 <= result.confidence <= 1.0

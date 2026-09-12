"""Tests for Module 3: history table, summary statistics and text report."""

from src.analytics import (
    build_text_report,
    class_distribution,
    history_to_dataframe,
    make_history_entry,
    summarize_history,
)
from src.prediction import PredictionResult
from src.utils import format_class_name, is_healthy_class, split_class_name


def make_result(class_name: str, confidence: float) -> PredictionResult:
    crop, condition = split_class_name(class_name)
    return PredictionResult(
        class_name=class_name,
        display_name=format_class_name(class_name),
        crop=crop,
        condition=condition,
        confidence=confidence,
        is_healthy=is_healthy_class(class_name),
        top_k=[(format_class_name(class_name), confidence)],
    )


def test_class_name_helpers():
    assert split_class_name("Tomato___Late_blight") == ("Tomato", "Late blight")
    assert split_class_name("Tomato___Tomato_mosaic_virus") == ("Tomato", "Tomato mosaic virus")
    assert split_class_name("Corn_(maize)___healthy") == ("Corn (maize)", "healthy")
    assert is_healthy_class("Apple___healthy")
    assert not is_healthy_class("Apple___Black_rot")
    assert format_class_name("Potato___Early_blight") == "Potato - Early blight"


def test_empty_history_summary():
    assert history_to_dataframe([]).empty
    summary = summarize_history([])
    assert summary["total"] == 0
    assert summary["most_common"] is None
    assert class_distribution([]).empty


def test_summary_counts_and_confidence():
    history = [
        make_history_entry("a.jpg", make_result("Tomato___Late_blight", 0.9)),
        make_history_entry("b.jpg", make_result("Tomato___healthy", 0.7)),
        make_history_entry("c.jpg", make_result("Tomato___Late_blight", 0.5)),
    ]
    summary = summarize_history(history)
    assert summary["total"] == 3
    assert summary["healthy"] == 1
    assert summary["diseased"] == 2
    assert abs(summary["mean_confidence"] - 0.7) < 1e-6
    assert summary["min_confidence"] == 0.5
    assert summary["max_confidence"] == 0.9
    assert summary["most_common"] == "Tomato - Late blight"

    distribution = class_distribution(history)
    assert distribution["Tomato - Late blight"] == 2
    assert distribution["Tomato - healthy"] == 1


def test_text_report_lists_every_prediction():
    history = [
        make_history_entry("a.jpg", make_result("Apple___Apple_scab", 0.81)),
        make_history_entry("b.jpg", make_result("Apple___healthy", 0.95)),
    ]
    report = build_text_report(history)
    assert "Images analysed : 2" in report
    assert "a.jpg" in report and "b.jpg" in report
    assert "Apple scab" in report
    assert "81.00%" in report

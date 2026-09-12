"""
Module 3: Analysis and reporting.

Predictions made during one app session are kept in a plain Python list
(stored in Streamlit session state by the app). This module turns that
list into a table, summary statistics, chart data and a downloadable
text report. Nothing is written to disk automatically, so there is no
database and the history disappears when the browser tab is closed.
"""

from datetime import datetime

import pandas as pd

from src.prediction import PredictionResult

HISTORY_COLUMNS = ["time", "file", "crop", "condition", "status", "confidence"]


def make_history_entry(file_name: str, result: PredictionResult) -> dict:
    """One row of the prediction history."""
    return {
        "time": datetime.now().strftime("%H:%M:%S"),
        "file": file_name,
        "crop": result.crop,
        "condition": result.condition,
        "status": "Healthy" if result.is_healthy else "Diseased",
        "confidence": round(result.confidence, 4),
    }


def history_to_dataframe(history: list[dict]) -> pd.DataFrame:
    if not history:
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    return pd.DataFrame(history, columns=HISTORY_COLUMNS)


def summarize_history(history: list[dict]) -> dict:
    """Counts and confidence statistics for the session."""
    df = history_to_dataframe(history)
    if df.empty:
        return {
            "total": 0,
            "healthy": 0,
            "diseased": 0,
            "mean_confidence": 0.0,
            "min_confidence": 0.0,
            "max_confidence": 0.0,
            "most_common": None,
        }
    labels = df["crop"] + " - " + df["condition"]
    return {
        "total": int(len(df)),
        "healthy": int((df["status"] == "Healthy").sum()),
        "diseased": int((df["status"] == "Diseased").sum()),
        "mean_confidence": float(df["confidence"].mean()),
        "min_confidence": float(df["confidence"].min()),
        "max_confidence": float(df["confidence"].max()),
        "most_common": labels.value_counts().idxmax(),
    }


def class_distribution(history: list[dict]) -> pd.Series:
    """How many times each predicted label appeared, for a bar chart."""
    df = history_to_dataframe(history)
    if df.empty:
        return pd.Series(dtype=int)
    labels = df["crop"] + " - " + df["condition"]
    return labels.value_counts()


def build_text_report(history: list[dict]) -> str:
    """Plain text summary that the user can download from the app."""
    summary = summarize_history(history)
    lines = [
        "Plant Disease Detection - Session Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"Images analysed : {summary['total']}",
        f"Healthy leaves  : {summary['healthy']}",
        f"Diseased leaves : {summary['diseased']}",
    ]
    if summary["total"] > 0:
        lines += [
            f"Mean confidence : {summary['mean_confidence']:.2%}",
            f"Lowest          : {summary['min_confidence']:.2%}",
            f"Highest         : {summary['max_confidence']:.2%}",
            f"Most common     : {summary['most_common']}",
            "",
            "Predictions",
            "-----------",
        ]
        for i, row in enumerate(history, start=1):
            lines.append(
                f"{i}. [{row['time']}] {row['file']}: {row['crop']} - {row['condition']} "
                f"({row['status']}, {row['confidence']:.2%})"
            )
    return "\n".join(lines) + "\n"

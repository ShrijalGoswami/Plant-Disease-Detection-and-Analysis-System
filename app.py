"""
Streamlit interface for the Plant Disease Detection and Analysis System.

Run from the project root with:
    streamlit run app.py

The page is organised in three tabs that map to the three functional
modules: image upload and preprocessing, disease prediction, and the
session analytics. A fourth tab shows the evaluation results of the
trained model so the numbers behind the confidence values are visible.
"""

import pandas as pd
import streamlit as st

from config import (
    CONFUSION_MATRIX_PLOT,
    HISTORY_PATH,
    IMAGE_SIZE,
    LOW_CONFIDENCE_THRESHOLD,
    METRICS_PATH,
)
from src.analytics import (
    build_text_report,
    class_distribution,
    history_to_dataframe,
    make_history_entry,
    summarize_history,
)
from src.model import load_trained_model
from src.prediction import predict
from src.preprocessing import ImageValidationError, preprocess_upload
from src.utils import get_device, load_json

st.set_page_config(page_title="Plant Disease Detection", page_icon=None, layout="wide")


@st.cache_resource(show_spinner="Loading the trained model...")
def get_model():
    """Load the network once per server process and reuse it for every request."""
    device = get_device()
    model, class_names = load_trained_model(device=device)
    return model, class_names, device


def load_optional_json(path):
    return load_json(path) if path.exists() else None


def show_preprocessing(original, resized, tensor):
    """Side by side view of the input image before and after preprocessing."""
    left, right = st.columns(2)
    with left:
        st.subheader("Original image")
        st.image(original, use_container_width=True)
        st.caption(f"Size: {original.shape[1]} x {original.shape[0]} pixels, {original.shape[2]} channels")
    with right:
        st.subheader("Processed image")
        st.image(resized, width=IMAGE_SIZE * 2)
        st.caption(f"Resized to {IMAGE_SIZE} x {IMAGE_SIZE} pixels")

    with st.expander("What happens during preprocessing"):
        st.markdown(
            f"""
1. The file bytes are decoded with OpenCV and converted from BGR to RGB.
2. The image is resized to {IMAGE_SIZE} x {IMAGE_SIZE} pixels using area interpolation.
3. Pixel values are scaled from 0-255 to 0-1.
4. Each channel is normalised with the ImageNet mean and standard deviation,
   because the pretrained MobileNetV2 weights expect that input range.
5. The array is reordered to (channels, height, width) and given a batch dimension.
"""
        )
        values = tensor.numpy()
        st.write(
            {
                "tensor shape": list(values.shape),
                "min value": round(float(values.min()), 3),
                "max value": round(float(values.max()), 3),
                "mean value": round(float(values.mean()), 3),
            }
        )


def show_prediction(result):
    """Predicted class, health status, confidence and the runner up classes."""
    status = "Healthy" if result.is_healthy else "Diseased"
    col1, col2, col3 = st.columns(3)
    col1.metric("Crop", result.crop)
    col2.metric("Condition", result.condition)
    col3.metric("Confidence", f"{result.confidence:.1%}")

    if result.is_healthy:
        st.success(f"Status: {status}. No disease pattern was detected on this leaf.")
    else:
        st.error(f"Status: {status}. The model predicts {result.display_name}.")

    st.progress(result.confidence, text="Confidence of the predicted class")
    if result.confidence < LOW_CONFIDENCE_THRESHOLD:
        st.warning(
            f"Confidence is below {LOW_CONFIDENCE_THRESHOLD:.0%}. The image may be blurry, "
            "poorly lit, contain more than one leaf, or belong to a crop the model was not "
            "trained on. Treat this result with caution."
        )

    st.markdown("**Top 3 classes**")
    st.dataframe(
        {
            "Class": [name for name, _ in result.top_k],
            "Probability": [f"{p:.2%}" for _, p in result.top_k],
        },
        hide_index=True,
        use_container_width=True,
    )


def detection_tab(model, class_names, device):
    st.markdown(
        "Upload a photo of a single plant leaf. The image is preprocessed, passed through a "
        "MobileNetV2 classifier fine tuned on the PlantVillage dataset, and the predicted "
        "crop and condition are shown together with the model confidence."
    )
    uploaded = st.file_uploader("Leaf image (JPG or PNG)", type=["jpg", "jpeg", "png"])
    if uploaded is None:
        st.info("Waiting for an image. Sample images are available in the assets/samples folder.")
        return

    try:
        original, resized, tensor = preprocess_upload(uploaded.name, uploaded.getvalue())
    except ImageValidationError as error:
        st.error(f"Could not process this file: {error}")
        return

    st.markdown("### Step 1: Image processing")
    show_preprocessing(original, resized, tensor)

    st.markdown("### Step 2: Disease detection")
    result = predict(model, tensor, class_names, device=device)
    show_prediction(result)

    # Streamlit reruns the script on every interaction, so remember which
    # upload was already recorded to avoid adding the same image twice.
    if st.session_state.get("last_recorded") != uploaded.file_id:
        st.session_state.history.append(make_history_entry(uploaded.name, result))
        st.session_state.last_recorded = uploaded.file_id


def analytics_tab():
    history = st.session_state.history
    st.markdown(
        "Every prediction made in this browser session is listed here. The history is kept in "
        "memory only and is cleared when the page is closed."
    )
    if not history:
        st.info("No predictions yet. Analyse an image in the Detect tab first.")
        return

    summary = summarize_history(history)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Images analysed", summary["total"])
    c2.metric("Healthy", summary["healthy"])
    c3.metric("Diseased", summary["diseased"])
    c4.metric("Mean confidence", f"{summary['mean_confidence']:.1%}")
    st.caption(f"Most frequent result: {summary['most_common']}")

    st.markdown("#### Prediction history")
    df = history_to_dataframe(history)
    st.dataframe(df, use_container_width=True, hide_index=True)

    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.markdown("#### Predicted class distribution")
        st.bar_chart(class_distribution(history))
    with chart_right:
        st.markdown("#### Confidence per prediction")
        st.bar_chart(df["confidence"], y_label="confidence")

    st.markdown("#### Export")
    d1, d2, d3 = st.columns(3)
    d1.download_button("Download history (CSV)", df.to_csv(index=False), "prediction_history.csv", "text/csv")
    d2.download_button("Download report (TXT)", build_text_report(history), "session_report.txt", "text/plain")
    if d3.button("Clear history"):
        # Only the list is cleared. last_recorded is kept on purpose, otherwise
        # the image still sitting in the uploader would be added again on rerun.
        st.session_state.history = []
        st.rerun()


def evaluation_tab(class_names):
    st.markdown(
        "These numbers come from `training/evaluate.py`, which scores the saved model on the "
        "test split that was never used during training."
    )
    metrics = load_optional_json(METRICS_PATH)
    training = load_optional_json(HISTORY_PATH)
    if metrics is None:
        st.warning("No metrics file found. Run `python training/evaluate.py` to generate it.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test accuracy", f"{metrics['accuracy']:.2%}")
    c2.metric("Macro precision", f"{metrics['macro_precision']:.2%}")
    c3.metric("Macro recall", f"{metrics['macro_recall']:.2%}")
    c4.metric("Macro F1", f"{metrics['macro_f1']:.2%}")
    st.caption(f"Test images: {metrics['num_samples']}. Classes: {len(class_names)}.")

    if training is not None:
        st.markdown("#### Training curve")
        epochs = training["epochs"]
        curve = pd.DataFrame(
            {
                "train accuracy": [e["train_acc"] for e in epochs],
                "validation accuracy": [e["val_acc"] for e in epochs],
            },
            index=pd.Index([e["epoch"] for e in epochs], name="epoch"),
        )
        st.line_chart(curve, x_label="epoch", y_label="accuracy")
        st.caption(
            f"{len(epochs)} epochs on {training['device']}, "
            f"{training['total_seconds'] / 60:.1f} minutes in total."
        )

    st.markdown("#### Per class scores")
    rows = [
        {
            "class": name,
            "precision": round(scores["precision"], 3),
            "recall": round(scores["recall"], 3),
            "f1": round(scores["f1-score"], 3),
            "test images": int(scores["support"]),
        }
        for name, scores in metrics["per_class"].items()
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)

    if CONFUSION_MATRIX_PLOT.exists():
        st.markdown("#### Confusion matrix")
        st.image(str(CONFUSION_MATRIX_PLOT), use_container_width=True)


def main():
    st.title("Plant Disease Detection and Analysis System")
    st.caption("Computer Vision project: leaf image classification with a fine tuned MobileNetV2")

    if "history" not in st.session_state:
        st.session_state.history = []
        st.session_state.last_recorded = None

    try:
        model, class_names, device = get_model()
    except FileNotFoundError as error:
        st.error(str(error))
        st.stop()

    with st.sidebar:
        st.markdown("### About")
        st.markdown(
            "The system classifies a leaf photo into one of the crop and disease "
            "combinations from the PlantVillage dataset."
        )
        st.markdown(f"**Classes:** {len(class_names)}")
        st.markdown(f"**Inference device:** {device.type.upper()}")
        st.markdown(f"**Input size:** {IMAGE_SIZE} x {IMAGE_SIZE}")
        with st.expander("Supported crops"):
            crops = sorted({name.split("___")[0].replace("_", " ") for name in class_names})
            st.write(", ".join(crops))

    tab_detect, tab_analytics, tab_eval = st.tabs(["Detect", "Session analytics", "Model evaluation"])
    with tab_detect:
        detection_tab(model, class_names, device)
    with tab_analytics:
        analytics_tab()
    with tab_eval:
        evaluation_tab(class_names)


if __name__ == "__main__":
    main()

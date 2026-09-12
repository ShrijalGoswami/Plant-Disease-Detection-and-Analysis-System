# Project Report Content

Plant Disease Detection and Analysis System

This file contains the text for every section of the VITyarthi project report, in the
order required by the guideline PDF. Copy it into the report template and add the
diagrams from `docs/diagrams.md` and the screenshots from `assets/screenshots/`.
Sections marked RESULTS use the numbers produced by `training/evaluate.py` and stored
in `models/metrics.json`, they were not typed by hand.

---

## 1. Cover Page

* Project title: Plant Disease Detection and Analysis System
* Course: Computer Vision
* Programme: VITyarthi Build Your Own Project
* Student name, registration number, and faculty name: (fill in)
* Date of submission: (fill in)

---

## 2. Introduction

Leaf images are one of the easiest signals to collect about the health of a crop. A
farmer or gardener can photograph a leaf with a phone in seconds, but reading that
photo correctly needs experience. Many diseases look alike in their early stage, and the
same disease looks different on different crops.

This project applies image classification, one of the core problems in Computer Vision,
to that task. A convolutional neural network (MobileNetV2) that was pretrained on
ImageNet is fine tuned on the PlantVillage dataset of leaf photographs. The trained
network is wrapped in a small Streamlit web application. The user uploads a leaf image,
the application shows the preprocessing steps, predicts the crop and condition, reports
the confidence, and keeps a session history with basic statistics and a downloadable
report.

The aim was to build something that works end to end, that can be explained layer by
layer, and that reports honest numbers from a held out test set. Every technique in
the project has a specific reason for being there, which is explained in the design
decisions section.

---

## 3. Problem Statement

Given a photograph of a single plant leaf, determine automatically whether the leaf is
healthy or diseased, identify the crop and the specific disease when one is present,
and express how confident the system is in that answer.

Constraints that shaped the solution:

* It must run on a normal laptop. Training should finish in minutes on a consumer GPU
  and prediction should take under a second on a CPU.
* It must reject bad input (wrong file type, corrupted file, empty file) with a clear
  message instead of crashing.
* Its accuracy must be measured on images that were never used during training.
* The code must be small enough for a student to explain in a viva.

### Objectives

1. Build a preprocessing pipeline (decode, resize, normalise) that is shared by
   training and inference so the model always receives identically prepared input.
2. Fine tune a lightweight pretrained CNN on a public leaf disease dataset with a
   reproducible train, validation and test split.
3. Evaluate the model with accuracy, precision, recall, F1 score and a confusion matrix.
4. Provide a web interface that shows the original and processed image, the prediction,
   the confidence and a session level analysis with export.
5. Cover the preprocessing, prediction and analytics code with unit tests.

---

## 4. Functional Requirements

The system has three major functional modules. Each maps to one source file and one
part of the interface.

### FR1: Image Processing (src/preprocessing.py)

| ID | Requirement | Where implemented |
|----|-------------|-------------------|
| FR1.1 | Accept an uploaded image in JPG, JPEG or PNG format | `validate_upload`, `st.file_uploader(type=[...])` |
| FR1.2 | Reject files with any other extension with a readable message | `validate_upload` raises `ImageValidationError` |
| FR1.3 | Reject empty files and files above 10 MB | `validate_upload` |
| FR1.4 | Reject files that cannot be decoded as an image (corrupted data) | `decode_image` returns an error when OpenCV cannot decode |
| FR1.5 | Convert grayscale and RGBA input to 3 channel RGB | `decode_image` |
| FR1.6 | Resize every image to 224 x 224 pixels | `resize_image` (OpenCV, area interpolation) |
| FR1.7 | Scale pixels to [0, 1] and normalise with ImageNet mean and std | `normalize_image` |
| FR1.8 | Show the original and the processed image side by side, with the tensor statistics | `show_preprocessing` in `app.py` |

### FR2: Disease Detection (src/model.py, src/prediction.py)

| ID | Requirement | Where implemented |
|----|-------------|-------------------|
| FR2.1 | Load the fine tuned MobileNetV2 weights and the class list once and reuse them | `load_trained_model`, cached with `st.cache_resource` |
| FR2.2 | Show a clear error when no trained model exists instead of a stack trace | `load_trained_model` raises `FileNotFoundError` with instructions |
| FR2.3 | Predict one of the 38 crop and condition classes for a preprocessed image | `predict` |
| FR2.4 | Return a confidence value in [0, 1] computed with softmax | `predict` |
| FR2.5 | Report whether the leaf is healthy or diseased | `is_healthy_class`, `PredictionResult.is_healthy` |
| FR2.6 | List the top three classes with their probabilities | `PredictionResult.top_k` |
| FR2.7 | Warn the user when the confidence is below 60% | `show_prediction` in `app.py` |

### FR3: Analysis and Reporting (src/analytics.py)

| ID | Requirement | Where implemented |
|----|-------------|-------------------|
| FR3.1 | Keep every prediction of the current session in memory | `st.session_state.history` |
| FR3.2 | Show the history as a table (time, file, crop, condition, status, confidence) | `history_to_dataframe` |
| FR3.3 | Show counts of images analysed, healthy and diseased, mean confidence, most frequent class | `summarize_history` |
| FR3.4 | Show a bar chart of the predicted class distribution and per prediction confidence | `class_distribution`, `st.bar_chart` |
| FR3.5 | Export the history as CSV and a text report | `build_text_report`, `st.download_button` |
| FR3.6 | Clear the history on request | Clear history button |
| FR3.7 | Show the test set metrics, per class scores, training curve and confusion matrix | `evaluation_tab` |

### Input / output structure

* Input: one image file (JPG or PNG) chosen by the user in the browser.
* Output for one image: crop, condition, status (healthy or diseased), confidence
  percentage, top three classes, and a warning if the confidence is low.
* Output for the session: history table, summary counts, charts, CSV file, text report.

---

## 5. Non-Functional Requirements

| ID | Category | Requirement | How it is met |
|----|----------|-------------|---------------|
| NFR1 | Performance | A prediction for one image completes in under one second on a CPU and the model loads only once per server process | MobileNetV2 has about 2.2 million parameters (measured with `sum(p.numel() for p in model.parameters())`). The model is cached with `st.cache_resource`. The measured inference time is reported in the results section. |
| NFR2 | Usability | A user with no ML background can get a result in one step: upload an image. Results use plain words (crop, condition, healthy or diseased) and a percentage | Single file uploader, three tab layout, readable labels produced by `format_class_name`, explanation expander for preprocessing |
| NFR3 | Reliability and error handling | Invalid input never crashes the app. Every failure (wrong type, empty, oversized, corrupted, missing model) produces a specific message and the app keeps running | `ImageValidationError` caught in `detection_tab`, `FileNotFoundError` caught in `main`, all covered by unit tests |
| NFR4 | Maintainability | The code is split into single purpose modules, all settings live in one config file, and there are no machine specific paths | `config.py` builds paths from `__file__`, seven source modules, three training scripts, three test files |
| NFR5 | Resource efficiency | Training fits in 6 GB of GPU memory and the saved model is under 10 MB so it can be committed to Git | Batch size 64 at 224 x 224 with MobileNetV2, weights file size reported in the results section |
| NFR6 | Reproducibility and logging | The data split is exactly repeatable, training is seeded, and every training run is logged | Fixed seed 42 gives the identical split every time and the split is stored as CSV. Training also seeds PyTorch, but GPU convolution kernels and data loader workers are not fully deterministic, so a retrain gives numbers close to, not identical to, the reported ones. `models/training.log` records every epoch of the run that produced the committed weights. |

---

## 6. System Architecture

The system has four parts. The first three run inside the Streamlit process when a
user interacts with the app. The fourth runs offline, before the app is used.

1. **Presentation layer** (`app.py`). Draws the page, receives the uploaded file, calls
   the three modules and shows their results. It holds the session history in
   `st.session_state`.
2. **Image processing module** (`src/preprocessing.py`). Pure functions that turn file
   bytes into a normalised tensor. No Streamlit code inside, so the same functions are
   used by the training data loader and the tests.
3. **Disease detection module** (`src/model.py`, `src/prediction.py`). Builds the
   MobileNetV2 architecture, loads the saved weights, and converts network output into
   a `PredictionResult`.
4. **Analysis and reporting module** (`src/analytics.py`). Works only on the in memory
   history list. Produces a DataFrame, summary numbers, chart data and a text report.
5. **Offline training pipeline** (`training/`). Three scripts that read the dataset
   folder, write the split CSVs, fine tune the network and evaluate it. Their outputs
   in `models/` are the only link between training and the app.

Data flows in one direction: file bytes -> tensor -> logits -> `PredictionResult` ->
history row -> summary. The architecture diagram in `docs/diagrams.md` shows the
modules and the files that connect them.

Storage: there is no database. The only persistent files are written by the training
scripts. The application itself writes nothing to disk. For this reason no ER diagram
or schema is included, as allowed by the guideline ("if applicable").

---

## 7. Design Diagrams

The six diagrams are rendered as PNG files in `assets/diagrams/`, ready to paste into
the report. Their Mermaid source is in `docs/diagrams.md` and can be edited and
re-rendered with https://mermaid.live.

| File | Diagram |
|------|---------|
| `01_system_architecture_diagram.png` | System architecture |
| `02_workflow_diagram_user_interaction.png` | User workflow |
| `03_training_workflow_diagram.png` | Training workflow |
| `04_use_case_diagram.png` | Use case diagram |
| `05_sequence_diagram_single_prediction.png` | Sequence diagram for one prediction |
| `06_class_component_diagram.png` | Class / component diagram |

There is no ER diagram because the application has no database (see section 6).

---

## 8. Design Decisions and Rationale

**Transfer learning instead of training from scratch.** A CNN trained from random
weights needs a lot of images and many epochs to learn basic filters (edges, colour
blobs, textures). MobileNetV2 already has those filters from ImageNet. Only the last
layer is replaced, and then the whole network is fine tuned with a small learning rate
(0.0001) so the pretrained features are adjusted rather than destroyed. This reaches
high accuracy in a handful of epochs on a laptop GPU.

**MobileNetV2 rather than ResNet50 or EfficientNet.** MobileNetV2 uses depthwise
separable convolutions and inverted residual blocks, which give it about one tenth
of the parameters of ResNet50. That means faster training, a weights file small enough
to commit to Git, and sub second prediction on a CPU. For 38 visually distinct classes
of leaf images this capacity is sufficient, which the test results confirm. EfficientNet
would give similar accuracy at similar size but its compound scaling is harder to
explain; ResNet50 would be slower with no benefit here.

**224 x 224 input with ImageNet normalisation.** These are the exact conditions under
which the pretrained weights were learned. Using a different input size or skipping
normalisation would make the first layers see a distribution they were not trained on.

**OpenCV for decoding and resizing.** OpenCV is fast, handles JPG and PNG including
alpha and grayscale, and `cv2.imdecode` returns `None` for corrupted data, which gives a
clean way to detect bad files. Area interpolation is used for resizing because almost
every upload is larger than 224 pixels and area interpolation avoids aliasing when
shrinking.

**One shared preprocessing function.** The training data loader (`LeafDataset`) and the
app both call `preprocess_image`. This removes the most common source of accuracy loss
in deployed classifiers, which is a mismatch between training and inference
preprocessing.

**Augmentation only on the training split.** Random horizontal and vertical flips,
rotation up to 20 degrees, and mild brightness, contrast and saturation jitter. Leaves
have no fixed orientation and lighting varies between photos, so these produce
realistic variations. Validation and test images are never augmented, otherwise the
reported metrics would not reflect real use.

**Stratified 70 / 15 / 15 split saved as CSV.** Stratification keeps the class
proportions the same in all three sets. A fixed seed and saved CSV files mean the split
never changes between runs, and the test set can be shown to have zero overlap with
training. Keeping the best validation epoch, not the last one, guards against
overfitting.

**Softmax probability as confidence, with a 60% warning threshold.** The softmax
output is the standard way to turn logits into a probability distribution. A threshold
warning is a simple and honest way to tell the user that the top class did not clearly
win, for example when the photo is blurry or shows a crop the model never saw.

**Session state instead of a database.** The guideline asks for reporting or analytics,
not for persistence. Streamlit session state is enough for a per session history, keeps
the project simple and avoids inventing a schema that nothing needs. CSV and text
export give the user a way to keep the results.

**Streamlit as the interface.** It runs as one Python file, needs no HTML or JavaScript,
and its widgets (file uploader, metrics, tables, bar charts, download buttons) cover
everything the three modules need to display.

---

## 9. Implementation Details

### Technology stack

| Purpose | Tool |
|---------|------|
| Deep learning | PyTorch 2.10, torchvision 0.25 (MobileNetV2 weights) |
| Image processing | OpenCV (opencv-python-headless), NumPy |
| Metrics | scikit-learn (accuracy, precision, recall, F1, confusion matrix) |
| Plots | matplotlib (confusion matrix image), Streamlit built in charts |
| Interface | Streamlit |
| Tables and export | pandas |
| Tests | pytest |
| Language | Python 3.11 |

### Dataset

PlantVillage (Hughes and Salathe, 2015), colour images, downloaded from the public
GitHub mirror `spMohanty/PlantVillage-Dataset` (folder `raw/color`). It contains leaf
photographs of 14 crops in 38 classes. Every class name has the form
`Crop___Condition`, for example `Tomato___Late_blight` or `Apple___healthy`. Twelve
crops have one healthy class each; two crops (orange and squash) only have a disease
class. The exact image count per split is printed by `prepare_data.py` and listed in
the results section. The dataset is not included in the repository because of its size.

Known limitation: the photos were taken on a plain background in a lab setting, so the
model is expected to be less accurate on field photos with soil, hands or several
leaves in the frame. This is discussed in the challenges section.

### Preprocessing pipeline (Module 1)

```
bytes -> cv2.imdecode -> BGR to RGB -> cv2.resize(224, 224, INTER_AREA)
      -> /255 -> (x - mean) / std -> transpose to (C, H, W) -> torch tensor
```

Each step is one function in `src/preprocessing.py`, and `preprocess_upload` chains
them for the app. The app shows the shape, minimum, maximum and mean of the final
tensor so the effect of normalisation is visible.

### Model (Module 2)

`build_model` loads `torchvision.models.mobilenet_v2` with ImageNet weights and replaces
`classifier[1]` (a `Linear(1280, 1000)`) with `Linear(1280, num_classes)`. The dropout
layer before it is kept. Training uses cross entropy loss and the Adam optimiser with
learning rate 0.0001, batch size 64, for 5 epochs. After each epoch the validation
accuracy is computed and the weights are saved only when it improves.

`predict` runs a forward pass without gradient tracking, applies softmax across the
class dimension, and takes the top three probabilities. The healthy flag is derived
from the class name.

### Analytics (Module 3)

The history is a list of dictionaries with keys time, file, crop, condition, status and
confidence. pandas converts it to a DataFrame for the table, `value_counts` gives the
class distribution, and a few aggregations give the summary. The text report is built
with plain string formatting so it can be read anywhere.

### Error handling strategy

* File validation errors are a custom exception (`ImageValidationError`) so the app can
  distinguish them from programming errors and show the message text directly.
* A corrupted file produces the same exception from `decode_image`.
* A missing model file produces a `FileNotFoundError` whose message tells the user
  which script to run.
* The Streamlit uploader itself restricts the file picker to jpg, jpeg and png, so the
  extension check is a second line of defence that also protects the function when it
  is called from tests or other code.

### Logging

`train.py` writes every epoch's loss, accuracy and duration to `models/training.log`
and to the console. `training_history.json` stores the same numbers in a form the app
can plot.

### Project structure

```
Plant-Disease-Detection-and-Analysis-System/
  app.py                     Streamlit interface
  config.py                  paths, image size, hyperparameters
  requirements.txt
  README.md
  statement.md
  src/
    preprocessing.py         Module 1
    model.py                 Module 2 (network)
    prediction.py            Module 2 (inference)
    analytics.py             Module 3
    dataset.py               split creation and Dataset class
    evaluation.py            metrics and confusion matrix
    utils.py                 seeds, device, class name helpers
  training/
    prepare_data.py          build train/val/test CSVs
    train.py                 fine tune MobileNetV2
    evaluate.py              test set metrics
  tests/
    conftest.py              synthetic image fixtures
    test_preprocessing.py
    test_prediction.py
    test_analytics.py
    test_app.py              runs app.py under Streamlit AppTest
    browser_check.py         optional Playwright checks of the running app
  models/                    weights, class list, metrics, confusion matrix, log
  assets/                    screenshots and sample images
  docs/                      report content, diagrams, test results
  data/                      dataset (not in Git) and split CSVs
```

---

## 10. Screenshots / Results (RESULTS)

See `docs/results.md` for the measured numbers, and `assets/screenshots/` for the
interface screenshots. Both were produced after training and are not estimates.

---

## 11. Testing Approach (RESULTS)

See `docs/test_results.md` for the list of tests, what each one checks, and the pytest
output from the final run.

---

## 12. Challenges Faced

See `docs/results.md`, section "Challenges", written after the training and
evaluation runs.

---

## 13. Learnings and Key Takeaways

See `docs/results.md`, section "Learnings".

---

## 14. Future Enhancements

* **Field images.** Collect or find photos taken outdoors and fine tune again, since
  PlantVillage backgrounds are uniform.
* **Unknown class handling.** Add an "unknown" decision when the maximum probability is
  low or when the image is not a leaf, using a threshold tuned on a validation set of
  out of distribution photos.
* **Explanation heatmaps.** Add Grad-CAM to show which part of the leaf drove the
  prediction. This would help verify the model looks at lesions rather than background.
* **Leaf segmentation.** Segment the leaf from the background before classification so
  that the classifier receives less irrelevant context.
* **Mobile deployment.** Export the model to ONNX or TorchScript and run it on a phone.
* **Persistent history.** Store predictions in SQLite if a multi session record is ever
  needed. This would then require a schema and an ER diagram.

---

## 15. References

1. D. P. Hughes and M. Salathe, "An open access repository of images on plant health to
   enable the development of mobile disease diagnostics", arXiv:1511.08060, 2015.
   (PlantVillage dataset.)
2. PlantVillage dataset mirror used in this project:
   https://github.com/spMohanty/PlantVillage-Dataset
3. M. Sandler, A. Howard, M. Zhu, A. Zhmoginov and L. C. Chen, "MobileNetV2: Inverted
   Residuals and Linear Bottlenecks", IEEE CVPR, 2018.
4. PyTorch documentation, torchvision.models.mobilenet_v2:
   https://pytorch.org/vision/stable/models/mobilenetv2.html
5. OpenCV documentation, image decoding and resizing:
   https://docs.opencv.org/
6. scikit-learn documentation, classification metrics:
   https://scikit-learn.org/stable/modules/model_evaluation.html
7. Streamlit documentation: https://docs.streamlit.io/
8. VITyarthi, "Build Your Own Project: General Project Instructions and Submission
   Guidelines" (course document).

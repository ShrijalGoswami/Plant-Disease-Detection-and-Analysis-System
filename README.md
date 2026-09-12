# Plant Disease Detection and Analysis System

A Computer Vision project that classifies a photograph of a plant leaf as healthy or as
one of the diseases in the PlantVillage dataset. The classifier is a MobileNetV2 network
fine tuned with transfer learning. A Streamlit interface shows the preprocessing steps,
the prediction with its confidence, and a session level analysis with export.

Built for the VITyarthi "Build Your Own Project" evaluation, Computer Vision course.

## Overview

The application works in three stages that match the three functional modules:

1. **Image processing.** The uploaded file is validated, decoded with OpenCV, resized to
   224 x 224 and normalised with the ImageNet mean and standard deviation. The original
   and processed images are displayed side by side.
2. **Disease detection.** The tensor is passed through the fine tuned MobileNetV2. The
   softmax output gives one probability per class. The app shows the crop, the
   condition, a healthy or diseased status, the confidence and the top three classes.
3. **Analysis and reporting.** Every prediction in the session is stored in memory and
   shown as a table with counts, average confidence, a class distribution chart and
   download buttons for a CSV file and a plain text report.

A fourth tab shows the evaluation results of the trained model (accuracy, precision,
recall, F1, per class scores, training curve, confusion matrix) so the numbers behind
the app are visible.

## Problem being solved

Recognising a leaf disease from a photo needs experience. Early stages of different
diseases look similar, and the same disease looks different on different crops. The
system gives a fast first opinion with a confidence value, and it keeps a record of a
batch of leaves so the results can be reviewed together.

## Features

* Upload JPG or PNG leaf images, with checks for file type, empty files, files over
  10 MB and corrupted data. Each failure shows a specific message.
* Side by side view of the original image and the 224 x 224 processed image, plus the
  shape and value range of the normalised tensor.
* Prediction of one of 38 crop and condition classes (14 crops) with confidence.
* Healthy or diseased status, top three candidate classes, and a warning when the
  confidence is under 60%.
* Session history table, summary counts, class distribution chart, confidence chart.
* Download the history as CSV or a text report. Clear the history with one button.
* Model evaluation tab fed by the metrics file produced by the evaluation script.
* Reproducible dataset split, training with augmentation, evaluation with a confusion
  matrix, and a pytest suite.

## Computer Vision approach

| Concept | Where it is used and why |
|---------|--------------------------|
| Image decoding and colour conversion | `decode_image`: OpenCV returns BGR, the model expects RGB. Grayscale and RGBA are expanded to 3 channels. |
| Resizing | `resize_image`: fixed 224 x 224 input required by the CNN. Area interpolation avoids aliasing when shrinking photos. |
| Pixel normalisation | `normalize_image`: scale to [0, 1] then subtract ImageNet mean and divide by std, the input distribution the pretrained weights expect. |
| Data augmentation | `train_augmentation` in `src/dataset.py`: flips, rotation and colour jitter on training images only, so the model is less sensitive to leaf orientation and lighting. |
| Convolutional neural network | MobileNetV2 (depthwise separable convolutions, inverted residual blocks). |
| Transfer learning | ImageNet pretrained weights, new final layer, whole network fine tuned with a small learning rate. |
| Image classification with confidence | Softmax over 38 logits, maximum probability reported as confidence. |
| Evaluation | Accuracy, macro and weighted precision, recall and F1, per class scores, confusion matrix on a held out test split. |

## Technologies used

* Python 3.11
* PyTorch and torchvision (model and training)
* OpenCV and NumPy (image processing)
* scikit-learn (metrics, stratified split)
* pandas (history table, CSV export)
* matplotlib (confusion matrix image)
* Streamlit (interface)
* pytest (tests)

## Dataset

PlantVillage colour leaf images (Hughes and Salathe, 2015), taken from the public
GitHub mirror at https://github.com/spMohanty/PlantVillage-Dataset (folder `raw/color`).

* 38 classes named `Crop___Condition`, for example `Tomato___Late_blight`,
  `Apple___healthy`.
* 14 crops: Apple, Blueberry, Cherry, Corn (maize), Grape, Orange, Peach, Pepper (bell),
  Potato, Raspberry, Soybean, Squash, Strawberry, Tomato.
* 54,305 images in total (count printed by `training/prepare_data.py`). Class sizes
  range from 152 (Potato healthy) to 5,507 (Orange Haunglongbing).
* Split: stratified 70% train (38,013), 15% validation (8,145), 15% test (8,147) with
  random seed 42. The split
  is written to `data/splits/*.csv` so it never changes between runs, and each image
  appears in exactly one file. The test split is only used by `training/evaluate.py`.

The dataset is not committed to the repository. See "How to train the model" for the
download steps.

## Model

MobileNetV2 with ImageNet weights from torchvision. The final linear layer is replaced
by a 38 output layer and the full network is fine tuned with Adam (learning rate
0.0001), cross entropy loss, batch size 64, for 5 epochs. The weights from the epoch
with the best validation accuracy are saved to `models/mobilenetv2_plant_disease.pt`.

Why MobileNetV2: it is small (about 2.2 million parameters), trains in minutes on a
laptop GPU, predicts in well under a second on a CPU, and its ImageNet features
transfer well to leaf textures. It is also a plain sequential CNN, which makes it easy
to explain. See `docs/report_content.md` section 8 for the full rationale.

Results on the 8,147 image test split (from `models/metrics.json`, produced by
`training/evaluate.py`):

| Metric | Value |
|--------|-------|
| Accuracy | 99.25% |
| Macro precision | 99.00% |
| Macro recall | 98.82% |
| Macro F1 | 98.89% |

Training took 23.6 minutes for 5 epochs on an RTX 3050 laptop GPU. Full details,
per class scores, the confusion pairs and the training curve are in `docs/results.md`,
and the same numbers are shown in the app's Model evaluation tab.

## Project structure

```
Plant-Disease-Detection-and-Analysis-System/
  app.py                    Streamlit interface (three tabs plus evaluation view)
  config.py                 all paths, image settings and hyperparameters
  requirements.txt
  README.md
  statement.md              problem statement, scope, target users, features
  src/
    preprocessing.py        Module 1: validate, decode, resize, normalise
    model.py                Module 2: build MobileNetV2, save and load weights
    prediction.py           Module 2: softmax, confidence, top-k, healthy flag
    analytics.py            Module 3: history table, summary, charts data, report
    dataset.py              stratified split and PyTorch Dataset with augmentation
    evaluation.py           metrics and confusion matrix
    utils.py                seed, device, class name helpers, JSON helpers
  training/
    prepare_data.py         write data/splits/{train,val,test}.csv
    train.py                fine tune and save the best model
    evaluate.py             score the test split, write metrics and confusion matrix
  tests/
    conftest.py             synthetic leaf image fixtures
    test_preprocessing.py
    test_prediction.py
    test_analytics.py
    test_app.py             runs app.py through Streamlit's AppTest
    browser_check.py        optional Playwright checks against the running app
  models/                   trained weights, class list, metrics, plots, training log
  assets/
    samples/                a few dataset images for trying the app
    screenshots/            interface screenshots for the report
    diagrams/               the six design diagrams rendered as PNG
  docs/
    report_content.md       text for every section of the project report
    diagrams.md             Mermaid source for all design diagrams
    results.md              measured results, challenges, learnings
    test_results.md         what each test checks and the pytest output
    vityarthi_checklist.md  guideline requirement to repository location map
    viva_questions.md       likely questions with answers based on this implementation
    report.md               full text of the project report, assembled from the files above
    build_report.py         renders report.md plus the figures into Project_Report.pdf
    Project_Report.pdf      the submitted report
  data/                     dataset (ignored by Git) and split CSVs
```

## Installation

Python 3.10 or newer is required. From the project root:

```
python -m venv .venv
.venv\Scripts\activate          (Windows)
source .venv/bin/activate       (Linux or macOS)
pip install -r requirements.txt
```

`requirements.txt` installs the CPU build of PyTorch, which is enough to run the app
and the tests. To train on an NVIDIA GPU, install the CUDA build first, for example:

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

## How to run

The repository includes the trained weights and class list in `models/`, so the app
can be started directly:

```
streamlit run app.py
```

The browser opens at http://localhost:8501. Upload one of the images from
`assets/samples/` or any leaf photo. Use the tabs to switch between detection, session
analytics and model evaluation.

## How to train the model

1. Download the dataset. Only the colour images are needed:

   ```
   git clone --depth 1 --filter=blob:none --sparse https://github.com/spMohanty/PlantVillage-Dataset.git data/_pv_download
   cd data/_pv_download
   git sparse-checkout set raw/color
   cd ../..
   ```

   Then move `data/_pv_download/raw/color` to `data/plantvillage` so that the folder
   contains the 38 class sub folders directly. The `_pv_download` folder can be deleted
   afterwards.

2. Create the split files:

   ```
   python training/prepare_data.py
   ```

3. Train (defaults come from `config.py`, all can be overridden):

   ```
   python training/train.py --epochs 5 --batch-size 64 --lr 1e-4
   ```

   Progress is printed and written to `models/training.log`.

4. Evaluate on the test split:

   ```
   python training/evaluate.py
   ```

   This writes `models/metrics.json`, `models/confusion_matrix.png` and
   `models/confusion_matrix.csv`.

## How to test

```
python -m pytest tests -v
```

The tests use small synthetic images, so they run without the dataset. One test loads
the real trained weights and is skipped automatically if `models/` is empty. The list of
tests and the output of the last run are in `docs/test_results.md`.

Upload behaviour cannot be tested with AppTest, so there is an optional browser script
that drives the running app with a headless Chromium and also produces the screenshots:

```
pip install playwright
python -m playwright install chromium
streamlit run app.py --server.headless true      (in a second terminal)
python tests/browser_check.py
```

## Screenshots

Taken from the running app with the trained model. All files are in `assets/screenshots/`.

| File | Content |
|------|---------|
| `01_home.png` | Start page with sidebar and the three tabs |
| `02_prediction_diseased.png` | Tomato late blight: original and processed image, prediction, confidence, top 3 |
| `03_prediction_healthy.png` | Apple healthy leaf with the healthy status banner |
| `04_corrupted_file_error.png` | Error message for a file that cannot be decoded |
| `05_session_analytics.png` | History table, counts, charts and export buttons |
| `06_model_evaluation.png` | Test metrics, training curve, per class scores, confusion matrix |

![Prediction screenshot](assets/screenshots/02_prediction_diseased.png)

## Limitations

* The training images come from a lab setting with plain backgrounds. Accuracy on
  photos taken in the field, with soil, hands or several leaves visible, will be lower.
* The model can only output one of the 38 trained classes. A leaf from another crop, or
  an image that is not a leaf at all, will still be assigned a class. The low confidence
  warning helps but does not fully solve this.
* Prediction history is kept in memory for the current browser session only.
* The model classifies the whole image and does not locate the diseased area.
* The raw PlantVillage folder contains 21 exact duplicate files (checked by hashing
  every image). Because the split is done by file, 4 test images and 4 validation
  images have a byte identical copy in the training set. All 4 test copies are
  predicted correctly, and removing them leaves the test accuracy at 99.25%. Near
  duplicate photos of the same leaf are also known to exist in PlantVillage and are
  not filtered, which is one reason published accuracies on this dataset are so high.

## Future improvements

* Add an "unknown" decision using a tuned confidence threshold, or a separate
  leaf / not leaf check.
* Add Grad-CAM heatmaps to show which region of the leaf influenced the prediction.
* Fine tune on field photographs so that accuracy holds up outside the lab setting.
* Export the model to ONNX for use on mobile devices.
* Optional SQLite storage if history across sessions is ever needed.

## References

* D. P. Hughes and M. Salathe, "An open access repository of images on plant health to
  enable the development of mobile disease diagnostics", arXiv:1511.08060, 2015.
* M. Sandler et al., "MobileNetV2: Inverted Residuals and Linear Bottlenecks", CVPR 2018.
* torchvision MobileNetV2 documentation:
  https://pytorch.org/vision/stable/models/mobilenetv2.html

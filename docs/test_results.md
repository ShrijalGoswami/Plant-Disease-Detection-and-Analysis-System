# Testing Approach and Results

## Approach

Tests are written with pytest and live in the `tests/` folder. They are organised by
functional module. The preprocessing and analytics tests use small synthetic images
generated in `conftest.py` (a green ellipse on a light background) so they run in a few
seconds without the dataset. The prediction tests use an untrained MobileNetV2 body for
shape and range checks, and one test loads the real trained weights. The application
test runs `app.py` through Streamlit's `AppTest` runner, which executes the script
exactly as `streamlit run` would and records any exception.

Run everything with:

```
python -m pytest tests -v
```

## What is tested

### Module 1: Image processing (`tests/test_preprocessing.py`)

| Test | What it checks |
|------|----------------|
| `test_valid_jpg_and_png_are_accepted` | jpg, JPEG (case insensitive) and png pass validation |
| `test_unsupported_extension_is_rejected` | txt and gif raise `ImageValidationError` with an "Unsupported file type" message |
| `test_empty_and_oversized_files_are_rejected` | 0 byte files and files over 10 MB are rejected with specific messages |
| `test_corrupted_bytes_raise_a_clear_error` | random bytes that are not an image raise an error mentioning corruption |
| `test_decode_returns_rgb_uint8` | decoded array is uint8, keeps the original size, and channels are RGB not BGR (checked on the green blob) |
| `test_grayscale_png_is_expanded_to_three_channels` | a single channel PNG becomes (H, W, 3) |
| `test_resize_produces_square_output` | output is exactly 224 x 224 x 3 uint8 |
| `test_normalisation_matches_imagenet_formula` | a white pixel becomes (1 - mean) / std per channel, float32 |
| `test_preprocess_image_tensor_layout` | the final tensor is a float32 torch tensor of shape (3, 224, 224) |
| `test_preprocess_upload_end_to_end` | the app wrapper returns original, resized image and a (1, 3, 224, 224) tensor |
| `test_preprocess_upload_rejects_bad_extension` | the wrapper applies the extension check before decoding |

### Module 2: Disease detection (`tests/test_prediction.py`)

| Test | What it checks |
|------|----------------|
| `test_build_model_output_size` | MobileNetV2 with a replaced head outputs one logit per class |
| `test_predict_returns_valid_result` | `predict` returns a `PredictionResult`, class is in the class list, confidence is in [0, 1], top 3 is sorted descending and its first entry equals the confidence |
| `test_healthy_flag_follows_class_name` | `is_healthy` is true exactly when the class name ends in "healthy" |
| `test_predict_rejects_batches` | a tensor with batch size 2 raises `ValueError` |
| `test_missing_weights_give_clear_error` | loading from a folder without weights raises `FileNotFoundError` that names `train.py` |
| `test_trained_model_loads_and_predicts` | the real saved weights load and produce a valid prediction (skipped if not trained yet) |

### Module 3: Analysis and reporting (`tests/test_analytics.py`)

| Test | What it checks |
|------|----------------|
| `test_class_name_helpers` | `Crop___Condition` folder names are split and formatted correctly, including names with brackets |
| `test_empty_history_summary` | empty history gives an empty table, zero counts and no most common class |
| `test_summary_counts_and_confidence` | totals, healthy and diseased counts, mean, min, max confidence and most common label are computed correctly from three entries; class distribution counts match |
| `test_text_report_lists_every_prediction` | the text report contains the totals, every file name, the readable class name and the formatted confidence |

### Application (`tests/test_app.py`)

| Test | What it checks |
|------|----------------|
| `test_app_starts_without_errors` | `app.py` runs under `AppTest` with no exception, shows the project title, has three tabs, shows the "waiting for an image" message, and starts with an empty history |

### Browser level tests (`tests/browser_check.py`)

Upload behaviour cannot be automated with `AppTest` because it does not support file
uploads, so these cases are executed by a script that drives the real app in a headless
Chromium browser (Playwright). The observed results are in the section below.

| Case | Expected behaviour |
|------|--------------------|
| Upload a JPG leaf from `assets/samples` | original and processed images shown, prediction and confidence shown, row added to history |
| Upload a healthy leaf | green healthy banner |
| Upload a file renamed from .txt to .jpg (corrupted content) | red error message "could not be decoded", app keeps running |
| Six uploads, then switch tabs Detect / Analytics / Detect / Analytics | exactly six history rows, no duplicates (Streamlit reruns the script on every interaction) |
| Session analytics tab | counts, charts and both download buttons present |
| Model evaluation tab | test accuracy and confusion matrix shown |
| Clear history | table and counts reset |

One further case was run by hand through `AppTest` with the `models/` folder temporarily
renamed: the app showed the red message "Trained model not found. Run `python
training/train.py` first (expected mobilenetv2_plant_disease.pt and class_names.json
in E:\Plant-Disease-Detection-and-Analysis-System\models)." and raised no exception. PNG upload is covered by the unit test
`test_preprocess_upload_end_to_end`, not by the browser script.

## Observed result of the final run

Run on 11 September 2026 with the trained model present in `models/`.

```
$ python -m pytest tests -v
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-9.0.3, pluggy-1.6.0
rootdir: E:\Plant-Disease-Detection-and-Analysis-System
collected 22 items

tests/test_analytics.py::test_class_name_helpers PASSED                  [  4%]
tests/test_analytics.py::test_empty_history_summary PASSED               [  9%]
tests/test_analytics.py::test_summary_counts_and_confidence PASSED       [ 13%]
tests/test_analytics.py::test_text_report_lists_every_prediction PASSED  [ 18%]
tests/test_app.py::test_app_starts_without_errors PASSED                 [ 22%]
tests/test_prediction.py::test_build_model_output_size PASSED            [ 27%]
tests/test_prediction.py::test_predict_returns_valid_result PASSED       [ 31%]
tests/test_prediction.py::test_healthy_flag_follows_class_name PASSED    [ 36%]
tests/test_prediction.py::test_predict_rejects_batches PASSED            [ 40%]
tests/test_prediction.py::test_missing_weights_give_clear_error PASSED   [ 45%]
tests/test_prediction.py::test_trained_model_loads_and_predicts PASSED   [ 50%]
tests/test_preprocessing.py::test_valid_jpg_and_png_are_accepted PASSED  [ 54%]
tests/test_preprocessing.py::test_unsupported_extension_is_rejected PASSED [ 59%]
tests/test_preprocessing.py::test_empty_and_oversized_files_are_rejected PASSED [ 63%]
tests/test_preprocessing.py::test_corrupted_bytes_raise_a_clear_error PASSED [ 68%]
tests/test_preprocessing.py::test_decode_returns_rgb_uint8 PASSED        [ 72%]
tests/test_preprocessing.py::test_grayscale_png_is_expanded_to_three_channels PASSED [ 77%]
tests/test_preprocessing.py::test_resize_produces_square_output PASSED   [ 81%]
tests/test_preprocessing.py::test_normalisation_matches_imagenet_formula PASSED [ 86%]
tests/test_preprocessing.py::test_preprocess_image_tensor_layout PASSED  [ 90%]
tests/test_preprocessing.py::test_preprocess_upload_end_to_end PASSED    [ 95%]
tests/test_preprocessing.py::test_preprocess_upload_rejects_bad_extension PASSED [100%]

============================= 22 passed in 14.93s =============================
```

The suite was run once more after the last edits to `app.py` (training curve axis,
top 3 table, clear history fix): 22 passed in 10.99 s.

## Observed result of the manual browser tests

The running app was driven with a headless Chromium browser (Playwright) on the same
date. The script is `tests/browser_check.py`. It uploads files through the real file
uploader, reads the page text, and prints PASS or FAIL for each of its 8 checks.

| Case | Observed |
|------|----------|
| Home page | loaded, model loaded on CUDA, sidebar shows 38 classes |
| Upload `Tomato___Late_blight.jpg` | Crop Tomato, Condition Late blight, Confidence 98.6%, red "Diseased" banner |
| Upload `Apple___healthy.jpg` | Crop Apple, Condition healthy, Confidence 100.0%, green "Healthy" banner |
| Upload a text file renamed to `corrupted_leaf.jpg` | red message "Could not process this file: The file could not be decoded as an image. It may be corrupted." App kept running |
| Six valid uploads, then switch tabs Detect / Analytics / Detect / Analytics | Images analysed 6, Healthy 2, Diseased 4, mean confidence 99.6%. No duplicate rows |
| Session analytics tab | CSV and TXT download buttons present |
| Model evaluation tab | Test accuracy 99.25% shown, confusion matrix section present |
| Clear history | "No predictions yet" message shown afterwards |

The first run of the browser script found a real bug: after "Clear history" the file
still sitting in the uploader was added to the history again on the rerun, because the
id of the last recorded upload was reset together with the list. The fix keeps that id
when the list is cleared. The table above is from the run after the fix. The script was
run once more after being moved into `tests/`, with the result "8 of 8 checks passed".

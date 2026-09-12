# Results, Challenges and Learnings

Every number on this page was produced by the scripts in this repository on
11 September 2026 and copied from `models/metrics.json`, `models/training_history.json`
and the console output. Nothing was estimated.

Hardware: laptop with an NVIDIA GeForce RTX 3050 6 GB, Python 3.11.9, PyTorch 2.10.0
(CUDA 12.8 build), torchvision 0.25.0, Windows 11.

## Dataset and split

Source: PlantVillage colour images from https://github.com/spMohanty/PlantVillage-Dataset
(`raw/color`). Output of `python training/prepare_data.py`:

| Item | Value |
|------|-------|
| Classes | 38 |
| Crops | 14 |
| Total images | 54,305 |
| Training images (70%) | 38,013 |
| Validation images (15%) | 8,145 |
| Test images (15%) | 8,147 |
| Random seed | 42 |
| Overlap between splits by file path | 0 images (checked by comparing the path columns of the three CSV files) |
| Exact duplicate files in the raw dataset | 21 (found by hashing the bytes of all 54,305 files; 54,284 unique contents) |
| Test images with a byte identical copy in the training set | 4 |
| Validation images with a byte identical copy in the training set | 4 |

The duplicate files are a property of the PlantVillage release, not of the split code.
The split treats every file as a separate image, so a duplicated photo can land in two
splits. The 4 affected test images (2 Tomato healthy, 1 Tomato late blight, 1 Apple
healthy) all carry the same label as their training copy and are all predicted
correctly. Excluding them changes the test accuracy from 8,086 / 8,147 to
8,082 / 8,143, which is still 99.25%. PlantVillage is also known to contain near
duplicate photographs of the same leaf taken from slightly different positions. Those
are not detected by hashing and were not filtered, and they are part of the reason
accuracies on this dataset are high for most published models. This is stated as a
limitation rather than hidden.

Smallest class: Potato healthy, 152 images. Largest class: Orange Haunglongbing, 5,507
images. Because of this imbalance the split is stratified and macro averaged metrics
are reported alongside accuracy.

## Training

Command: `python training/train.py` with the defaults from `config.py` (5 epochs, batch
size 64, Adam, learning rate 0.0001, 4 data loader workers).

| Epoch | Train loss | Train accuracy | Validation loss | Validation accuracy | Time |
|-------|-----------|----------------|-----------------|---------------------|------|
| 1 | 0.3837 | 91.37% | 0.0672 | 98.05% | 401 s |
| 2 | 0.0606 | 98.31% | 0.0532 | 98.22% | 299 s |
| 3 | 0.0408 | 98.78% | 0.0267 | 99.18% | 261 s |
| 4 | 0.0287 | 99.10% | 0.0281 | 99.13% | 228 s |
| 5 | 0.0260 | 99.22% | 0.0194 | 99.47% | 229 s |

Total training time: 23.6 minutes. The epoch 5 weights had the best validation
accuracy and were saved. The first epoch is slower because the operating system file
cache was still cold.

Training accuracy stays slightly below validation accuracy throughout. That is
expected: augmentation and dropout are only active during training, so the training
images are harder than the clean validation images.

## Test set evaluation

Command: `python training/evaluate.py` on the 8,147 test images, which were never
used for training or for choosing the best epoch.

| Metric | Value |
|--------|-------|
| Accuracy | 99.25% |
| Macro precision | 99.00% |
| Macro recall | 98.82% |
| Macro F1 score | 98.89% |
| Weighted F1 score | 99.25% |
| Misclassified images | 61 of 8,147 |
| Classes with a perfect F1 of 1.00 | 12 of 38 |

Lowest scoring classes (from `models/metrics.json`):

| Class | Precision | Recall | F1 | Test images |
|-------|-----------|--------|----|-------------|
| Corn (maize) Cercospora leaf spot / Gray leaf spot | 0.957 | 0.870 | 0.912 | 77 |
| Tomato Early blight | 0.917 | 0.953 | 0.935 | 150 |
| Corn (maize) Northern Leaf Blight | 0.923 | 0.980 | 0.950 | 147 |
| Potato healthy | 1.000 | 0.913 | 0.955 | 23 |
| Tomato Late blight | 0.996 | 0.951 | 0.973 | 287 |

Most frequent confusions (from `models/confusion_matrix.csv`):

| Count | True class | Predicted class |
|-------|-----------|-----------------|
| 10 | Corn Cercospora / Gray leaf spot | Corn Northern Leaf Blight |
| 6 | Tomato Late blight | Tomato Early blight |
| 4 | Tomato Septoria leaf spot | Tomato Early blight |
| 4 | Tomato Late blight | Potato Late blight |
| 3 | Tomato Early blight | Tomato Target Spot |
| 3 | Corn Northern Leaf Blight | Corn Cercospora / Gray leaf spot |

These pairs make sense. The two corn leaf diseases both produce elongated grey brown
lesions, and the tomato blights and spots all appear as dark patches with rings. Late
blight looks similar on tomato and potato because it is the same pathogen. The
confusion matrix image is at `models/confusion_matrix.png`.

## Model size and speed

| Item | Value |
|------|-------|
| Trainable parameters | 2,272,550 |
| Saved weights file | 9.34 MB |
| CPU inference time, first measurement (mean of 10 runs after warm up, taken right after the GPU evaluation) | 44.8 ms, slowest 48.8 ms |
| CPU inference time, second measurement (mean of 20 runs after warm up, idle machine) | 17.9 ms, fastest 14.3 ms, slowest 22.8 ms |

Both CPU numbers were measured on the laptop processor with the batch size 1 path used
by the app (`predict` on an already preprocessed tensor), so they exclude image
decoding and page rendering. The two runs differ because of background load on the
machine at the time. Either way the performance requirement (NFR1) of under one second
is met by a wide margin.

## Unit test run

`python -m pytest tests -v`: 22 passed, 0 failed, 0 skipped, in 14.93 seconds. The full
output is in `docs/test_results.md`.

## Screenshots

Stored in `assets/screenshots/`. Taken from the running app with the trained model.

| File | What it shows |
|------|---------------|
| `01_home.png` | Start page: title, sidebar with class count and device, the three tabs, the uploader |
| `02_prediction_diseased.png` | Tomato late blight sample: original and processed image, crop, condition, confidence 98.6%, diseased banner, top 3 classes |
| `03_prediction_healthy.png` | Apple healthy sample: healthy banner and confidence 100.0% |
| `04_corrupted_file_error.png` | A text file renamed to .jpg: the error message, app still usable |
| `05_session_analytics.png` | After six uploads: counts, history table, class distribution and confidence charts, download buttons |
| `06_model_evaluation.png` | Test metrics, training curve, per class table and confusion matrix inside the app |

## Challenges

1. **Getting the data onto the machine.** The dataset is about 1 GB of small JPEG
   files. A single sparse git checkout failed twice on a slow Wi-Fi link ("early EOF"),
   and one stalled transfer left a lock file that blocked every later attempt. The
   working approach was to fetch one class folder at a time in a retry loop and to set
   `http.lowSpeedLimit` so git aborts a stalled connection instead of waiting forever.
   The README documents the single command for a normal connection.
2. **CPU only PyTorch by default.** `pip install torch` gives a CPU build. Training
   38,000 images per epoch on the CPU would have taken hours, so the CUDA build had to
   be installed from the PyTorch index. Once torch and torchvision were both CUDA builds
   the GPU trained an epoch in about four minutes.
3. **Keeping training and inference preprocessing identical.** It is easy to resize
   with PIL in training and OpenCV in the app, or to forget normalisation in one place.
   The fix was a single `preprocess_image` function that both the Dataset class and the
   app call, and a unit test that checks the normalisation formula.
4. **Class folder names with spaces and punctuation.** Names like
   `Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot` and `Pepper,_bell___healthy`
   broke a shell loop and needed quoting. In Python the names are only used as
   dictionary keys and labels, so `split_class_name` just replaces underscores and does
   not try to be clever.
5. **Duplicate history rows in Streamlit.** Streamlit reruns the whole script on every
   widget interaction, so a naive `history.append` added the same upload again each
   time the user switched tabs. The app now stores the id of the last recorded upload
   in session state and only appends when it changes. A browser level test later
   caught a second version of the same problem: the clear history button also reset
   that id, so the upload still in the widget was re-added right after clearing.
6. **Data loader speed on Windows.** With 4 worker processes the GPU was only partly
   busy because JPEG decoding is the bottleneck. This is acceptable for a project of
   this size, but it is the reason the epochs take four minutes rather than one.

## Learnings

* Transfer learning is far more efficient than training from scratch for this kind of
  problem. One epoch of fine tuning already reached 98% validation accuracy.
* Accuracy on PlantVillage is high for almost any reasonable CNN, so the confusion
  matrix and per class scores are more informative than the headline number. They show
  exactly which diseases are visually similar.
* A held out test set and a saved split file are what make the reported numbers
  trustworthy. Without them it is easy to evaluate on images the model has seen.
* Augmentation makes training accuracy lower than validation accuracy, which looks
  odd at first but is a sign the augmentation is doing something.
* Error handling for user input (wrong file type, corrupted bytes, missing model)
  needs the same care as the model itself, because those are the cases a demo will
  actually hit.
* Softmax confidence is useful but not calibrated: an image of a crop that is not in
  the dataset can still get a high probability for the nearest looking class.

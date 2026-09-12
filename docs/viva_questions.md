# Likely Viva Questions and Answers

Answers are based on what this project actually does. Numbers are from `docs/results.md`.

## Dataset and preprocessing

**Q: Which dataset did you use and how big is it?**
PlantVillage colour images: 54,305 leaf photos, 38 classes, 14 crops. Each class is a
crop and condition pair such as Tomato late blight or Apple healthy. Twelve of the crops
have a healthy class.

**Q: How did you split the data and why?**
Stratified 70 / 15 / 15 into train (38,013), validation (8,145) and test (8,147) with
random seed 42. Stratified means every class keeps the same proportion in each set,
which matters because class sizes range from 152 to 5,507. The split is saved as CSV
files so it never changes, and I checked that no image path appears in two files.

**Q: Could there be any leakage between train and test?**
By file path, none: each of the 54,305 files is in exactly one CSV and I checked the
intersections are empty. By content, the raw PlantVillage folder has 21 exact duplicate
files, so 4 test images have a byte identical copy in training. All 4 are predicted
correctly and removing them leaves accuracy at 99.25%. PlantVillage also has near
duplicate photos of the same leaf that hashing cannot catch, which I list as a
limitation. A cleaner setup would deduplicate by image content before splitting.

**Q: What is the difference between the validation and test sets?**
Validation is used during training to pick the best epoch. Test is touched once, by
`evaluate.py`, after training is finished. The numbers in the report are from the test
set, so they were not used to make any decision about the model.

**Q: Walk me through the preprocessing pipeline.**
Decode the bytes with OpenCV, convert BGR to RGB, resize to 224 x 224 with area
interpolation, divide by 255, subtract the ImageNet channel means and divide by the
channel standard deviations, then reorder to (channels, height, width) and add a batch
dimension. Each step is one function in `src/preprocessing.py`.

**Q: Why normalise with the ImageNet mean and standard deviation?**
Because MobileNetV2's pretrained weights were learned on inputs prepared that way. If
I fed raw 0 to 255 values or 0 to 1 values, the first layers would see a very different
distribution and the pretrained filters would be much less useful.

**Q: Why 224 x 224?**
That is the resolution the network was pretrained at. It is also small enough to train
quickly. The PlantVillage images are 256 x 256, so very little detail is lost.

**Q: Why INTER_AREA for resizing?**
Almost every upload is larger than 224 pixels, so we are shrinking. Area interpolation
averages the source pixels that map to each output pixel, which avoids the aliasing you
get from nearest neighbour or plain bilinear when shrinking.

**Q: What augmentation did you use and why?**
Random horizontal and vertical flips, rotation up to 20 degrees, and mild brightness,
contrast and saturation jitter. Leaves can be photographed at any angle and under
different light, so these create realistic variations. Augmentation is applied only to
the training split. Applying it to validation or test would make the reported metrics
meaningless.

**Q: How do you make sure the app preprocesses images the same way as training?**
Both call the same `preprocess_image` function. The training Dataset class uses it for
every image, and the app calls it through `preprocess_upload`. A unit test checks the
normalisation formula.

## Model

**Q: What model did you use and why?**
MobileNetV2 pretrained on ImageNet, with the last linear layer replaced by one with 38
outputs, then fine tuned end to end. It has about 2.27 million parameters, trains an
epoch in about four minutes on my laptop GPU, predicts in 20 to 45 ms on the CPU
depending on machine load, and the weights are 9.3 MB. It is accurate enough for this dataset (99.25% test accuracy),
so a larger network like ResNet50 would add cost without benefit.

**Q: What is transfer learning and why did you use it?**
Starting from weights that were learned on a large dataset (ImageNet) and adapting
them to a new task. The early layers already detect edges, textures and colour blobs,
which are useful for leaves too. It means I need far fewer images and epochs than
training from scratch. After one epoch the validation accuracy was already 98%.

**Q: Did you freeze any layers?**
No. The whole network is trained but with a small learning rate (0.0001) so the
pretrained features are adjusted gently rather than overwritten. Freezing the backbone
would be faster per epoch but the leaf textures differ from ImageNet objects, so
letting the convolutional layers adapt helps.

**Q: What is special about the MobileNetV2 architecture?**
Depthwise separable convolutions, which split a normal convolution into a per channel
spatial convolution and a 1 x 1 pointwise convolution, cutting the computation a lot.
Inverted residual blocks, which expand the channels, apply the depthwise convolution,
then project back down, with a skip connection between the narrow ends. Linear
bottlenecks, meaning no ReLU after the projection so that information is not lost in
the narrow layers.

**Q: What loss function and optimiser?**
Cross entropy loss, which is the standard choice for multi class classification, and
Adam with learning rate 0.0001, batch size 64, 5 epochs.

**Q: Why is training accuracy lower than validation accuracy in your curve?**
Augmentation and dropout are active only during training, so the training images are
harder than the clean validation images. That is normal and shows the augmentation is
doing something.

**Q: How do you avoid overfitting?**
Pretrained weights, augmentation, the dropout layer that MobileNetV2 already has before
the classifier, a small learning rate, only 5 epochs, and saving the weights from the
best validation epoch rather than the last one.

## Prediction and confidence

**Q: How is the confidence computed?**
The network outputs 38 raw scores (logits). Softmax turns them into probabilities that
sum to one. The largest probability is the confidence and its index gives the class.

**Q: Is the confidence reliable?**
It is useful but not calibrated. On test images from the dataset it is almost always
above 95%. On a photo of a crop that is not in the 38 classes, the model still has to
pick one of them and can be confidently wrong. That is why the app warns below 60% and
why an unknown class is listed as a future improvement.

**Q: How does the app decide healthy versus diseased?**
From the class name. The PlantVillage labels end in "healthy" for healthy leaves, so
`is_healthy_class` checks that. There is no separate binary classifier.

## Evaluation

**Q: What metrics did you report and why not just accuracy?**
Accuracy 99.25%, macro precision 99.00%, macro recall 98.82%, macro F1 98.89%, per class
scores and a confusion matrix. With imbalanced classes, accuracy can look good while a
small class performs badly. Macro averaging gives every class equal weight so small
classes count.

**Q: What is precision and recall in this context?**
For one class, precision is the fraction of images predicted as that class that really
are that class. Recall is the fraction of images of that class that were found. F1 is
their harmonic mean.

**Q: Which classes did the model confuse most?**
Corn Cercospora / gray leaf spot with corn northern leaf blight (10 images), tomato late
blight with tomato early blight (6), tomato late blight with potato late blight (4). They
are visually similar diseases, and the last pair is the same pathogen on two crops.

**Q: Where did you get the numbers in your report?**
From `models/metrics.json`, `models/training_history.json` and `models/confusion_matrix.csv`,
written by my scripts on 11 September 2026. Nothing was typed by hand.

## Application and engineering

**Q: What are the three functional modules?**
Image processing (`src/preprocessing.py`), disease detection (`src/model.py` and
`src/prediction.py`), and analysis and reporting (`src/analytics.py`). The app has one
tab for each of the first two combined and one for analytics, plus a model evaluation
tab.

**Q: Where is the prediction history stored?**
In a Python list inside Streamlit's session state. It lives in server memory for that
browser session and disappears when the tab is closed. There is no database, which is
why there is no ER diagram. The user can download the history as CSV or a text report.

**Q: Why Streamlit?**
It runs from one Python file, needs no HTML or JavaScript, and has widgets for file
upload, metrics, tables, charts and downloads, which is everything the three modules
need to show.

**Q: How does the app handle bad input?**
Extension check, empty file check, 10 MB limit, and a decode check. All raise
`ImageValidationError` with a message that the app shows in red. A missing model raises
`FileNotFoundError` with instructions to run `train.py`. These are all unit tested, and
the corrupted file case was also tested in a real browser.

**Q: Why does the model load only once?**
`load_trained_model` is wrapped in `st.cache_resource`, so Streamlit keeps the loaded
network across reruns and across users on the same server. Loading takes about a
second, predicting takes milliseconds.

**Q: What bug did you find during testing?**
Streamlit reruns the script on every interaction. The first version added the current
upload to the history on every rerun, so switching tabs duplicated rows. I fixed it by
remembering the id of the last recorded upload. A browser test then showed that the
clear history button reset that id too, so the upload still in the widget was re-added.
The fix was to keep the id when clearing.

**Q: What did you test and how?**
22 pytest tests: 11 for preprocessing, 6 for prediction, 4 for analytics, 1 that runs the
app through Streamlit's AppTest. Plus 8 checks in a real headless browser with
Playwright for the upload flow, the corrupted file, tab switching and clear history.

**Q: What are the limitations?**
Training images have plain backgrounds, so field photos will be less accurate. Only 38
classes; anything else gets forced into one of them. Whole image classification, no
localisation of the lesion. History is per session only.

**Q: What would you do next?**
An unknown class threshold, Grad-CAM heatmaps to show which region drives the
prediction, fine tuning on field photos, ONNX export for mobile.

## Reproducibility

**Q: How can someone reproduce your results?**
Download the dataset folder as described in the README, run `prepare_data.py` (seed
42 gives the identical split), `train.py`, then `evaluate.py`. Training is seeded but
GPU kernels are not fully deterministic, so a retrain lands close to the reported
numbers rather than exactly on them. The trained weights are committed, so the app and
`evaluate.py` can be run without retraining and will reproduce the metrics exactly.

# Project Statement: Plant Disease Detection and Analysis System

## Problem Statement

Plant diseases reduce crop yield, and the first visible sign is usually a change on the leaves: spots, discolouration, curling or mould. Identifying the disease correctly needs an expert, and small farmers or gardeners often do not have one nearby. A wrong guess leads to the wrong treatment.

This project builds a Computer Vision system that takes a photograph of a single leaf, classifies it as healthy or as one of the known diseases for that crop, and reports how confident the model is. It also keeps track of the predictions made during a session so that a batch of leaves can be reviewed together.

## Scope

In scope:

* Image upload with validation of file type, size and content.
* A visible preprocessing pipeline (decode, resize, normalise) applied before classification.
* Image classification with a MobileNetV2 network fine tuned on the PlantVillage leaf dataset (38 crop and condition classes across 14 crops).
* Prediction output with class name, healthy or diseased status, confidence value and the top three alternatives.
* A session history with summary statistics, class distribution chart and a downloadable CSV and text report.
* Evaluation of the model on a held out test set with accuracy, precision, recall, F1 score and a confusion matrix.
* Unit tests for preprocessing, prediction and analytics.

Out of scope:

* Detecting diseases on crops that are not in the training dataset.
* Locating the diseased region inside the image (segmentation or detection).
* Recommending treatments.
* User accounts, a database, or a hosted deployment. Prediction history lives in memory for the current browser session only.

## Target Users

* Students and instructors who want a working example of transfer learning for image classification.
* Agriculture students and hobby gardeners who want a quick first opinion on a leaf photo.
* Anyone evaluating how a small CNN behaves on leaf images before building something larger.

## High-Level Features

1. **Image processing**: upload a JPG or PNG, see the original and the 224 x 224 processed version, and get clear error messages for unsupported or corrupted files.
2. **Disease detection**: the fine tuned MobileNetV2 predicts the crop and condition, shows the confidence as a percentage, flags low confidence results and lists the top three candidate classes.
3. **Analysis and reporting**: every prediction in the session is listed in a table with counts of healthy and diseased leaves, average confidence, a bar chart of predicted classes and export buttons for a CSV file and a text report.
4. **Model evaluation view**: the test set metrics, per class scores, training curve and confusion matrix produced by the evaluation script are shown inside the app.

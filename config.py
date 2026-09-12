"""
Central configuration for the Plant Disease Detection project.

Every path is built relative to this file so the project runs from any
location without editing machine specific paths.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# Dataset locations. The raw PlantVillage folder is expected to contain one
# sub folder per class, for example data/plantvillage/Tomato___Late_blight.
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATASET_DIR = DATA_DIR / "plantvillage"
SPLIT_DIR = DATA_DIR / "splits"

# Where the trained weights and evaluation outputs are stored.
MODELS_DIR = PROJECT_ROOT / "models"
WEIGHTS_PATH = MODELS_DIR / "mobilenetv2_plant_disease.pt"
CLASS_NAMES_PATH = MODELS_DIR / "class_names.json"
METRICS_PATH = MODELS_DIR / "metrics.json"
HISTORY_PATH = MODELS_DIR / "training_history.json"
CONFUSION_MATRIX_PLOT = MODELS_DIR / "confusion_matrix.png"
CONFUSION_MATRIX_CSV = MODELS_DIR / "confusion_matrix.csv"
TRAINING_LOG_PATH = MODELS_DIR / "training.log"

# Image settings. MobileNetV2 was pretrained on 224x224 ImageNet images and
# the pretrained weights expect inputs normalised with the ImageNet mean and
# standard deviation, so the same values are used for training and inference.
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Accepted upload formats in the Streamlit app.
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
MAX_UPLOAD_MB = 10

# Reproducible data split. The remaining 0.15 is the held out test set.
RANDOM_SEED = 42
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15

# Training hyperparameters.
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 1e-4
NUM_WORKERS = 4

# Predictions below this probability are shown with a warning in the app.
LOW_CONFIDENCE_THRESHOLD = 0.60

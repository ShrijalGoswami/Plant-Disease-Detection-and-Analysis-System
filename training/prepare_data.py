"""
Step 1 of training: build the reproducible train / val / test split.

Usage (from the project root):
    python training/prepare_data.py

Reads data/plantvillage/<class>/<image> and writes three CSV files to
data/splits/. Run this once. Training and evaluation read the CSVs, they
never touch the raw folder structure directly, so the split cannot drift
between runs.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import RANDOM_SEED  # noqa: E402
from src.dataset import create_splits  # noqa: E402
from src.utils import set_seed  # noqa: E402


def main() -> None:
    set_seed(RANDOM_SEED)
    info = create_splits()
    print(f"Classes      : {len(info['class_names'])}")
    print(f"Total images : {info['num_images']}")
    print(f"Train        : {info['train']}")
    print(f"Validation   : {info['val']}")
    print(f"Test         : {info['test']}")


if __name__ == "__main__":
    main()

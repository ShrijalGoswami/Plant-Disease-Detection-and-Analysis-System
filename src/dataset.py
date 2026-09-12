"""
Dataset preparation for training.

Two responsibilities live here:

* building a reproducible train / validation / test split of the raw
  PlantVillage folder and saving it as CSV files, so that every training
  run and every evaluation uses exactly the same images
* a PyTorch Dataset that reads one of those CSV files, applies the shared
  preprocessing pipeline and (for the training split only) light data
  augmentation
"""

from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision.transforms import v2 as transforms

from config import PROJECT_ROOT, RANDOM_SEED, RAW_DATASET_DIR, SPLIT_DIR, TRAIN_FRACTION, VAL_FRACTION
from src.preprocessing import load_image, preprocess_image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def list_dataset_images(dataset_dir: Path = RAW_DATASET_DIR) -> pd.DataFrame:
    """
    Walk the dataset folder and return a table with one row per image.

    Each class is a sub folder. The class index is the position of the
    class name in alphabetical order, which is also the order stored in
    models/class_names.json after training.
    """
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {dataset_dir}. See README for download steps."
        )
    class_names = sorted(p.name for p in dataset_dir.iterdir() if p.is_dir())
    rows = []
    for label, class_name in enumerate(class_names):
        for image_path in (dataset_dir / class_name).iterdir():
            if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                # Paths are stored relative to the project root so the CSV
                # files work on any machine that has the dataset folder.
                relative = image_path.relative_to(PROJECT_ROOT).as_posix()
                rows.append({"path": relative, "class_name": class_name, "label": label})
    return pd.DataFrame(rows)


def create_splits(dataset_dir: Path = RAW_DATASET_DIR, split_dir: Path = SPLIT_DIR) -> dict:
    """
    Split the dataset into train, validation and test CSV files.

    The split is stratified so each class keeps the same proportion in all
    three sets, and it uses a fixed random seed so it can be recreated.
    An image only ever appears in one of the three files, which is what
    prevents test images from leaking into training.
    """
    table = list_dataset_images(dataset_dir)
    train_df, rest_df = train_test_split(
        table,
        train_size=TRAIN_FRACTION,
        stratify=table["label"],
        random_state=RANDOM_SEED,
    )
    # Split the remaining 30% evenly into validation and test.
    val_share = VAL_FRACTION / (1.0 - TRAIN_FRACTION)
    val_df, test_df = train_test_split(
        rest_df,
        train_size=val_share,
        stratify=rest_df["label"],
        random_state=RANDOM_SEED,
    )

    split_dir.mkdir(parents=True, exist_ok=True)
    for name, df in (("train", train_df), ("val", val_df), ("test", test_df)):
        df.sort_values("path").to_csv(split_dir / f"{name}.csv", index=False)

    class_names = sorted(table["class_name"].unique())
    return {
        "class_names": class_names,
        "num_images": len(table),
        "train": len(train_df),
        "val": len(val_df),
        "test": len(test_df),
    }


def load_split(name: str, split_dir: Path = SPLIT_DIR) -> pd.DataFrame:
    path = split_dir / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Split file {path} not found. Run training/prepare_data.py first.")
    return pd.read_csv(path)


# Augmentation for the training split only. Leaves can be photographed from
# any angle, so flips and rotations produce realistic new samples. The small
# colour jitter simulates different lighting. Validation and test images are
# never augmented, otherwise the reported metrics would not reflect real use.
train_augmentation = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    ]
)


class LeafDataset(Dataset):
    """Reads images listed in a split CSV and returns (tensor, label) pairs."""

    def __init__(self, split_name: str, augment: bool = False):
        self.table = load_split(split_name)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, index: int):
        row = self.table.iloc[index]
        image = load_image(PROJECT_ROOT / row["path"])
        if self.augment:
            # Augment on the uint8 image before normalisation so that the
            # colour jitter operates on real pixel intensities.
            tensor = torch.from_numpy(image).permute(2, 0, 1)
            tensor = train_augmentation(tensor)
            image = tensor.permute(1, 2, 0).numpy()
        return preprocess_image(image), int(row["label"])

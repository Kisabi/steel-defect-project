"""
PyTorch Dataset for the Severstal Steel Defect Detection dataset.

Reads image/mask pairs from a manifest CSV (produced by
scripts/preprocess_severstal.py) and applies albumentations transforms.

Usage:
    from src.datasets.severstal_dataset import SeverstalDataset, get_transforms

    train_ds = SeverstalDataset(
        manifest_csv="data/splits/train.csv",
        raw_images_dir="data/raw/severstal-steel-defect-detection",
        processed_dir="data/processed",
        transform=get_transforms(split="train"),
    )
"""

from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

# ImageNet mean/std — required because the encoder (ResNet34) is pretrained on ImageNet
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_transforms(split: str) -> A.Compose:
    """Returns an albumentations pipeline for the given split.

    Only "train" gets stochastic augmentations — val/test must stay
    deterministic so metrics are comparable across experiments.
    """
    if split == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.OneOf(
                    [
                        A.GaussNoise(p=1.0),
                        A.GaussianBlur(p=1.0),
                    ],
                    p=0.2,
                ),
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2(),
            ]
        )
    elif split in ("val", "test"):
        return A.Compose(
            [
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2(),
            ]
        )
    else:
        raise ValueError(f"Unknown split: {split}. Expected 'train', 'val' or 'test'.")


class SeverstalDataset(Dataset):
    """Loads (image, mask) pairs described in a manifest CSV.

    The manifest is expected to have columns: image_id, image_path,
    mask_path, has_defect — exactly what preprocess_severstal.py produces.
    image_path/mask_path are relative to raw_images_dir/processed_dir
    respectively.
    """

    def __init__(
        self,
        manifest_csv: str,
        raw_images_dir: str,
        processed_dir: str,
        transform: A.Compose = None,
    ):
        self.manifest = pd.read_csv(manifest_csv)
        self.raw_images_dir = Path(raw_images_dir)
        self.processed_dir = Path(processed_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int):
        row = self.manifest.iloc[idx]

        image_path = self.raw_images_dir / row["image_path"]
        mask_path = self.processed_dir / row["mask_path"]

        # cv2 reads BGR by default — convert to RGB before augmentation/normalization
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # mask stores class ids 0-4 as raw pixel values — read as grayscale, no color conversion
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # CrossEntropyLoss expects a LongTensor of class indices, shape (H, W)
        mask = mask.long() if isinstance(mask, torch.Tensor) else torch.as_tensor(mask, dtype=torch.long)

        return image, mask


if __name__ == "__main__":
    # Quick sanity check — run inside the container:
    #   python -m src.datasets.severstal_dataset
    dataset = SeverstalDataset(
        manifest_csv="data/splits/train.csv",
        raw_images_dir="data/raw/severstal-steel-defect-detection",
        processed_dir="data/processed",
        transform=get_transforms(split="train"),
    )

    print(f"Dataset size: {len(dataset)}")
    image, mask = dataset[0]
    print(f"Image shape/dtype: {image.shape}, {image.dtype}")
    print(f"Mask shape/dtype: {mask.shape}, {mask.dtype}")
    print(f"Unique mask values in sample: {torch.unique(mask).tolist()}")
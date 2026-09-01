"""
Dataset for Pix2Pix: paired (multi-channel mask, real image) samples.

Source masks are single-channel 'L' PNGs with pixel values 0-4
(0 = background, 1-4 = defect class id), produced by
scripts/preprocess_severstal.py. Where classes overlap in the source
data, the higher class id wins (see preprocessing script docstring) -
this dataset inherits that same simplification, so the GAN target is
consistent with what the segmentation baseline was trained on.

For Pix2Pix conditioning we expand the label map into a 4-channel
binary one-hot tensor, one channel per defect class (1..4), matching
the "multi-channel mask" conditioning strategy decided for this stage.

Class oversampling: this dataset also exposes helpers to build a
WeightedRandomSampler that boosts the sampling frequency of images
containing a chosen target class, calibrated to a target per-epoch
fraction rather than a fixed multiplier - this keeps the oversampling
strength an explicit, reportable knob for the thesis (useful for
ablations).

Currently targeting class_2 (the genuinely rare class - 3.7% of defect
images, 0.5% of defect pixel area; see progress_log.md). Note that
class_2 has only ~247 unique images in the whole dataset, so aggressive
oversampling (e.g. the target_fraction=0.25 originally used for the
class_3 investigation) would repeat a small pool of ~170 train images
8x+ per epoch - a real mode-collapse/memorization risk for a GAN, not
just a segmentation-style class-imbalance concern. Default
target_fraction is kept moderate (0.15, ~4.6x weight) for this reason;
treat it as a hyperparameter worth an ablation, not a fixed choice.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler

IMG_HEIGHT = 256
IMG_WIDTH = 1600
NUM_CLASSES = 4
DEFAULT_TARGET_CLASS_ID = 2  # class_2: genuinely rare class, see progress_log.md


class Pix2PixSteelDataset(Dataset):
    """Paired (mask, image) dataset for Pix2Pix training.

    Args:
        split_csv: path to a splits/{train,val,test}.csv file
            (columns: image_id, image_path, mask_path, has_defect).
        images_root: root dir that image_path is relative to
            (e.g. data/raw/severstal-steel-defect-detection).
        masks_root: root dir that mask_path is relative to
            (e.g. data/processed).
        horizontal_flip_prob: probability of a joint horizontal flip
            applied identically to image and mask. Set to 0 to disable.
    """

    def __init__(
        self,
        split_csv: str | Path,
        images_root: str | Path,
        masks_root: str | Path,
        horizontal_flip_prob: float = 0.5,
    ):
        self.df = pd.read_csv(split_csv).reset_index(drop=True)
        self.images_root = Path(images_root)
        self.masks_root = Path(masks_root)
        self.horizontal_flip_prob = horizontal_flip_prob

    def __len__(self) -> int:
        return len(self.df)

    def _load_mask_onehot(self, mask_path: Path) -> np.ndarray:
        """Load a 0-4 label map PNG and expand it to a (H, W, 4) binary array."""
        label_map = np.array(Image.open(mask_path))  # (H, W), values 0..4
        channels = [(label_map == c).astype(np.float32) for c in range(1, NUM_CLASSES + 1)]
        return np.stack(channels, axis=-1)  # (H, W, 4)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        img_path = self.images_root / row["image_path"]
        mask_path = self.masks_root / row["mask_path"]

        image = np.array(Image.open(img_path).convert("RGB")).astype(np.float32)
        mask = self._load_mask_onehot(mask_path)  # (H, W, 4)

        if np.random.rand() < self.horizontal_flip_prob:
            image = np.ascontiguousarray(image[:, ::-1, :])
            mask = np.ascontiguousarray(mask[:, ::-1, :])

        # image -> [-1, 1] for tanh generator output convention
        image = (image / 127.5) - 1.0

        image_t = torch.from_numpy(image).permute(2, 0, 1).float()  # (3, H, W)
        mask_t = torch.from_numpy(mask).permute(2, 0, 1).float()  # (4, H, W)

        return {
            "mask": mask_t,
            "image": image_t,
            "image_id": row["image_id"],
        }


def get_class_image_ids(raw_train_csv: str | Path, class_id: int = DEFAULT_TARGET_CLASS_ID) -> set[str]:
    """Return the set of image_ids that contain at least one defect of `class_id`.

    Reads the original Severstal train.csv (ImageId, ClassId, EncodedPixels)
    directly, which is far cheaper than re-scanning every mask PNG.
    """
    df = pd.read_csv(raw_train_csv)
    matches = df[(df["ClassId"] == class_id) & df["EncodedPixels"].notna()]
    return set(matches["ImageId"].unique())


def build_class_weighted_sampler(
    dataset: Pix2PixSteelDataset,
    target_image_ids: set[str],
    target_fraction: float = 0.15,
    class_label: str = "class_2",
) -> WeightedRandomSampler:
    """Build a WeightedRandomSampler calibrated to a target class fraction.

    Rather than an arbitrary oversampling multiplier, this solves for the
    per-sample weight w such that, in expectation, `target_fraction` of the
    samples drawn per epoch contain the target class. Other samples keep
    weight 1.0.

    Default target_fraction=0.15 is deliberately moderate for class_2:
    it's natural frequency is only ~3.7% of images, and with only ~247
    unique images containing it dataset-wide, an aggressive target (e.g.
    0.25, used earlier for the much more common class_3) would push the
    weight past 8x - repeating a small pool of images heavily enough to
    risk the GAN memorizing specific instances rather than generalizing
    the defect pattern. Tune this as an explicit, reportable ablation
    knob rather than assuming a single "correct" value.

    Args:
        dataset: the Pix2PixSteelDataset (or a Subset of it) to sample from.
        target_image_ids: output of get_class_image_ids().
        target_fraction: desired expected fraction of target-class samples
            per epoch, e.g. 0.15 means ~1 in ~6-7 samples will contain it.
        class_label: human-readable label used only in the printed log line.

    Returns:
        A WeightedRandomSampler with replacement=True and
        num_samples=len(dataset), ready to pass to a DataLoader.
    """
    image_ids = dataset.df["image_id"].tolist()
    is_target = np.array([img_id in target_image_ids for img_id in image_ids])

    n_total = len(is_target)
    n_target = int(is_target.sum())
    n_other = n_total - n_target

    if n_target == 0:
        raise ValueError(f"No {class_label} samples found in this split - check target_image_ids.")
    if not 0.0 < target_fraction < 1.0:
        raise ValueError("target_fraction must be strictly between 0 and 1.")

    # solve w * n_target / (w * n_target + n_other) = target_fraction  for w
    w_target = (target_fraction * n_other) / (n_target * (1.0 - target_fraction))

    weights = np.where(is_target, w_target, 1.0)

    print(
        f"[{class_label} sampler] n_total={n_total} n_target={n_target} ({n_target / n_total:.1%} natural) "
        f"-> weight={w_target:.2f} targeting {target_fraction:.0%} per epoch"
    )

    return WeightedRandomSampler(
        weights=torch.from_numpy(weights).double(),
        num_samples=n_total,
        replacement=True,
    )
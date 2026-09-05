from src.gans.pix2pix.dataset import (
    Pix2PixSteelDataset, get_class_image_ids, build_class_weighted_sampler
)
from torch.utils.data import DataLoader

train_ds = Pix2PixSteelDataset(
    split_csv="data/splits/train.csv",
    images_root="data/raw/severstal-steel-defect-detection",
    masks_root="data/processed",
)

class2_ids = get_class_image_ids(
    "data/raw/severstal-steel-defect-detection/train.csv", class_id=2
)
sampler = build_class_weighted_sampler(
    train_ds, class2_ids, target_fraction=0.15, class_label="class_2"
)
loader = DataLoader(train_ds, batch_size=2, sampler=sampler, num_workers=2)

batch = next(iter(loader))
print("image:", batch["image"].shape, batch["image"].min().item(), batch["image"].max().item())
print("mask :", batch["mask"].shape, batch["mask"].min().item(), batch["mask"].max().item())
print("max overlap per pixel:", batch["mask"].sum(dim=1).max().item())  # должно быть <= 1
print("image_ids:", batch["image_id"])
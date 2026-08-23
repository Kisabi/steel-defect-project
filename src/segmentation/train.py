"""
Training script for the baseline segmentation model: U-Net (ResNet34 encoder).

Trains on a manifest produced by scripts/preprocess_severstal.py, logs
metrics/params/checkpoints to MLflow, and saves the best checkpoint by
validation mIoU.

Usage (inside the container, from /workspace):
    python -m src.segmentation.train \
        --epochs 30 \
        --batch-size 16 \
        --lr 1e-4

If you hit CUDA out of memory, lower --batch-size (e.g. 8) or increase
--accumulation-steps to simulate a larger effective batch without more VRAM.
"""

import argparse
from pathlib import Path

import mlflow
import segmentation_models_pytorch as smp
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.datasets.severstal_dataset import SeverstalDataset, get_transforms

NUM_CLASSES = 5  # background (0) + 4 defect classes (1-4)


def build_model() -> nn.Module:
    """U-Net with a ResNet34 encoder pretrained on ImageNet."""
    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=NUM_CLASSES,
    )


def build_loss():
    """Combined Dice + CrossEntropy loss.

    CrossEntropy gives a stable gradient signal from the start of training;
    Dice directly optimizes overlap with the ground-truth mask and helps
    counter the heavy background/defect class imbalance. Summing both is a
    common, robust choice for segmentation with imbalanced classes.
    """
    dice = smp.losses.DiceLoss(mode="multiclass")
    ce = nn.CrossEntropyLoss()

    def loss_fn(logits, target):
        return dice(logits, target) + ce(logits, target)

    return loss_fn


def compute_miou(logits: torch.Tensor, target: torch.Tensor) -> float:
    """Mean IoU across all classes (macro-imagewise), including background."""
    preds = logits.argmax(dim=1)
    tp, fp, fn, tn = smp.metrics.get_stats(
        preds, target, mode="multiclass", num_classes=NUM_CLASSES
    )
    return smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro-imagewise").item()


def train_one_epoch(model, loader, loss_fn, optimizer, scaler, device, accumulation_steps):
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()

    for step, (images, masks) in enumerate(loader):
        images, masks = images.to(device), masks.to(device)

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(images)
            loss = loss_fn(logits, masks) / accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        running_loss += loss.item() * accumulation_steps

    return running_loss / len(loader)


@torch.no_grad()
def validate(model, loader, loss_fn, device):
    model.eval()
    running_loss = 0.0
    running_miou = 0.0

    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(images)
            loss = loss_fn(logits, masks)

        running_loss += loss.item()
        running_miou += compute_miou(logits, masks)

    n = len(loader)
    return running_loss / n, running_miou / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--splits-dir", type=str, default="data/splits")
    parser.add_argument("--raw-images-dir", type=str, default="data/raw/severstal-steel-defect-detection")
    parser.add_argument("--processed-dir", type=str, default="data/processed")
    parser.add_argument("--checkpoint-dir", type=str, default="experiments/checkpoints/unet_baseline")
    parser.add_argument("--experiment-name", type=str, default="unet-baseline")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_ds = SeverstalDataset(
        manifest_csv=f"{args.splits_dir}/train.csv",
        raw_images_dir=args.raw_images_dir,
        processed_dir=args.processed_dir,
        transform=get_transforms(split="train"),
    )
    val_ds = SeverstalDataset(
        manifest_csv=f"{args.splits_dir}/val.csv",
        raw_images_dir=args.raw_images_dir,
        processed_dir=args.processed_dir,
        transform=get_transforms(split="val"),
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    model = build_model().to(device)
    loss_fn = build_loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )
    scaler = torch.amp.GradScaler("cuda")

    mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run():
        mlflow.log_params(vars(args))

        best_miou = 0.0
        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(
                model, train_loader, loss_fn, optimizer, scaler, device, args.accumulation_steps
            )
            val_loss, val_miou = validate(model, val_loader, loss_fn, device)
            scheduler.step(val_miou)

            print(
                f"Epoch {epoch}/{args.epochs} | "
                f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_mIoU={val_miou:.4f}"
            )

            mlflow.log_metrics(
                {"train_loss": train_loss, "val_loss": val_loss, "val_miou": val_miou},
                step=epoch,
            )

            if val_miou > best_miou:
                best_miou = val_miou
                best_path = checkpoint_dir / "best.pth"
                torch.save(model.state_dict(), best_path)
                mlflow.log_artifact(str(best_path))
                print(f"  New best model saved (val_mIoU={best_miou:.4f})")

        last_path = checkpoint_dir / "last.pth"
        torch.save(model.state_dict(), last_path)
        mlflow.log_artifact(str(last_path))
        mlflow.log_metric("best_val_miou", best_miou)

    print(f"Training finished. Best val_mIoU: {best_miou:.4f}")


if __name__ == "__main__":
    main()
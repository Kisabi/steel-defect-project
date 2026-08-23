"""
Evaluates a trained segmentation checkpoint on the held-out test split.

The test split is never used during training or checkpoint selection, so
this gives an unbiased estimate of model quality — the number that should
actually be reported/compared across experiments (baseline vs. every
synthetic-data ratio for every GAN model).

Usage (inside the container, from /workspace):
    python -m src.segmentation.evaluate \
        --checkpoint experiments/checkpoints/unet_baseline/best.pth
"""

import argparse

import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader

from src.datasets.severstal_dataset import SeverstalDataset, get_transforms
from src.segmentation.train import NUM_CLASSES, build_model, compute_miou


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()

    total_miou = 0.0
    per_class_iou_sum = torch.zeros(NUM_CLASSES)
    n_batches = 0

    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(images)

        preds = logits.argmax(dim=1)
        tp, fp, fn, tn = smp.metrics.get_stats(
            preds, masks, mode="multiclass", num_classes=NUM_CLASSES
        )

        total_miou += smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro-imagewise").item()
        per_class_iou_sum += smp.metrics.iou_score(tp, fp, fn, tn, reduction="none").mean(dim=0).cpu()
        n_batches += 1

    return {
        "mean_iou": total_miou / n_batches,
        "per_class_iou": (per_class_iou_sum / n_batches).tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--splits-dir", type=str, default="data/splits")
    parser.add_argument("--raw-images-dir", type=str, default="data/raw/severstal-steel-defect-detection")
    parser.add_argument("--processed-dir", type=str, default="data/processed")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_ds = SeverstalDataset(
        manifest_csv=f"{args.splits_dir}/test.csv",
        raw_images_dir=args.raw_images_dir,
        processed_dir=args.processed_dir,
        transform=get_transforms(split="test"),
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    model = build_model().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    print(f"Loaded checkpoint: {args.checkpoint}")

    results = evaluate(model, test_loader, device)

    print(f"\nTest mIoU (mean over all classes): {results['mean_iou']:.4f}")
    class_names = ["background", "class_1", "class_2", "class_3", "class_4"]
    for name, iou in zip(class_names, results["per_class_iou"]):
        print(f"  {name}: {iou:.4f}")


if __name__ == "__main__":
    main()
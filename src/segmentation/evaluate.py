"""
Evaluates a trained segmentation checkpoint on the held-out test split.

The test split is never used during training or checkpoint selection, so
this gives an unbiased estimate of model quality - the number that should
actually be reported/compared across experiments (baseline vs. every
synthetic-data ratio for every GAN model).

IMPORTANT - metric fix (see analyze_errors.py investigation):
The original version of this script computed per-class IoU with
reduction="none" per batch, then averaged those per-image ratios across
images. Because smp.metrics.iou_score defaults to zero_division=1, any
image that doesn't contain a given class (and wasn't falsely predicted
as that class) scores a trivial IoU=1.0 for it. For rare classes
(class_1: 13.5% of images, class_2: 3.7%) this massively inflates the
reported per-image-averaged IoU - it mostly measures "correctly predicted
absence", not real segmentation quality.

This version instead accumulates tp/fp/fn/tn across the ENTIRE test set
first, then computes IoU once from the totals ("sum then divide" instead
of "divide then average") - this is the standard dataset-level per-class
IoU convention (same as building a full confusion matrix and reading IoU
off the diagonal, which is what analyze_errors.py does independently as
a cross-check). Both the corrected per-class numbers and the original
macro-imagewise numbers are printed, labeled, for transparency.

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

CLASS_NAMES = ["background", "class_1", "class_2", "class_3", "class_4"]


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()

    # legacy metric, kept for transparency/comparison with earlier results
    total_miou_imagewise = 0.0
    per_class_iou_imagewise_sum = torch.zeros(NUM_CLASSES)
    n_batches = 0

    # corrected metric: accumulate raw counts across the whole test set first
    tp_total = torch.zeros(NUM_CLASSES, dtype=torch.int64)
    fp_total = torch.zeros(NUM_CLASSES, dtype=torch.int64)
    fn_total = torch.zeros(NUM_CLASSES, dtype=torch.int64)
    tn_total = torch.zeros(NUM_CLASSES, dtype=torch.int64)

    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(images)

        preds = logits.argmax(dim=1)
        tp, fp, fn, tn = smp.metrics.get_stats(
            preds, masks, mode="multiclass", num_classes=NUM_CLASSES
        )

        # legacy: per-image ratios averaged afterwards (inflated for rare classes)
        total_miou_imagewise += smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro-imagewise").item()
        per_class_iou_imagewise_sum += smp.metrics.iou_score(tp, fp, fn, tn, reduction="none").mean(dim=0).cpu()
        n_batches += 1

        # corrected: accumulate raw pixel counts, summed over the batch's images
        tp_total += tp.sum(dim=0).cpu()
        fp_total += fp.sum(dim=0).cpu()
        fn_total += fn.sum(dim=0).cpu()
        tn_total += tn.sum(dim=0).cpu()

    # single division on dataset-total counts -> honest per-class IoU
    corrected_per_class_iou = smp.metrics.iou_score(
        tp_total.unsqueeze(0), fp_total.unsqueeze(0), fn_total.unsqueeze(0), tn_total.unsqueeze(0),
        reduction=None,
    ).squeeze(0)
    corrected_miou = corrected_per_class_iou.mean().item()

    return {
        "mean_iou_imagewise_legacy": total_miou_imagewise / n_batches,
        "per_class_iou_imagewise_legacy": (per_class_iou_imagewise_sum / n_batches).tolist(),
        "mean_iou_corrected": corrected_miou,
        "per_class_iou_corrected": corrected_per_class_iou.tolist(),
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

    print(f"\n=== CORRECTED (dataset-level, not inflated by rare-class trivial negatives) ===")
    print(f"Test mIoU: {results['mean_iou_corrected']:.4f}")
    for name, iou in zip(CLASS_NAMES, results["per_class_iou_corrected"]):
        print(f"  {name}: {iou:.4f}")

    print(f"\n=== legacy macro-imagewise (kept for comparison - inflated for rare classes) ===")
    print(f"Test mIoU: {results['mean_iou_imagewise_legacy']:.4f}")
    for name, iou in zip(CLASS_NAMES, results["per_class_iou_imagewise_legacy"]):
        print(f"  {name}: {iou:.4f}")


if __name__ == "__main__":
    main()
"""
Confusion matrix and per-class error analysis for the baseline U-Net checkpoint.

Motivation: analyze_class_distribution.py showed class_3 is the dominant
class by both image presence (77.3%) and pixel area (80.3%), which rules
out "not enough data" as the explanation for its weak test mIoU (0.61 vs
0.9+ for other classes). This script builds a confusion matrix to check
what class_3 pixels actually get misclassified as - systematic confusion
with a specific class (e.g. class_4, background) points to a different
root cause than random/uniform error.

Reuses build_model, SeverstalDataset and get_transforms from the existing
training/eval code so preprocessing stays identical to what the model was
trained and evaluated with.

Usage (inside the container, from /workspace):
    python -m src.segmentation.analyze_errors \
        --checkpoint experiments/checkpoints/unet_baseline/best.pth
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.datasets.severstal_dataset import SeverstalDataset, get_transforms
from src.segmentation.train import NUM_CLASSES, build_model

CLASS_NAMES = ["background", "class_1", "class_2", "class_3", "class_4"]


@torch.no_grad()
def accumulate_confusion_matrix(model, loader, device, num_classes) -> torch.Tensor:
    """Accumulate a (num_classes, num_classes) confusion matrix over the whole loader.

    Rows = ground truth class, columns = predicted class. Uses a bincount
    trick instead of sklearn so it stays fast on full-resolution
    (256x1600) masks without materializing huge flattened arrays repeatedly.
    """
    conf = torch.zeros(num_classes * num_classes, dtype=torch.int64)

    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(images)
        preds = logits.argmax(dim=1)

        idx = masks.flatten() * num_classes + preds.flatten()
        conf += torch.bincount(idx.cpu(), minlength=num_classes * num_classes)

    return conf.reshape(num_classes, num_classes)


def per_class_metrics(conf: torch.Tensor) -> dict:
    """Precision, recall (=IoU numerator context) and F1 per class from the confusion matrix."""
    conf = conf.float()
    tp = conf.diag()
    fp = conf.sum(dim=0) - tp  # predicted as class c, but wasn't
    fn = conf.sum(dim=1) - tp  # was class c, but predicted otherwise

    precision = tp / (tp + fp).clamp(min=1)
    recall = tp / (tp + fn).clamp(min=1)
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-8)

    return {
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
    }


def print_confusion_report(conf: torch.Tensor):
    num_classes = conf.shape[0]
    row_normalized = conf.float() / conf.sum(dim=1, keepdim=True).clamp(min=1)

    print("\n=== Confusion matrix (row-normalized: share of each true class's pixels) ===")
    header = "true\\pred".ljust(12) + "".join(name.ljust(12) for name in CLASS_NAMES)
    print(header)
    for i in range(num_classes):
        row = CLASS_NAMES[i].ljust(12)
        row += "".join(f"{row_normalized[i, j].item():.1%}".ljust(12) for j in range(num_classes))
        print(row)

    print("\n=== Per-class precision / recall / F1 ===")
    metrics = per_class_metrics(conf)
    for i, name in enumerate(CLASS_NAMES):
        print(
            f"  {name}: precision={metrics['precision'][i]:.3f}  "
            f"recall={metrics['recall'][i]:.3f}  f1={metrics['f1'][i]:.3f}"
        )

    print("\n=== Where does each class's error mass go? (excluding correct predictions) ===")
    for i in range(num_classes):
        total_errors = row_normalized[i].sum() - row_normalized[i, i]
        if total_errors <= 0:
            continue
        print(f"  {CLASS_NAMES[i]} errors go to:")
        errs = [(CLASS_NAMES[j], row_normalized[i, j].item()) for j in range(num_classes) if j != i]
        for name, share in sorted(errs, key=lambda x: -x[1]):
            if share > 0.001:
                print(f"    -> {name}: {share:.1%} of {CLASS_NAMES[i]}'s pixels")


def save_confusion_csv(conf: torch.Tensor, out_path: Path):
    import csv

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred"] + CLASS_NAMES)
        for i, name in enumerate(CLASS_NAMES):
            writer.writerow([name] + conf[i].tolist())
    print(f"\nRaw confusion matrix saved to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--splits-dir", type=str, default="data/splits")
    parser.add_argument("--raw-images-dir", type=str, default="data/raw/severstal-steel-defect-detection")
    parser.add_argument("--processed-dir", type=str, default="data/processed")
    parser.add_argument("--output-csv", type=str, default="experiments/analysis/confusion_matrix.csv")
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

    conf = accumulate_confusion_matrix(model, test_loader, device, NUM_CLASSES)
    print_confusion_report(conf)
    save_confusion_csv(conf, Path(args.output_csv))


if __name__ == "__main__":
    main()
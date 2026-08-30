"""
Quick per-class distribution check: image-level presence vs pixel-level area.

Motivation: the Pix2Pix class_3 sampler showed class_3 present in ~78% of
defect train images - the opposite of "rare class" at the image level.
Before deciding on any oversampling scheme (or reframing the thesis
narrative around class_3), we need to know whether the imbalance is at
the pixel level instead (e.g. class_3 defects are common but small),
or whether class_3 is genuinely well-represented at both levels and the
weak U-Net performance comes from something else (visual ambiguity,
confusion with other classes, boundary irregularity).

Run:
    python scripts/analyze_class_distribution.py
"""

import pandas as pd

RAW_TRAIN_CSV = "data/raw/severstal-steel-defect-detection/train.csv"
IMG_HEIGHT = 256
IMG_WIDTH = 1600


def rle_area(rle_string) -> int:
    """Total pixel count encoded by an RLE string (sum of run lengths)."""
    if not isinstance(rle_string, str) or rle_string.strip() == "":
        return 0
    parts = rle_string.strip().split()
    lengths = [int(x) for x in parts[1::2]]
    return sum(lengths)


def main():
    df = pd.read_csv(RAW_TRAIN_CSV)
    df = df.dropna(subset=["EncodedPixels"]).copy()
    df["area"] = df["EncodedPixels"].apply(rle_area)

    total_pixels_per_image = IMG_HEIGHT * IMG_WIDTH
    n_defect_images = df["ImageId"].nunique()

    print("=== Image-level presence (share of defect images containing class) ===")
    for class_id in range(1, 5):
        n_images = df[df["ClassId"] == class_id]["ImageId"].nunique()
        print(f"class_{class_id}: {n_images}/{n_defect_images} images ({n_images / n_defect_images:.1%})")

    print()
    print("=== Pixel-level area (share of total defect pixels belonging to class) ===")
    total_area = df["area"].sum()
    for class_id in range(1, 5):
        class_df = df[df["ClassId"] == class_id]
        class_area = class_df["area"].sum()
        avg_area = class_df["area"].mean()
        print(
            f"class_{class_id}: {class_area}/{total_area} px total ({class_area / total_area:.1%}), "
            f"avg instance area: {avg_area:.0f} px "
            f"({avg_area / total_pixels_per_image:.2%} of a full image)"
        )


if __name__ == "__main__":
    main()
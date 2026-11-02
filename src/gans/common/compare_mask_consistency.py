"""
Paired comparison of two mask_consistency runs (e.g. StyleGAN2-ADA epoch 50 vs 55 on val).

Both runs must be on the same split, with the same --seed and --samples-per-mask,
so that per-mask results are paired. The bootstrap resamples MASKS (the same
resampled indices for both runs) and reports the distribution of the difference
A - B. If the 95% CI contains 0, the two checkpoints are not distinguishable by
this metric on this split - pick by another criterion (FID, visual quality).

Usage:
    python -m src.gans.common.compare_mask_consistency \
        experiments/mask_consistency/stylegan2_ada/generator_epoch0050_val.npz \
        experiments/mask_consistency/stylegan2_ada/generator_epoch0055_val.npz
"""

import argparse

import numpy as np

from src.gans.common.mask_consistency import CLASS_NAMES, iou_from_conf


def paired_bootstrap(conf_a, conf_b, n_boot, rng, chunk=100):
    """Returns (B, 5) per-class IoU differences A-B plus (B,) mIoU(5) and mIoU(1-4) differences."""
    n = conf_a.shape[0]
    d_iou = []
    for start in range(0, n_boot, chunk):
        b = min(chunk, n_boot - start)
        idx = rng.integers(0, n, size=(b, n))
        d_iou.append(iou_from_conf(conf_a[idx].sum(1)) - iou_from_conf(conf_b[idx].sum(1)))
    d_iou = np.concatenate(d_iou, axis=0)
    return d_iou, np.nanmean(d_iou, axis=1), np.nanmean(d_iou[:, 1:], axis=1)


def report(label, a, b, n_boot, rng):
    observed = iou_from_conf(a.sum(0)) - iou_from_conf(b.sum(0))
    d_iou, d_miou, d_miou_def = paired_bootstrap(a, b, n_boot, rng)

    print(f"\n=== scope: {label} ({a.shape[0]} masks), difference A - B ===")
    print(f"{'metric':<12}{'observed':>10}   {'95% CI':<20}{'P(A>B)':>8}")

    def line(name, obs, samples):
        lo, hi = np.nanpercentile(samples, [2.5, 97.5])
        p = float(np.nanmean(samples > 0))
        flag = "" if lo <= 0 <= hi else "  *"
        print(f"{name:<12}{obs:>+10.4f}   [{lo:+.4f}, {hi:+.4f}]  {p:>8.2f}{flag}")

    for c, name in enumerate(CLASS_NAMES):
        line(name, observed[c], d_iou[:, c])
    line("mIoU (5)", float(np.nanmean(observed)), d_miou)
    line("mIoU (1-4)", float(np.nanmean(observed[1:])), d_miou_def)
    print("(* = 95% CI excludes 0)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_a", type=str, help=".npz written by mask_consistency (A)")
    parser.add_argument("run_b", type=str, help=".npz written by mask_consistency (B)")
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    A, B = np.load(args.run_a, allow_pickle=True), np.load(args.run_b, allow_pickle=True)
    if list(A["image_ids"]) != list(B["image_ids"]):
        raise ValueError("The two runs were made on different masks / order - cannot pair them.")

    rng = np.random.default_rng(args.seed)
    print(f"A = {args.run_a}\nB = {args.run_b}")
    report("overall", A["conf_fake"], B["conf_fake"], args.n_bootstrap, rng)
    is_class = A["is_class"]
    if is_class.any():
        report("class subset", A["conf_fake"][is_class], B["conf_fake"][is_class], args.n_bootstrap, rng)


if __name__ == "__main__":
    main()
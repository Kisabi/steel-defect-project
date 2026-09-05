"""
FID evaluation for the Pix2Pix generator: real vs. synthetic steel images.

Two FID scopes are reported, each with a raw (as-computed) value and a
bias-corrected value (Chong & Forsyth, 2020):

  - "overall": all test-split images, real vs. Pix2Pix output conditioned
    on the corresponding test masks (paired - one or more fake images per
    real image, using the *same* mask; see --samples-per-mask).
  - "class_<id>": the subset of test-split images that contain the target
    oversampling class (see progress_log.md), same paired methodology.

Why bias-corrected as well as raw
----------------------------------
FID computed from InceptionV3 features is a biased estimator whose bias
grows as sample size shrinks (Chong & Forsyth, "Effectively Unbiased FID
and Inception Score", 2020). The class_2 test subset is small (tens of
images, not the ~10k typically recommended for FID) - see
progress_log.md. Raw FID(N) is a decreasing function of 1/N; fitting a
line against 1/N over several subsample sizes and reading the intercept
(1/N -> 0) estimates the "true" FID at infinite sample size, removing
most of the small-N bias. The overall FID uses the same procedure for
consistency even though its sample size is less of a concern.

Feature extractor
------------------
clean-fid's InceptionV3 port (mode="clean"), which fixes several
inconsistencies in the original TF/PyTorch FID implementations - mainly
image resizing (Parmar et al., "On Aliased Resizing and Surprising
Subtleties in GAN Evaluation", CVPR 2022). Verified working under ROCm
before this script was written (feature shape (N, 2048) confirmed).

Generator inference
--------------------
Kept in `.train()` mode (not `.eval()`) so that Dropout(0.5) near the
bottleneck stays active, matching the Pix2Pix convention of dropout as
the only source of output stochasticity at inference time (see
generator.py docstring). InstanceNorm2d always uses per-instance
statistics regardless of train/eval mode (track_running_stats=False by
default), so `.train()` here only affects Dropout - no BatchNorm-style
running-stat leakage risk.

Assumptions worth checking before the first full run
------------------------------------------------------
- A held-out `data/splits/test.csv` exists with the same schema as
  train.csv/val.csv (image_id, image_path, mask_path, has_defect) and
  was NOT used during Pix2Pix training (train.py only touches
  train.csv/val.csv - test.csv should be untouched, but double check
  nothing else in the pipeline leaked it into training).
- `--checkpoint` points at the accepted 100-epoch run's generator
  (generator_last.pth by default).

Usage (inside the container, from /workspace):
    python -m src.gans.pix2pix.compute_fid \
        --checkpoint experiments/checkpoints/pix2pix/generator_last.pth \
        --class-target-id 2 \
        --samples-per-mask 5 \
        --stage final \
        --note "generator_variant=convtranspose_k4_no_icnr, accepted arch"

Quick dry run on a handful of images first (recommended before the full
test split - mirrors the shape-check habit used for generator.py):
    python -m src.gans.pix2pix.compute_fid --dry-run-limit 8 --samples-per-mask 2
"""

import argparse
import subprocess
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
from cleanfid.fid import build_feature_extractor, fid_from_feats, frechet_distance, get_files_features
from PIL import Image

from src.gans.pix2pix.dataset import get_class_image_ids
from src.gans.pix2pix.generator import build_generator

NUM_CLASSES = 4


def get_git_commit_hash() -> str:
    """Short git commit hash for the current checkout, for MLflow tags.
    Returns 'unknown' outside a git repo (e.g. a stripped-down container)
    rather than failing the run over a bookkeeping detail. Appends '-dirty'
    if there are uncommitted changes, since "which code produced this
    number" matters more once results start going into the thesis.
    """
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        dirty = subprocess.call(
            ["git", "diff", "--quiet"], stderr=subprocess.DEVNULL
        ) != 0
        return f"{commit}-dirty" if dirty else commit
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def load_test_split(splits_dir: Path) -> pd.DataFrame:
    test_csv = splits_dir / "test.csv"
    if not test_csv.exists():
        raise FileNotFoundError(
            f"{test_csv} not found. compute_fid.py assumes a held-out test "
            "split exists alongside train/val (see dataset.py docstring). "
            "If your project only has train/val, point --splits-dir at "
            "whatever CSV should stand in for it - but confirm it was not "
            "used during Pix2Pix training."
        )
    return pd.read_csv(test_csv).reset_index(drop=True)


def load_mask_onehot(mask_path: Path) -> torch.Tensor:
    """Same encoding as Pix2PixSteelDataset._load_mask_onehot, standalone
    (no augmentation - FID inputs must be deterministic per mask)."""
    label_map = np.array(Image.open(mask_path))  # (H, W), values 0..4
    channels = [(label_map == c).astype(np.float32) for c in range(1, NUM_CLASSES + 1)]
    mask = np.stack(channels, axis=-1)  # (H, W, 4)
    return torch.from_numpy(mask).permute(2, 0, 1).float()  # (4, H, W)


def denormalize_to_uint8(image_tensor: torch.Tensor) -> np.ndarray:
    """(3, H, W) in [-1, 1] -> (H, W, 3) uint8, for saving as PNG.
    Mirrors train.py's denormalize()."""
    image = (image_tensor.clamp(-1, 1) + 1) / 2 * 255.0
    return image.byte().permute(1, 2, 0).cpu().numpy()


@torch.no_grad()
def generate_fakes(
    generator: torch.nn.Module,
    df: pd.DataFrame,
    masks_root: Path,
    out_dir: Path,
    device: torch.device,
    samples_per_mask: int,
) -> list[str]:
    """Run the generator on each mask in `df`, `samples_per_mask` times
    each (Dropout gives a different stochastic sample per call, inflating
    the fake-side pool for the bias-correction step without touching the
    real-side pool - see module docstring). Saves PNGs to `out_dir` and
    returns the list of saved file paths.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    generator.train()  # keep dropout active - see module docstring
    saved_paths = []

    for _, row in df.iterrows():
        mask = load_mask_onehot(masks_root / row["mask_path"]).unsqueeze(0).to(device)
        for k in range(samples_per_mask):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                fake = generator(mask)
            fake_img = denormalize_to_uint8(fake[0].float())
            out_path = out_dir / f"{row['image_id']}_s{k}.png"
            Image.fromarray(fake_img).save(out_path)
            saved_paths.append(str(out_path))

    return saved_paths


def compute_raw_fid(
    real_paths: list[str], fake_paths: list[str], fx, device
) -> tuple[float, np.ndarray, np.ndarray]:
    real_feats = get_files_features(real_paths, model=fx, device=device, mode="clean", verbose=True)
    fake_feats = get_files_features(fake_paths, model=fx, device=device, mode="clean", verbose=True)
    fid = fid_from_feats(real_feats, fake_feats)
    return float(fid), real_feats, fake_feats


def _sqrt_psd(matrix: np.ndarray) -> np.ndarray:
    """Matrix square root of a symmetric PSD matrix via eigendecomposition."""
    eigvals, eigvecs = np.linalg.eigh(matrix)
    eigvals = np.clip(eigvals, 0, None)  # numerical noise can produce tiny negatives
    return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T


def fast_frechet_distance(mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray, eps: float = 1e-6) -> float:
    """Numerically equivalent to cleanfid.fid.frechet_distance, but computes
    trace(sqrtm(sigma1 @ sigma2)) via the identity eig(A @ B) == eig(A^0.5 @ B @ A^0.5)
    for symmetric PSD A, B - this only needs eigh()/eigvalsh() (fast, stable)
    instead of scipy.linalg.sqrtm's general Schur algorithm.

    This matters here specifically because bias_corrected_fid() calls a
    Frechet-distance computation n_sizes * n_repeats times per scope (160
    by default) on the full 2048x2048 InceptionV3 covariance matrices -
    scipy.linalg.sqrtm at that size, called that many times, is what was
    pegging the CPU for a long time on the "overall" scope (N up to 1000,
    but the matrix size - and thus the cost per call - doesn't depend on
    N, only on the 2048-d feature dimensionality). The single frechet
    distance call inside compute_raw_fid() (via cleanfid's fid_from_feats)
    stays on the original scipy implementation - it only runs once per
    scope, so it isn't the bottleneck and there's no reason to touch it.
    """
    diff = mu1 - mu2
    sigma1 = sigma1 + np.eye(sigma1.shape[0]) * eps
    sigma2 = sigma2 + np.eye(sigma2.shape[0]) * eps

    sigma1_sqrt = _sqrt_psd(sigma1)
    cross = sigma1_sqrt @ sigma2 @ sigma1_sqrt
    cross_eigvals = np.clip(np.linalg.eigvalsh(cross), 0, None)
    tr_covmean = np.sum(np.sqrt(cross_eigvals))

    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


def bias_corrected_fid(
    real_feats: np.ndarray,
    fake_feats: np.ndarray,
    n_sizes: int = 8,
    n_repeats: int = 20,
    min_size: int = 20,
    seed: int = 0,
) -> dict:
    """Chong & Forsyth (2020) extrapolation, applied only to the REAL sample
    size. FID(n_real) is regressed against 1/n_real over several subsample
    sizes, and the intercept (n_real -> infinity) is the bias-corrected
    estimate. The fake side is held fixed at its FULL available pool
    (mu_f/sigma_f computed once from all of fake_feats, reused for every
    real subsample and repeat) - it is not subsampled down to match n_real.

    This matters because generate_fakes() intentionally produces more fake
    samples than there are real images (--samples-per-mask > 1, since
    Dropout-based stochastic sampling is cheap and unlimited, unlike real
    steel defect photos - see progress_log.md re: class_2's scarcity being
    the actual bottleneck). raw FID (see compute_raw_fid) already uses the
    full asymmetric real/fake counts; an earlier version of this function
    instead capped BOTH sides down to a shared N = min(n_real, n_fake) at
    every grid point, discarding most of the fake-side sample-size
    advantage - that produced bias-corrected estimates *higher* than raw
    FID on the "overall" scope (87-89 vs raw ~80), the wrong direction,
    since a larger effective sample (via the fixed full fake pool) should
    only ever reduce estimated bias, not increase it. Fixing the fake side
    at its full pool resolves this: the largest real-subsample point
    (n_real = full real count) now uses the *same* fake statistics as raw
    FID, so it should closely reproduce raw_fid, and the extrapolated
    curve trends the expected direction (bias decreasing as n_real grows).
    """
    rng = np.random.default_rng(seed)
    max_real = len(real_feats)
    if max_real < min_size:
        raise ValueError(
            f"Only {max_real} real samples (need >= {min_size}) for "
            "bias-corrected FID - see progress_log.md re: class_2 sample size."
        )

    # fake side fixed at its full pool - computed once, reused across every
    # real-subsample size and repeat (matches how raw FID uses the full
    # fake pool; see docstring above)
    mu_f = fake_feats.mean(axis=0)
    sigma_f = np.cov(fake_feats, rowvar=False)

    # Sizes are spaced evenly in 1/n (not in n) - the regression is fit
    # against 1/n, so leverage in that space is what needs to be balanced.
    # Spacing evenly in n instead concentrates almost all the 1/n range
    # into a single small-N point (e.g. n=min_size) with huge leverage,
    # which was observed to distort the fitted intercept badly - see
    # progress_log.md for the diagnostic run that caught this.
    inv_max_real = 1.0 / max_real
    inv_min_size = 1.0 / min_size
    inv_grid = np.linspace(inv_max_real, inv_min_size, num=n_sizes)
    sizes = np.unique(np.clip(np.round(1.0 / inv_grid).astype(int), min_size, max_real))
    sizes.sort()

    avg_fids = []
    for n in sizes:
        fids_at_n = []
        for _ in range(n_repeats):
            r_idx = rng.choice(max_real, size=n, replace=False)
            mu_r = real_feats[r_idx].mean(axis=0)
            sigma_r = np.cov(real_feats[r_idx], rowvar=False)
            fids_at_n.append(fast_frechet_distance(mu_r, sigma_r, mu_f, sigma_f))
        avg_fids.append(float(np.mean(fids_at_n)))

    inv_sizes = 1.0 / sizes
    slope, intercept = np.polyfit(inv_sizes, avg_fids, deg=1)

    return {
        "sizes": sizes.tolist(),
        "avg_fid_per_size": avg_fids,
        "slope": float(slope),
        "fid_infinity": float(intercept),
        "max_size_used": int(max_real),
        "n_fake_used": int(len(fake_feats)),
    }


def run_scope(
    label: str,
    df: pd.DataFrame,
    generator: torch.nn.Module,
    images_root: Path,
    masks_root: Path,
    fake_out_dir: Path,
    device: torch.device,
    fx,
    samples_per_mask: int,
    bias_n_sizes: int,
    bias_n_repeats: int,
    bias_min_size: int,
) -> dict:
    print(f"\n=== FID scope: {label} ({len(df)} real images) ===")
    real_paths = [str(images_root / p) for p in df["image_path"]]
    fake_paths = generate_fakes(generator, df, masks_root, fake_out_dir, device, samples_per_mask)
    print(f"  generated {len(fake_paths)} fake samples ({samples_per_mask} per mask)")

    raw_fid, real_feats, fake_feats = compute_raw_fid(real_paths, fake_paths, fx, device)
    print(f"  raw FID: {raw_fid:.3f}")

    try:
        corrected = bias_corrected_fid(
            real_feats, fake_feats,
            n_sizes=bias_n_sizes, n_repeats=bias_n_repeats, min_size=bias_min_size,
        )
        print(
            f"  bias-corrected FID (n_real->inf, fake pool fixed at {corrected['n_fake_used']}): "
            f"{corrected['fid_infinity']:.3f} "
            f"(max real N used: {corrected['max_size_used']}, "
            f"{bias_n_sizes} sizes x {bias_n_repeats} repeats)"
        )
        print("  FID(n_real) curve (diagnostic - should trend downward as n grows,")
        print("  and the last point should sit close to raw_fid):")
        for n, fid_n in zip(corrected["sizes"], corrected["avg_fid_per_size"]):
            print(f"    n_real={n:>5d}  avg_fid={fid_n:.3f}")
        print(f"    (raw_fid, for comparison, was {raw_fid:.3f})")
    except ValueError as e:
        print(f"  bias correction skipped: {e}")
        corrected = None

    return {
        "label": label,
        "raw_fid": raw_fid,
        "corrected": corrected,
        "n_real": len(real_paths),
        "n_fake": len(fake_paths),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="experiments/checkpoints/pix2pix/generator_last.pth")
    parser.add_argument("--splits-dir", type=str, default="data/splits")
    parser.add_argument("--raw-images-dir", type=str, default="data/raw/severstal-steel-defect-detection")
    parser.add_argument("--processed-dir", type=str, default="data/processed")
    parser.add_argument("--class-target-id", type=int, default=2)
    parser.add_argument("--samples-per-mask", type=int, default=5)
    parser.add_argument("--fid-output-dir", type=str, default="experiments/fid/pix2pix")
    parser.add_argument("--experiment-name", type=str, default="pix2pix_fid")
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="MLflow run name. Defaults to '<checkpoint-stem>_class<id>_fid' "
        "so runs are identifiable in the UI without opening each one - "
        "e.g. 'generator_last_class2_fid' instead of a random hash.",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="final",
        help="Free-text tag for grouping related runs in the MLflow UI/search "
        "(e.g. 'checkerboard_fix_attempts', 'final', 'smoke_test').",
    )
    parser.add_argument(
        "--note",
        type=str,
        default="",
        help="Free-text tag for anything not captured by other tags/params "
        "(e.g. 'generator_variant=convtranspose_k4_no_icnr').",
    )
    parser.add_argument(
        "--dry-run-limit",
        type=int,
        default=None,
        help="If set, only use the first N rows of the (overall and class) test "
        "dataframes - for a fast shape/sanity check before the full run.",
    )
    parser.add_argument(
        "--bias-n-sizes", type=int, default=8,
        help="Number of subsample sizes N used for the Chong & Forsyth "
        "extrapolation. Cost is n_sizes * n_repeats Frechet-distance calls "
        "per scope, independent of the actual dataset size (each call works "
        "on the fixed 2048x2048 InceptionV3 covariance matrix) - lower this "
        "for a quicker/coarser estimate, e.g. on the large 'overall' scope "
        "where bias is less of a concern than for the small class_2 subset.",
    )
    parser.add_argument(
        "--bias-n-repeats", type=int, default=20,
        help="Repeats averaged per subsample size - trades runtime for lower "
        "variance in the per-size FID estimate. Same cost note as --bias-n-sizes.",
    )
    parser.add_argument(
        "--bias-min-size", type=int, default=20,
        help="Smallest subsample size used in the extrapolation grid. Scopes "
        "with fewer than this many paired samples skip bias correction "
        "entirely (raw FID is still reported).",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    splits_dir = Path(args.splits_dir)
    images_root = Path(args.raw_images_dir)
    masks_root = Path(args.processed_dir)
    fid_out_dir = Path(args.fid_output_dir)

    test_df = load_test_split(splits_dir)

    target_ids = get_class_image_ids(f"{args.raw_images_dir}/train.csv", class_id=args.class_target_id)
    class_df = test_df[test_df["image_id"].isin(target_ids)].reset_index(drop=True)
    print(f"Test split: {len(test_df)} total, {len(class_df)} contain class_{args.class_target_id}")

    if args.dry_run_limit is not None:
        test_df = test_df.head(args.dry_run_limit).reset_index(drop=True)
        class_df = class_df.head(args.dry_run_limit).reset_index(drop=True)
        print(f"[dry run] limited to {len(test_df)} overall / {len(class_df)} class_{args.class_target_id} rows")

    generator = build_generator().to(device)
    generator.load_state_dict(torch.load(args.checkpoint, map_location=device))
    print(f"Loaded generator checkpoint: {args.checkpoint}")

    fx = build_feature_extractor("clean", device=device)

    checkpoint_stem = Path(args.checkpoint).stem
    run_name = args.run_name or f"{checkpoint_stem}_class{args.class_target_id}_fid"

    mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(vars(args))
        mlflow.set_tags(
            {
                "stage": args.stage,
                "note": args.note,
                "git_commit": get_git_commit_hash(),
                "checkpoint": args.checkpoint,
            }
        )

        results = [
            run_scope(
                "overall", test_df, generator, images_root, masks_root,
                fid_out_dir / "overall", device, fx, args.samples_per_mask,
                args.bias_n_sizes, args.bias_n_repeats, args.bias_min_size,
            ),
            run_scope(
                f"class_{args.class_target_id}", class_df, generator, images_root, masks_root,
                fid_out_dir / f"class_{args.class_target_id}", device, fx, args.samples_per_mask,
                args.bias_n_sizes, args.bias_n_repeats, args.bias_min_size,
            ),
        ]

        for r in results:
            mlflow.log_metric(f"fid_raw_{r['label']}", r["raw_fid"])
            if r["corrected"] is not None:
                mlflow.log_metric(f"fid_corrected_{r['label']}", r["corrected"]["fid_infinity"])
            mlflow.log_metric(f"fid_n_real_{r['label']}", r["n_real"])
            mlflow.log_metric(f"fid_n_fake_{r['label']}", r["n_fake"])

    print("\nDone.")


if __name__ == "__main__":
    main()
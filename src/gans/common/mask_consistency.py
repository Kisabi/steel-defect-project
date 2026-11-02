"""
Mask-consistency check shared by Pix2Pix, SPADE and StyleGAN2-ADA.

Question answered: does the generated defect appear where (and as the class) the
input mask asked for? FID cannot tell - it only measures realism of the image
distribution, not adherence to the conditioning mask.

Method
------
1. For every mask in the chosen split, generate `--samples-per-mask` fake images.
2. Feed each fake image to the FROZEN baseline U-Net and compare its prediction
   with the mask that produced the image (the mask is the reference).
3. The same U-Net is run on the REAL image of the same mask -> "real ceiling".
   The segmenter is itself weak on class_1/class_2, so the absolute fake IoU is
   hard to interpret; the meaningful number is the ratio fake IoU / real IoU.

Metrics (dataset-level, "sum then divide" - same convention as the corrected
src/segmentation/evaluate.py; no per-image averaging, no zero_division=1 trap):
- per-class IoU and mIoU (5 classes incl. background, comparable to the 0.6336
  baseline) plus mIoU over the 4 defect classes only;
- image-level presence recall / precision per defect class: a defect of class c
  counts as "present" in a prediction if at least `--presence-min-pixels` pixels
  are predicted as c (a ground-truth class counts as present if it has >= 1 pixel);
- bootstrap 95% CI over MASKS (all samples of one mask stay together).

Preprocessing of generated images: [-1, 1] -> uint8 (same truncating .byte()
quantisation as compute_fid.py, i.e. what ends up in the PNGs on disk) ->
ImageNet mean/std, as the U-Net expects.

Extra diagnostics (same generated images, no extra generation pass), each reported
for fake vs real of the SAME masks, with bootstrap CIs over masks:
- brightness: mean gray level, fraction of "lit" pixels (gray > --lit-threshold),
  and mean gray NEAR defects (within --near-px of any defect pixel) vs FAR from them.
  A "spotlight" generator (bright only around the mask, dark elsewhere) shows up as
  a far/real ratio much lower than the near/real ratio;
- colour: mean chroma (max(R,G,B) - min(R,G,B)) and the fraction of pixels with
  chroma > --chroma-threshold. Real steel images are (nearly) gray, so colour noise
  in fakes can confuse a U-Net trained on gray data;
- local defect contrast: |mean gray inside the class region - mean gray in a ring of
  --ring-px around it| per defect class (ring excludes other defect pixels).
  Low fake contrast means the defect was drawn weakly (GAN problem) rather than
  drawn but not recognised (U-Net problem). The signed contrast is also stored.
- NULL CONTROL for the contrast: the same |region - ring| measure computed on the
  class region shifted sideways by each of --null-shifts pixels (positions where
  there is no defect), averaged. This is the "noise floor" of the measure: the
  contrast a random patch of that image would show anyway. signal-to-null =
  contrast / null; a value near 1 means the defect is not distinguishable from
  ordinary texture variation. Only masks with a valid null position are used.
These are paired with the real image of the same mask; for a ratio, real must be > 0.

Tolerance metrics (--tolerances, default 0,2,5,10,25 px): exact-pixel IoU is brutal for
thin 1-2 px lines - a 2-3 px shift already gives zero. For every defect class (and for
"any_defect" = any of the four) and every tolerance d:
- tolerant recall    = share of ground-truth pixels of the class that have a predicted
                       pixel of the SAME class within d px (square neighbourhood);
- tolerant precision = share of predicted pixels of the class that lie within d px of a
                       ground-truth pixel of that class.
Both are dataset-level (hits and totals summed over all masks and samples, then divided),
d = 0 is exact-pixel recall/precision. Reading: recall rising steeply with d means the
defect is drawn (and recognised) near the right place but boundaries are off;
recall staying low even at d = 25 means it is not recognised near the mask at all.
Comparing a class with "any_defect" separates "wrong place / not drawn" from "recognised
as another class".

--grayscale-fakes converts the generated images to gray (luma, replicated to 3
channels) BEFORE segmentation and all diagnostics - real images are exactly R=G=B,
so this removes colour noise (SPADE) and makes fakes consistent with the real data.
Outputs get a "_gray" suffix so they never overwrite the plain runs.

Scopes: "overall" (whole split) and "class_<id>" (subset of the split containing
the target class). Both are computed from ONE generation pass: the class subset
is a subset of the per-mask results of the overall pass.

Sampling mode: Pix2Pix / SPADE in .train() (Dropout active), StyleGAN2-ADA in
.eval() (diversity from z / noise). --seed makes the fake pool reproducible; use
the same seed for checkpoints you want to compare (paired comparison, see
compare_mask_consistency.py).

Use --split val to choose between checkpoints (e.g. StyleGAN2-ADA epoch 50 vs 55)
and --split test for the final numbers.

Usage (inside the container, from /workspace):
    python -m src.gans.common.mask_consistency --model pix2pix \
        --checkpoint experiments/checkpoints/pix2pix/generator_last.pth --split test

    python -m src.gans.common.mask_consistency --model spade --ngf 32 \
        --checkpoint experiments/checkpoints/spade/generator_epoch0100.pth --split test

    python -m src.gans.common.mask_consistency --model stylegan2_ada \
        --checkpoint experiments/checkpoints/stylegan2_ada/generator_epoch0050.pth --split val

Quick dry run first:
    ... --dry-run-limit 8 --samples-per-mask 2
"""

import argparse
import json
import subprocess
import warnings
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from PIL import Image

from src.gans.pix2pix.dataset import get_class_image_ids  # dataset/splits are shared, unchanged

NUM_GAN_CLASSES = 4  # one-hot mask channels (classes 1..4)
NUM_SEG_CLASSES = 5  # background + 4 defect classes (U-Net output)
CLASS_NAMES = ["background", "class_1", "class_2", "class_3", "class_4"]

IMG_STAT_NAMES = [
    "mean_gray", "lit_fraction", "chroma_mean", "chroma_fraction", "gray_near_defects", "gray_far",
]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# BGR-free RGB palette for the example figures
PALETTE = np.array(
    [[0, 0, 0], [230, 50, 50], [50, 200, 50], [60, 100, 240], [240, 220, 40]], dtype=np.uint8
)


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------

def get_git_commit_hash() -> str:
    """Short git hash (+ '-dirty'), or 'unknown' outside a git repo."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        dirty = subprocess.call(["git", "diff", "--quiet"], stderr=subprocess.DEVNULL) != 0
        return f"{commit}-dirty" if dirty else commit
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def load_label_map(mask_path: Path) -> np.ndarray:
    """(H, W) uint8 label map with values 0..4."""
    return np.array(Image.open(mask_path))


def label_map_to_onehot(label_map: np.ndarray) -> torch.Tensor:
    """(H, W) labels 0..4 -> (4, H, W) float one-hot (classes 1..4).
    Same encoding as Pix2PixSteelDataset._load_mask_onehot."""
    channels = [(label_map == c).astype(np.float32) for c in range(1, NUM_GAN_CLASSES + 1)]
    return torch.from_numpy(np.stack(channels, axis=0))


def float_to_uint8(image: torch.Tensor) -> torch.Tensor:
    """[-1, 1] float (B, 3, H, W) -> uint8. Identical quantisation to
    compute_fid.denormalize_to_uint8 (truncation via .byte())."""
    return ((image.clamp(-1, 1) + 1) / 2 * 255.0).byte()


def to_gray_u8(image_u8: torch.Tensor) -> torch.Tensor:
    """uint8 (B, 3, H, W) -> luma gray, rounded, replicated to 3 identical channels (uint8).
    Real Severstal images have R=G=B exactly, so this matches their format."""
    x = image_u8.float()
    gray = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).round().clamp(0, 255).to(torch.uint8)
    return gray.unsqueeze(1).expand(-1, 3, -1, -1).contiguous()


def uint8_to_unet_input(image_u8: torch.Tensor) -> torch.Tensor:
    """uint8 (B, 3, H, W) -> ImageNet-normalised float, like albumentations Normalize."""
    mean = torch.tensor(IMAGENET_MEAN, device=image_u8.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=image_u8.device).view(1, 3, 1, 1)
    return (image_u8.float() / 255.0 - mean) / std


def clean_for_json(obj):
    """Recursively convert numpy types and NaN -> JSON-safe Python values."""
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_for_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return clean_for_json(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


# ----------------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------------

def build_unet(checkpoint: str, device: torch.device) -> torch.nn.Module:
    """Baseline U-Net (same architecture as src.segmentation.train.build_model).
    encoder_weights=None: the real weights come from the checkpoint, no download needed."""
    model = smp.Unet(
        encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=NUM_SEG_CLASSES
    )
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def build_gan_generator(model_name: str, checkpoint: str, ngf: int, device: torch.device):
    """Adapter: build + load the generator and put it in its sampling mode."""
    if model_name == "pix2pix":
        from src.gans.pix2pix.generator import build_generator

        generator = build_generator()
        sampling_mode = "train"  # Dropout stays active
    elif model_name == "spade":
        from src.gans.spade.generator import build_generator

        generator = build_generator(ngf=ngf)
        sampling_mode = "train"  # Dropout2d stays active
    elif model_name == "stylegan2_ada":
        from src.gans.stylegan2_ada.generator import build_generator

        generator = build_generator()
        sampling_mode = "eval"  # diversity from z / noise
    else:
        raise ValueError(f"Unknown model: {model_name}")

    generator = generator.to(device)
    generator.load_state_dict(torch.load(checkpoint, map_location=device))
    generator.train() if sampling_mode == "train" else generator.eval()
    return generator


@torch.no_grad()
def segment(unet: torch.nn.Module, image_u8: torch.Tensor) -> torch.Tensor:
    """uint8 (B, 3, H, W) on device -> predicted label map (B, H, W) long."""
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        logits = unet(uint8_to_unet_input(image_u8))
    return logits.argmax(dim=1)


# ----------------------------------------------------------------------------
# Per-mask statistics
# ----------------------------------------------------------------------------

def pair_stats(preds: torch.Tensor, gt: torch.Tensor, min_px: int):
    """Statistics for ONE mask and its k predictions.

    preds: (k, H, W) long, gt: (H, W) long (on the same device).
    Returns:
      conf: (5, 5) int64, rows = ground truth, cols = prediction, summed over k;
      pres: (3, 4) int64, [tp, fp, fn] of image-level presence for classes 1..4, summed over k.
    """
    idx = (gt.unsqueeze(0) * NUM_SEG_CLASSES + preds).flatten()
    conf = torch.bincount(idx, minlength=NUM_SEG_CLASSES**2).reshape(NUM_SEG_CLASSES, NUM_SEG_CLASSES)

    gt_has = torch.stack([(gt == c).any() for c in range(1, NUM_SEG_CLASSES)])  # (4,)
    pred_cnt = torch.stack([(preds == c).flatten(1).sum(1) for c in range(1, NUM_SEG_CLASSES)], dim=1)  # (k, 4)
    pred_has = pred_cnt >= min_px

    tp = (pred_has & gt_has).sum(0)
    fp = (pred_has & ~gt_has).sum(0)
    fn = (~pred_has & gt_has).sum(0)
    pres = torch.stack([tp, fp, fn])
    return conf.cpu().numpy().astype(np.int64), pres.cpu().numpy().astype(np.int64)


def dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """(..., H, W) bool -> same shape bool, square dilation by `radius` pixels.
    Separable max-pool (rows then columns) - exactly a square structuring element, but cheap."""
    shape = mask.shape
    x = mask.float().reshape(-1, 1, shape[-2], shape[-1])
    k = 2 * radius + 1
    x = F.max_pool2d(x, kernel_size=(1, k), stride=1, padding=(0, radius))
    x = F.max_pool2d(x, kernel_size=(k, 1), stride=1, padding=(radius, 0))
    return (x > 0).reshape(shape)


TOL_CLASS_NAMES = ["class_1", "class_2", "class_3", "class_4", "any_defect"]


def tolerance_stats(preds: torch.Tensor, gt: torch.Tensor, tolerances: list) -> np.ndarray:
    """Tolerant recall / precision counts for ONE mask and its k predictions.

    preds: (k, H, W) long, gt: (H, W) long. Returns (T, 5, 4) int64 with, per tolerance and
    per class (class_1..class_4, any_defect): [recall_hits, n_gt, precision_hits, n_pred],
    everything summed over the k samples (n_gt counts the gt region once per sample).
    """
    k = preds.shape[0]
    out = np.zeros((len(tolerances), NUM_GAN_CLASSES + 1, 4), dtype=np.int64)
    for j in range(NUM_GAN_CLASSES + 1):
        if j < NUM_GAN_CLASSES:
            region, pred = gt == (j + 1), preds == (j + 1)
        else:
            region, pred = gt > 0, preds > 0
        n_g = int(region.sum()) * k
        n_p = int(pred.sum())
        if n_g == 0 and n_p == 0:
            continue
        for t, d in enumerate(tolerances):
            if d == 0:
                rec_hit = prec_hit = int((pred & region.unsqueeze(0)).sum())
            elif n_g > 0 and n_p > 0:
                rec_hit = int((region.unsqueeze(0) & dilate(pred, d)).sum())
                prec_hit = int((pred & dilate(region, d).unsqueeze(0)).sum())
            else:
                rec_hit = prec_hit = 0
            out[t, j] = [rec_hit, n_g, prec_hit, n_p]
    return out


def extra_stats(images_u8: torch.Tensor, gt: torch.Tensor, p: dict):
    """Brightness / colour / local-contrast diagnostics for ONE mask.

    images_u8: (k, 3, H, W) uint8 (fakes, or the single real image), gt: (H, W) long.
    Everything is averaged over the k samples.
    Returns:
      img: (6,) float64, order = IMG_STAT_NAMES (NaN where the region is empty);
      con: (4, 3) float64, per defect class [signed contrast, absolute contrast,
           null-control absolute contrast] (NaN if the class is absent from the mask,
           has no ring, or - for the null - has no valid shifted position).
    """
    x = images_u8.float()
    gray = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]  # (k, H, W)
    chroma = x.amax(dim=1) - x.amin(dim=1)  # (k, H, W)

    defect = gt > 0
    near = dilate(defect, p["near_px"]) if defect.any() else torch.zeros_like(defect)
    far = ~near

    def mean_in(region: torch.Tensor) -> float:
        return gray[:, region].mean().item() if region.any() else float("nan")

    img = np.array([
        gray.mean().item(),
        (gray > p["lit_threshold"]).float().mean().item(),
        chroma.mean().item(),
        (chroma > p["chroma_threshold"]).float().mean().item(),
        mean_in(near),
        mean_in(far),
    ], dtype=np.float64)

    con = np.full((NUM_GAN_CLASSES, 3), np.nan, dtype=np.float64)
    defect_d = dilate(defect, p["ring_px"]) if defect.any() else torch.zeros_like(defect)
    for c in range(1, NUM_SEG_CLASSES):
        region = gt == c
        if not region.any():
            continue
        n_region = int(region.sum())

        ring = dilate(region, p["ring_px"]) & ~defect
        if ring.any():
            diff = gray[:, region].mean(dim=1) - gray[:, ring].mean(dim=1)  # (k,)
            con[c - 1, 0] = diff.mean().item()
            con[c - 1, 1] = diff.abs().mean().item()

        # null control: same region shape moved sideways to defect-free positions
        null_vals = []
        for shift in p["null_shifts"]:
            null_region = torch.roll(region, shifts=int(shift), dims=1) & ~defect_d
            if int(null_region.sum()) < 0.5 * n_region:  # too much of the patch fell on real defects
                continue
            null_ring = dilate(null_region, p["ring_px"]) & ~null_region & ~defect_d
            if not null_ring.any():
                continue
            d = gray[:, null_region].mean(dim=1) - gray[:, null_ring].mean(dim=1)
            null_vals.append(d.abs().mean().item())
        if null_vals:
            con[c - 1, 2] = float(np.mean(null_vals))
    return img, con


@torch.no_grad()
def run_fakes(generator, unet, df, masks_root, device, samples_per_mask, min_px, extra_params, grayscale=False):
    """Generate fakes for every mask in df, segment them, collect per-mask stats."""
    n = len(df)
    conf_all = np.zeros((n, NUM_SEG_CLASSES, NUM_SEG_CLASSES), dtype=np.int64)
    pres_all = np.zeros((n, 3, NUM_GAN_CLASSES), dtype=np.int64)
    img_all = np.full((n, len(IMG_STAT_NAMES)), np.nan)
    con_all = np.full((n, NUM_GAN_CLASSES, 3), np.nan)
    tol_all = np.zeros((n, len(extra_params["tolerances"]), NUM_GAN_CLASSES + 1, 4), dtype=np.int64)

    for i, row in enumerate(df.itertuples(index=False)):
        label_map = load_label_map(masks_root / row.mask_path)
        mask = label_map_to_onehot(label_map).unsqueeze(0).to(device)
        gt = torch.from_numpy(label_map).long().to(device)

        # one forward per sample (batch of 1), exactly like compute_fid.generate_fakes
        fakes = []
        for _ in range(samples_per_mask):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                fake = generator(mask)
            fakes.append(float_to_uint8(fake.float()))
        fakes_u8 = torch.cat(fakes, dim=0)  # (k, 3, H, W) uint8
        if grayscale:
            fakes_u8 = to_gray_u8(fakes_u8)

        preds = segment(unet, fakes_u8)
        conf_all[i], pres_all[i] = pair_stats(preds, gt, min_px)
        img_all[i], con_all[i] = extra_stats(fakes_u8, gt, extra_params)
        tol_all[i] = tolerance_stats(preds, gt, extra_params["tolerances"])

        if (i + 1) % 100 == 0 or i + 1 == n:
            print(f"  fakes: {i + 1}/{n} masks done")
    return conf_all, pres_all, img_all, con_all, tol_all


@torch.no_grad()
def run_real(unet, df, images_root, masks_root, device, min_px, extra_params):
    """Real ceiling: the same U-Net on the REAL image of every mask (k = 1)."""
    n = len(df)
    conf_all = np.zeros((n, NUM_SEG_CLASSES, NUM_SEG_CLASSES), dtype=np.int64)
    pres_all = np.zeros((n, 3, NUM_GAN_CLASSES), dtype=np.int64)
    img_all = np.full((n, len(IMG_STAT_NAMES)), np.nan)
    con_all = np.full((n, NUM_GAN_CLASSES, 3), np.nan)
    tol_all = np.zeros((n, len(extra_params["tolerances"]), NUM_GAN_CLASSES + 1, 4), dtype=np.int64)

    for i, row in enumerate(df.itertuples(index=False)):
        label_map = load_label_map(masks_root / row.mask_path)
        gt = torch.from_numpy(label_map).long().to(device)
        image = np.array(Image.open(images_root / row.image_path).convert("RGB"))
        image_u8 = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device)

        preds = segment(unet, image_u8)
        conf_all[i], pres_all[i] = pair_stats(preds, gt, min_px)
        img_all[i], con_all[i] = extra_stats(image_u8, gt, extra_params)
        tol_all[i] = tolerance_stats(preds, gt, extra_params["tolerances"])

        if (i + 1) % 200 == 0 or i + 1 == n:
            print(f"  real:  {i + 1}/{n} masks done")
    return conf_all, pres_all, img_all, con_all, tol_all


def get_real_ceiling(unet, df, split, images_root, masks_root, device, min_px, extra_params, cache_dir, use_cache):
    """Real-ceiling stats, cached on disk (they depend only on U-Net + split + all thresholds)."""
    key = (f"px{min_px}_lit{extra_params['lit_threshold']}_chr{extra_params['chroma_threshold']}"
           f"_ring{extra_params['ring_px']}_near{extra_params['near_px']}"
           f"_null{'_'.join(str(v) for v in extra_params['null_shifts'])}"
           f"_tol{'_'.join(str(v) for v in extra_params['tolerances'])}")
    cache_path = cache_dir / f"real_ceiling_{split}_{key}.npz"
    if use_cache and cache_path.exists():
        data = np.load(cache_path, allow_pickle=True)
        if list(data["image_ids"]) == df["image_id"].tolist():
            print(f"Real ceiling loaded from cache: {cache_path}")
            return data["conf"], data["pres"], data["img"], data["con"], data["tol"]
        print("Real-ceiling cache does not match the current split - recomputing")

    print("Computing real ceiling (U-Net on real images)...")
    conf, pres, img, con, tol = run_real(unet, df, images_root, masks_root, device, min_px, extra_params)
    if use_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, conf=conf, pres=pres, img=img, con=con, tol=tol,
                 image_ids=np.array(df["image_id"].tolist()))
    return conf, pres, img, con, tol


# ----------------------------------------------------------------------------
# Aggregation / bootstrap
# ----------------------------------------------------------------------------

def iou_from_conf(conf: np.ndarray) -> np.ndarray:
    """(..., 5, 5) confusion matrices (rows = gt) -> (..., 5) per-class IoU.
    NaN where a class has no gt pixels and no predicted pixels."""
    diag = np.diagonal(conf, axis1=-2, axis2=-1).astype(np.float64)
    denom = conf.sum(-1) + conf.sum(-2) - diag
    return np.where(denom > 0, diag / np.maximum(denom, 1), np.nan)


def presence_metrics(pres_sum: np.ndarray) -> dict:
    """pres_sum: (3, 4) summed [tp, fp, fn] -> recall / precision per defect class."""
    tp, fp, fn = pres_sum.astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.where(tp + fn > 0, tp / (tp + fn), np.nan)
        precision = np.where(tp + fp > 0, tp / (tp + fp), np.nan)
    return {"recall": recall, "precision": precision}


def bootstrap_scores(conf_fake, conf_real, n_boot, rng, chunk=100):
    """Bootstrap over masks. Returns per-resample arrays:
    fake_iou (B,5), real_iou (B,5), fake_miou (B,), real_miou (B,), fake_miou_def, real_miou_def."""
    n = conf_fake.shape[0]
    out = {k: [] for k in ("fake_iou", "real_iou")}
    for start in range(0, n_boot, chunk):
        b = min(chunk, n_boot - start)
        idx = rng.integers(0, n, size=(b, n))
        out["fake_iou"].append(iou_from_conf(conf_fake[idx].sum(1)))
        out["real_iou"].append(iou_from_conf(conf_real[idx].sum(1)))
    res = {k: np.concatenate(v, axis=0) for k, v in out.items()}
    for who in ("fake", "real"):
        res[f"{who}_miou"] = np.nanmean(res[f"{who}_iou"], axis=1)
        res[f"{who}_miou_def"] = np.nanmean(res[f"{who}_iou"][:, 1:], axis=1)
    return res


def ci(samples: np.ndarray):
    """2.5 / 97.5 percentiles along axis 0 (NaN-safe)."""
    return np.nanpercentile(samples, [2.5, 97.5], axis=0)


def summarize_scope(label, sel, conf_fake, pres_fake, conf_real, pres_real, n_boot, rng):
    """All metrics for one scope. `sel` is an index array into the per-mask arrays."""
    cf, cr = conf_fake[sel], conf_real[sel]
    iou_f, iou_r = iou_from_conf(cf.sum(0)), iou_from_conf(cr.sum(0))
    miou_f, miou_r = float(np.nanmean(iou_f)), float(np.nanmean(iou_r))
    miou_f_def, miou_r_def = float(np.nanmean(iou_f[1:])), float(np.nanmean(iou_r[1:]))

    boot = bootstrap_scores(cf, cr, n_boot, rng)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_boot = boot["fake_iou"] / boot["real_iou"]
        miou_ratio_boot = boot["fake_miou"] / boot["real_miou"]
        miou_def_ratio_boot = boot["fake_miou_def"] / boot["real_miou_def"]

    conf_total = cf.sum(0).astype(np.float64)
    row_norm = conf_total / np.maximum(conf_total.sum(1, keepdims=True), 1)  # where gt pixels of class c end up

    return {
        "label": label,
        "n_masks": int(len(sel)),
        "fake": {
            "iou": iou_f, "iou_ci": ci(boot["fake_iou"]),
            "miou": miou_f, "miou_ci": ci(boot["fake_miou"]),
            "miou_defect": miou_f_def, "miou_defect_ci": ci(boot["fake_miou_def"]),
            "presence": presence_metrics(pres_fake[sel].sum(0)),
            "confusion_row_normalized": row_norm,
        },
        "real": {
            "iou": iou_r, "iou_ci": ci(boot["real_iou"]),
            "miou": miou_r, "miou_ci": ci(boot["real_miou"]),
            "miou_defect": miou_r_def, "miou_defect_ci": ci(boot["real_miou_def"]),
            "presence": presence_metrics(pres_real[sel].sum(0)),
        },
        "ratio_fake_over_real": {
            "iou": iou_f / iou_r, "iou_ci": ci(ratio_boot),
            "miou": miou_f / miou_r, "miou_ci": ci(miou_ratio_boot),
            "miou_defect": miou_f_def / miou_r_def, "miou_defect_ci": ci(miou_def_ratio_boot),
        },
    }


def print_scope(s: dict) -> None:
    f, r, q = s["fake"], s["real"], s["ratio_fake_over_real"]
    print(f"\n=== scope: {s['label']} ({s['n_masks']} masks) ===")
    print(f"{'class':<11}{'fake IoU [95% CI]':<27}{'real IoU':<10}{'fake/real [95% CI]':<24}{'pres.recall f/r':<17}{'pres.prec f/r'}")
    for c, name in enumerate(CLASS_NAMES):
        fi = f"{f['iou'][c]:.4f} [{f['iou_ci'][0][c]:.3f}, {f['iou_ci'][1][c]:.3f}]"
        qi = f"{q['iou'][c]:.3f} [{q['iou_ci'][0][c]:.2f}, {q['iou_ci'][1][c]:.2f}]"
        if c >= 1:
            rec = f"{f['presence']['recall'][c - 1]:.2f}/{r['presence']['recall'][c - 1]:.2f}"
            prec = f"{f['presence']['precision'][c - 1]:.2f}/{r['presence']['precision'][c - 1]:.2f}"
        else:
            rec = prec = "-"
        print(f"{name:<11}{fi:<27}{r['iou'][c]:<10.4f}{qi:<24}{rec:<17}{prec}")
    print(
        f"{'mIoU (5)':<11}{f['miou']:.4f} [{f['miou_ci'][0]:.3f}, {f['miou_ci'][1]:.3f}]"
        f"    real {r['miou']:.4f}   ratio {q['miou']:.3f} [{q['miou_ci'][0]:.2f}, {q['miou_ci'][1]:.2f}]"
    )
    print(
        f"{'mIoU (1-4)':<11}{f['miou_defect']:.4f} [{f['miou_defect_ci'][0]:.3f}, {f['miou_defect_ci'][1]:.3f}]"
        f"    real {r['miou_defect']:.4f}   ratio {q['miou_defect']:.3f} [{q['miou_defect_ci'][0]:.2f}, {q['miou_defect_ci'][1]:.2f}]"
    )


def _nan_safe_ratio(f: np.ndarray, r: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(r > 1e-9, f / r, np.nan)


def summarize_extra(img_f, img_r, con_f, con_r, sel, n_boot, rng, chunk=100):
    """Brightness / colour / contrast aggregates for one scope, bootstrap over masks.

    Per statistic: mean over masks (NaN-aware: masks without the class / region are skipped),
    95% CI of that mean, and of the fake/real ratio of means. The contrast signal-to-null
    ratio uses only masks where fake AND real have both a contrast and a valid null value.
    """
    img_f, img_r, con_f, con_r = img_f[sel], img_r[sel], con_f[sel], con_r[sel]
    n = len(sel)
    abs_f, abs_r = con_f[..., 1], con_r[..., 1]
    null_f, null_r = con_f[..., 2], con_r[..., 2]
    valid = ~(np.isnan(abs_f) | np.isnan(abs_r) | np.isnan(null_f) | np.isnan(null_r))

    def matched(a):
        return np.where(valid, a, np.nan)

    arrays = {
        "img": (img_f, img_r),
        "con_abs": (abs_f, abs_r),  # all masks with the class (kept comparable with older runs)
        "con_abs_matched": (matched(abs_f), matched(abs_r)),
        "con_null": (matched(null_f), matched(null_r)),
    }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN slices are expected (absent classes)
        boot = {k: ([], []) for k in arrays}
        for start in range(0, n_boot, chunk):
            idx = rng.integers(0, n, size=(min(chunk, n_boot - start), n))
            for k, (f, r) in arrays.items():
                boot[k][0].append(np.nanmean(f[idx], axis=1))
                boot[k][1].append(np.nanmean(r[idx], axis=1))
        bs = {k: (np.concatenate(v[0]), np.concatenate(v[1])) for k, v in boot.items()}

        out = {}
        for k, (f, r) in arrays.items():
            bf, br = bs[k]
            out[k] = {
                "fake": np.nanmean(f, axis=0), "fake_ci": np.nanpercentile(bf, [2.5, 97.5], axis=0),
                "real": np.nanmean(r, axis=0), "real_ci": np.nanpercentile(br, [2.5, 97.5], axis=0),
                "ratio": _nan_safe_ratio(np.nanmean(f, axis=0), np.nanmean(r, axis=0)),
                "ratio_ci": np.nanpercentile(_nan_safe_ratio(bf, br), [2.5, 97.5], axis=0),
            }
        out["img"]["names"] = IMG_STAT_NAMES
        out["con_abs"]["n_masks_with_class"] = (~np.isnan(abs_f)).sum(axis=0)
        out["con_signed"] = {
            "fake": np.nanmean(con_f[..., 0], axis=0), "real": np.nanmean(con_r[..., 0], axis=0),
        }

        # signal-to-null = |contrast| / null contrast, same masks for numerator and denominator
        snr_f = _nan_safe_ratio(np.nanmean(matched(abs_f), axis=0), np.nanmean(matched(null_f), axis=0))
        snr_r = _nan_safe_ratio(np.nanmean(matched(abs_r), axis=0), np.nanmean(matched(null_r), axis=0))
        snr_f_b = _nan_safe_ratio(bs["con_abs_matched"][0], bs["con_null"][0])
        snr_r_b = _nan_safe_ratio(bs["con_abs_matched"][1], bs["con_null"][1])
        out["con_snr"] = {
            "fake": snr_f, "fake_ci": np.nanpercentile(snr_f_b, [2.5, 97.5], axis=0),
            "real": snr_r, "real_ci": np.nanpercentile(snr_r_b, [2.5, 97.5], axis=0),
            "fake_over_real": _nan_safe_ratio(snr_f, snr_r),
            "fake_over_real_ci": np.nanpercentile(_nan_safe_ratio(snr_f_b, snr_r_b), [2.5, 97.5], axis=0),
            "n_masks_matched": valid.sum(axis=0),
        }
    return out


def print_extra(label: str, e: dict) -> None:
    print(f"\n--- extra diagnostics, scope: {label} (fake vs real of the same masks) ---")
    print(f"{'statistic':<20}{'fake [95% CI]':<26}{'real [95% CI]':<26}{'fake/real [95% CI]'}")

    def fmt(v, lo, hi, nd=3):
        return "nan" if np.isnan(v) else f"{v:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"

    d = e["img"]
    for j, name in enumerate(IMG_STAT_NAMES):
        print(f"{name:<20}{fmt(d['fake'][j], *d['fake_ci'][:, j]):<26}"
              f"{fmt(d['real'][j], *d['real_ci'][:, j]):<26}{fmt(d['ratio'][j], *d['ratio_ci'][:, j], nd=2)}")

    print("local contrast |region - ring|, gray levels")
    d = e["con_abs"]
    for c in range(NUM_GAN_CLASSES):
        name = f"{CLASS_NAMES[c + 1]} (n={int(d['n_masks_with_class'][c])})"
        print(f"{name:<20}{fmt(d['fake'][c], *d['fake_ci'][:, c], nd=2):<26}"
              f"{fmt(d['real'][c], *d['real_ci'][:, c], nd=2):<26}{fmt(d['ratio'][c], *d['ratio_ci'][:, c], nd=2)}")
    sg = e["con_signed"]
    print("signed contrast (region - ring; negative = darker defect), fake / real: "
          + ", ".join(f"{CLASS_NAMES[c + 1]} {sg['fake'][c]:+.1f}/{sg['real'][c]:+.1f}" for c in range(NUM_GAN_CLASSES)))

    print("\nnull control: contrast / contrast at shifted defect-free positions (signal-to-null; ~1 = indistinguishable from texture)")
    print(f"{'class':<20}{'fake contrast / null':<24}{'fake SNR [95% CI]':<24}{'real contrast / null':<24}{'real SNR [95% CI]':<24}{'fakeSNR/realSNR [CI]'}")
    a, nl, sn = e["con_abs_matched"], e["con_null"], e["con_snr"]
    for c in range(NUM_GAN_CLASSES):
        name = f"{CLASS_NAMES[c + 1]} (n={int(sn['n_masks_matched'][c])})"
        fc = "nan" if np.isnan(a["fake"][c]) else f"{a['fake'][c]:.2f} / {nl['fake'][c]:.2f}"
        rc = "nan" if np.isnan(a["real"][c]) else f"{a['real'][c]:.2f} / {nl['real'][c]:.2f}"
        print(f"{name:<20}{fc:<24}{fmt(sn['fake'][c], *sn['fake_ci'][:, c], nd=2):<24}{rc:<24}"
              f"{fmt(sn['real'][c], *sn['real_ci'][:, c], nd=2):<24}"
              f"{fmt(sn['fake_over_real'][c], *sn['fake_over_real_ci'][:, c], nd=2)}")


def summarize_tolerance(tol_f, tol_r, sel, tolerances, n_boot, rng, chunk=100):
    """Tolerant recall / precision per class and tolerance, dataset-level, bootstrap CI over masks."""
    tol_f, tol_r = tol_f[sel], tol_r[sel]
    n = len(sel)

    def rates(t):  # (..., T, 5, 4) counts summed over masks -> recall, precision (..., T, 5)
        with np.errstate(divide="ignore", invalid="ignore"):
            rec = np.where(t[..., 1] > 0, t[..., 0] / np.maximum(t[..., 1], 1), np.nan)
            prec = np.where(t[..., 3] > 0, t[..., 2] / np.maximum(t[..., 3], 1), np.nan)
        return rec, prec

    rec_f, prec_f = rates(tol_f.sum(0))
    rec_r, prec_r = rates(tol_r.sum(0))

    parts = {"rec_f": [], "prec_f": [], "rec_r": [], "prec_r": []}
    for start in range(0, n_boot, chunk):
        idx = rng.integers(0, n, size=(min(chunk, n_boot - start), n))
        rf, pf = rates(tol_f[idx].sum(1))
        rr, pr = rates(tol_r[idx].sum(1))
        for key, val in (("rec_f", rf), ("prec_f", pf), ("rec_r", rr), ("prec_r", pr)):
            parts[key].append(val)
    b = {k: np.concatenate(v, axis=0) for k, v in parts.items()}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)

        def ci_of(a):
            return np.nanpercentile(a, [2.5, 97.5], axis=0)

        return {
            "tolerances": list(tolerances),
            "classes": TOL_CLASS_NAMES,
            "fake": {"recall": rec_f, "recall_ci": ci_of(b["rec_f"]),
                     "precision": prec_f, "precision_ci": ci_of(b["prec_f"])},
            "real": {"recall": rec_r, "recall_ci": ci_of(b["rec_r"]),
                     "precision": prec_r, "precision_ci": ci_of(b["prec_r"])},
            "recall_fake_over_real": _nan_safe_ratio(rec_f, rec_r),
            "recall_fake_over_real_ci": ci_of(_nan_safe_ratio(b["rec_f"], b["rec_r"])),
        }


def print_tolerance(label: str, t: dict) -> None:
    tols = t["tolerances"]
    print(f"\n--- tolerance metrics, scope: {label} (cell = fake / real) ---")
    print("recall = share of GT pixels with a predicted pixel of the same class within d px; "
          "precision = share of predicted pixels within d px of GT of that class")

    def table(title, key):
        print(f"\n{title}")
        print(f"{'class':<12}" + "".join(f"{'d=' + str(d):<14}" for d in tols))
        for j, name in enumerate(t["classes"]):
            cells = []
            for i in range(len(tols)):
                f, r = t["fake"][key][i][j], t["real"][key][i][j]
                cells.append("-" if np.isnan(f) and np.isnan(r)
                             else f"{'nan' if np.isnan(f) else format(f, '.2f')}/{'nan' if np.isnan(r) else format(r, '.2f')}")
            print(f"{name:<12}" + "".join(f"{c:<14}" for c in cells))

    table("tolerant recall", "recall")
    table("tolerant precision", "precision")


# ----------------------------------------------------------------------------
# Qualitative examples
# ----------------------------------------------------------------------------

def colorize(label_map: np.ndarray) -> np.ndarray:
    return PALETTE[label_map]


@torch.no_grad()
def save_examples(generator, unet, df, ids, images_root, masks_root, out_dir, device, grayscale=False):
    """Per example one PNG with 5 stacked rows (each 1600x256):
    real image / fake image / GT mask / U-Net on fake / U-Net on real."""
    out_dir.mkdir(parents=True, exist_ok=True)
    sub = df[df["image_id"].isin(ids)]
    for row in sub.itertuples(index=False):
        label_map = load_label_map(masks_root / row.mask_path)
        mask = label_map_to_onehot(label_map).unsqueeze(0).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            fake = generator(mask)
        fake_u8 = float_to_uint8(fake.float())
        if grayscale:
            fake_u8 = to_gray_u8(fake_u8)
        real = np.array(Image.open(images_root / row.image_path).convert("RGB"))
        real_u8 = torch.from_numpy(real).permute(2, 0, 1).unsqueeze(0).to(device)

        pred_fake = segment(unet, fake_u8)[0].cpu().numpy()
        pred_real = segment(unet, real_u8)[0].cpu().numpy()
        rows = [
            real,
            fake_u8[0].permute(1, 2, 0).cpu().numpy(),
            colorize(label_map),
            colorize(pred_fake),
            colorize(pred_real),
        ]
        Image.fromarray(np.concatenate(rows, axis=0)).save(out_dir / f"{row.image_id}.png")
    print(f"Saved {len(sub)} example figures to {out_dir}")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["pix2pix", "spade", "stylegan2_ada"])
    parser.add_argument("--checkpoint", type=str, required=True, help="Generator state_dict (EMA for StyleGAN2-ADA)")
    parser.add_argument("--ngf", type=int, default=32, help="SPADE only; must match training (default 32)")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"],
                        help="val for checkpoint selection, test for final numbers")
    parser.add_argument("--unet-checkpoint", type=str, default="experiments/checkpoints/unet_baseline/best.pth")
    parser.add_argument("--samples-per-mask", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--class-target-id", type=int, default=2)
    parser.add_argument("--presence-min-pixels", type=int, default=50,
                        help="Min predicted pixels of a class to count it as 'present' in an image")
    parser.add_argument("--lit-threshold", type=float, default=30.0,
                        help="Gray level (0-255) above which a pixel counts as 'lit'")
    parser.add_argument("--chroma-threshold", type=float, default=8.0,
                        help="max(R,G,B)-min(R,G,B) above which a pixel counts as coloured")
    parser.add_argument("--ring-px", type=int, default=12, help="Ring width around a defect for local contrast")
    parser.add_argument("--near-px", type=int, default=100,
                        help="Distance from any defect pixel that counts as 'near defects'")
    parser.add_argument("--tolerances", type=str, default="0,2,5,10,25",
                        help="Comma-separated pixel tolerances for the tolerant recall/precision (0 = exact)")
    parser.add_argument("--null-shifts", type=str, default="200,-200,400,-400",
                        help="Comma-separated horizontal shifts (px) for the contrast null control")
    parser.add_argument("--grayscale-fakes", action="store_true",
                        help="Convert generated images to gray before segmentation/diagnostics (output tag gets _gray)")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--save-examples", type=int, default=4,
                        help="Number of qualitative figures from the target-class subset (0 to disable)")
    parser.add_argument("--splits-dir", type=str, default="data/splits")
    parser.add_argument("--raw-images-dir", type=str, default="data/raw/severstal-steel-defect-detection")
    parser.add_argument("--processed-dir", type=str, default="data/processed")
    parser.add_argument("--output-dir", type=str, default="experiments/mask_consistency")
    parser.add_argument("--experiment-name", type=str, default="mask_consistency")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--stage", type=str, default="final")
    parser.add_argument("--note", type=str, default="")
    parser.add_argument("--dry-run-limit", type=int, default=None,
                        help="Use only the first N rows of the split (sanity check; disables caching)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    images_root = Path(args.raw_images_dir)
    masks_root = Path(args.processed_dir)
    out_root = Path(args.output_dir)
    ckpt_stem = Path(args.checkpoint).stem
    tag = f"{ckpt_stem}_{args.split}" + ("_gray" if args.grayscale_fakes else "")

    df = pd.read_csv(Path(args.splits_dir) / f"{args.split}.csv").reset_index(drop=True)
    if args.dry_run_limit is not None:
        df = df.head(args.dry_run_limit).reset_index(drop=True)
        print(f"[dry run] limited to {len(df)} masks")

    target_ids = get_class_image_ids(f"{args.raw_images_dir}/train.csv", class_id=args.class_target_id)
    is_class = df["image_id"].isin(target_ids).to_numpy()
    print(f"{args.split} split: {len(df)} masks, {int(is_class.sum())} contain class_{args.class_target_id}")

    unet = build_unet(args.unet_checkpoint, device)
    print(f"Loaded U-Net: {args.unet_checkpoint}")
    generator = build_gan_generator(args.model, args.checkpoint, args.ngf, device)
    print(f"Loaded {args.model} generator: {args.checkpoint}")

    extra_params = {
        "lit_threshold": args.lit_threshold, "chroma_threshold": args.chroma_threshold,
        "ring_px": args.ring_px, "near_px": args.near_px,
        "null_shifts": [int(v) for v in args.null_shifts.split(",") if v.strip()],
        "tolerances": sorted({int(v) for v in args.tolerances.split(",") if v.strip()}),
    }
    conf_real, pres_real, img_real, con_real, tol_real = get_real_ceiling(
        unet, df, args.split, images_root, masks_root, device, args.presence_min_pixels, extra_params,
        out_root, use_cache=args.dry_run_limit is None,
    )

    torch.manual_seed(args.seed)
    print(f"Generating {args.samples_per_mask} fake(s) per mask and segmenting...")
    conf_fake, pres_fake, img_fake, con_fake, tol_fake = run_fakes(
        generator, unet, df, masks_root, device, args.samples_per_mask, args.presence_min_pixels, extra_params,
        grayscale=args.grayscale_fakes,
    )

    rng = np.random.default_rng(args.seed)
    scopes = [("overall", np.arange(len(df)))]
    if is_class.any():
        scopes.append((f"class_{args.class_target_id}", np.flatnonzero(is_class)))
    results = [
        summarize_scope(label, sel, conf_fake, pres_fake, conf_real, pres_real, args.n_bootstrap, rng)
        for label, sel in scopes
    ]
    # separate RNG: the IoU bootstrap above stays identical to runs made before these diagnostics existed
    rng_extra = np.random.default_rng(args.seed + 1)
    for (label, sel), s in zip(scopes, results):
        s["extra"] = summarize_extra(img_fake, img_real, con_fake, con_real, sel, args.n_bootstrap, rng_extra)
        s["tolerance"] = summarize_tolerance(
            tol_fake, tol_real, sel, extra_params["tolerances"], args.n_bootstrap, rng_extra)
    for s in results:
        print_scope(s)
        print_extra(s["label"], s["extra"])
        print_tolerance(s["label"], s["tolerance"])

    # ---- save artifacts ----
    model_dir = out_root / args.model
    model_dir.mkdir(parents=True, exist_ok=True)
    json_path = model_dir / f"{tag}.json"
    npz_path = model_dir / f"{tag}.npz"
    payload = {
        "model": args.model, "checkpoint": args.checkpoint, "split": args.split,
        "samples_per_mask": args.samples_per_mask, "seed": args.seed,
        "presence_min_pixels": args.presence_min_pixels, "n_bootstrap": args.n_bootstrap,
        "unet_checkpoint": args.unet_checkpoint, "grayscale_fakes": args.grayscale_fakes,
        "extra_params": extra_params, "scopes": results,
    }
    json_path.write_text(json.dumps(clean_for_json(payload), indent=2))
    # per-mask arrays -> paired bootstrap between checkpoints (compare_mask_consistency.py)
    np.savez(
        npz_path, conf_fake=conf_fake, pres_fake=pres_fake, conf_real=conf_real,
        img_fake=img_fake, img_real=img_real, con_fake=con_fake, con_real=con_real,
        tol_fake=tol_fake, tol_real=tol_real,
        image_ids=np.array(df["image_id"].tolist()), is_class=is_class,
    )
    print(f"\nSaved: {json_path}\n       {npz_path}")

    if args.save_examples > 0 and is_class.any():
        ex_ids = set(df.loc[is_class, "image_id"].head(args.save_examples))
        torch.manual_seed(args.seed)
        save_examples(generator, unet, df, ex_ids, images_root, masks_root, model_dir / "examples" / tag, device,
                      grayscale=args.grayscale_fakes)

    # ---- MLflow ----
    run_name = args.run_name or f"{args.model}_{tag}_maskcons"
    mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(vars(args))
        mlflow.set_tags({
            "stage": args.stage, "note": args.note, "git_commit": get_git_commit_hash(),
            "checkpoint": args.checkpoint, "generator_variant": args.model, "split": args.split,
            "grayscale_fakes": str(args.grayscale_fakes),
        })
        for s in results:
            lb = s["label"]
            mlflow.log_metric(f"fake_miou_{lb}", s["fake"]["miou"])
            mlflow.log_metric(f"real_miou_{lb}", s["real"]["miou"])
            mlflow.log_metric(f"ratio_miou_{lb}", s["ratio_fake_over_real"]["miou"])
            mlflow.log_metric(f"fake_miou_defect_{lb}", s["fake"]["miou_defect"])
            mlflow.log_metric(f"ratio_miou_defect_{lb}", s["ratio_fake_over_real"]["miou_defect"])
            for c, name in enumerate(CLASS_NAMES):
                for key, val in (("fake_iou", s["fake"]["iou"][c]), ("real_iou", s["real"]["iou"][c]),
                                 ("ratio_iou", s["ratio_fake_over_real"]["iou"][c])):
                    if not np.isnan(val):
                        mlflow.log_metric(f"{key}_{lb}_{name}", float(val))
            for c in range(NUM_GAN_CLASSES):
                for key in ("recall", "precision"):
                    val = s["fake"]["presence"][key][c]
                    if not np.isnan(val):
                        mlflow.log_metric(f"fake_presence_{key}_{lb}_class_{c + 1}", float(val))
            ex = s["extra"]
            for j, name in enumerate(IMG_STAT_NAMES):
                for key, val in (("fake", ex["img"]["fake"][j]), ("real", ex["img"]["real"][j]),
                                 ("ratio", ex["img"]["ratio"][j])):
                    if not np.isnan(val):
                        mlflow.log_metric(f"{key}_{name}_{lb}", float(val))
            for c in range(NUM_GAN_CLASSES):
                for key, val in (("fake", ex["con_abs"]["fake"][c]), ("real", ex["con_abs"]["real"][c]),
                                 ("ratio", ex["con_abs"]["ratio"][c])):
                    if not np.isnan(val):
                        mlflow.log_metric(f"{key}_contrast_abs_{lb}_class_{c + 1}", float(val))
                for key, val in (("fake", ex["con_snr"]["fake"][c]), ("real", ex["con_snr"]["real"][c]),
                                 ("fake_over_real", ex["con_snr"]["fake_over_real"][c])):
                    if not np.isnan(val):
                        mlflow.log_metric(f"{key}_contrast_snr_{lb}_class_{c + 1}", float(val))
            tl = s["tolerance"]
            for t_i, d in enumerate(tl["tolerances"]):
                for j, cname in enumerate(TOL_CLASS_NAMES):
                    for key, val in (("fake_recall", tl["fake"]["recall"][t_i][j]),
                                     ("real_recall", tl["real"]["recall"][t_i][j]),
                                     ("fake_precision", tl["fake"]["precision"][t_i][j])):
                        if not np.isnan(val):
                            mlflow.log_metric(f"tol_{key}_{lb}_{cname}_d{d}", float(val))
        mlflow.log_artifact(str(json_path))

    print("\nDone.")


if __name__ == "__main__":
    main()
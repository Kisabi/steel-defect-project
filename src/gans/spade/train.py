"""
Training script for SPADE: mask-conditioned synthetic steel defect
generation.

SPADE training recipe (Park et al., 2019), kept faithful to the paper:
- G_loss = hinge_g(D(fake)) + lambda_feat * feature_matching
                              + lambda_vgg  * vgg_perceptual
- D_loss = hinge_d(D(real), D(fake.detach())), summed over both
  discriminator scales.
- Adam, betas=(0.5, 0.999) for both G and D - same beta1=0.5 fix as
  Pix2Pix (see progress_log.md for why beta1=0.9, the Adam default,
  makes GAN training visibly unstable).
- Asymmetric learning rates (TTUR-style): lr_g=4e-4, lr_d=1e-4 by
  default - NOT the same lr=2e-4 used for both G/D in Pix2Pix. Added
  after the 25-epoch smoke test showed d_loss falling monotonically
  while g_loss climbed monotonically and vgg_loss stayed flat across
  every sampled epoch - the discriminator steadily overpowering the
  generator under hinge loss. Slowing D relative to G is the standard
  first fix for this pattern. Documented here as a deliberate deviation
  from the "reuse Pix2Pix's optimizer settings" default, not an
  oversight - see progress_log.md when this session's findings are
  written up.
- lambda_feat=10, lambda_vgg=10 are the paper's defaults.

Reuses the same class_2-weighted sampler, dataset, and train/val/test
splits as Pix2Pix (src/gans/pix2pix/dataset.py) - no changes needed
there, see chat discussion when starting this session. Also reuses the
same qualitative sample-grid-every-N-epochs pattern and MLflow logging
conventions as pix2pix/train.py, so both runs are comparable side by
side in the MLflow UI.

Stochastic sampling: SPADEGenerator has no VAE branch (see
spade_resblock.py); diversity across --samples-per-mask calls in
compute_fid.py instead comes from Dropout2d staying active because the
generator is kept in .train() mode during sampling, exactly like
Pix2Pix already does. No special-casing needed in compute_fid.py itself.

VRAM: SPADE's default ngf=64 is substantially heavier than Pix2Pix's
U-Net at a comparable bottleneck (see generator.py docstring - the SPADE
mask-conditioning branches run at every resolution, not just at the
input). --ngf defaults to 32 here (not the paper's 64) specifically
because it lands within ~1M params of Pix2Pix's generator (29.41M vs
29.24M) - a deliberate, reportable choice to keep generator capacity a
controlled variable between the two architectures, not just a VRAM
workaround. Raise it if your GPU has headroom; the dry-run shape checks
in this project's chat history only verified ngf=16/32 forward+backward
correctness on CPU, not real VRAM fit - watch rocm-smi on the first run
the same way Pix2Pix's smoke tests did.

Usage:
    python -m src.gans.spade.train \
        --epochs 100 \
        --batch-size 2 \
        --class-target-fraction 0.15
"""

import argparse
from pathlib import Path

import mlflow
import torch
import torch.nn as nn
import torchvision.utils as vutils
from torch.utils.data import DataLoader

from src.gans.common.losses import HingeGANLoss, VGGLoss, feature_matching_loss
from src.gans.pix2pix.dataset import (
    Pix2PixSteelDataset,
    build_class_weighted_sampler,
    get_class_image_ids,
)
from src.gans.spade.discriminator import build_discriminator
from src.gans.spade.generator import build_generator


def denormalize(image_tensor: torch.Tensor) -> torch.Tensor:
    """[-1, 1] -> [0, 1], for saving/visualizing generator output."""
    return (image_tensor.clamp(-1, 1) + 1) / 2


def save_sample_grid(generator, val_batch, device, out_path: Path):
    """Save a (mask-as-RGB, fake, real) comparison grid for a fixed
    validation batch. generator.eval() is deliberately NOT used here -
    Dropout2d must stay active for the sample to reflect the same
    stochastic-sampling behavior compute_fid.py relies on (see module
    docstring)."""
    generator.train()
    with torch.no_grad():
        masks = val_batch["mask"].to(device)
        real_images = val_batch["image"].to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            fake_images = generator(masks)

    mask_vis = masks.argmax(dim=1, keepdim=True).float() / masks.shape[1]
    mask_vis = mask_vis.repeat(1, 3, 1, 1)

    grid_rows = torch.cat([mask_vis, denormalize(fake_images.float()), denormalize(real_images)], dim=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vutils.save_image(grid_rows, out_path, nrow=masks.shape[0])


def train_one_epoch(
    generator, discriminator, loader, optimizer_g, optimizer_d,
    scaler_g, scaler_d, vgg_loss_fn, lambda_feat, lambda_vgg,
    num_d, n_layers_d, d_update_every, device, log_every: int, epoch: int, global_step: int,
):
    generator.train()
    discriminator.train()

    running = {"g_loss": 0.0, "d_loss": 0.0, "adv_loss": 0.0, "fm_loss": 0.0, "vgg_loss": 0.0}
    d_updates_this_epoch = 0

    for step, batch in enumerate(loader):
        masks = batch["mask"].to(device)
        real_images = batch["image"].to(device)

        # --- discriminator step (only every d_update_every steps) ---
        # Diagnosed from two 25-epoch smoke tests (equal lr, then TTUR
        # lr_g=4e-4/lr_d=1e-4): d_loss fell monotonically and g_loss rose
        # monotonically in BOTH runs, at nearly identical rates - TTUR's
        # learning-rate asymmetry alone didn't touch the underlying
        # dynamic, pointing to a capacity/update-frequency imbalance
        # rather than a step-size one. Halving D's update frequency
        # (d_update_every=2 default) gives G roughly twice as many
        # effective learning steps per D step, the standard next
        # escalation after TTUR fails - see chat discussion.
        do_d_update = (global_step % d_update_every == 0)

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            fake_images = generator(masks)
            real_pair = torch.cat([masks, real_images], dim=1)

        if do_d_update:
            optimizer_d.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                fake_pair_detached = torch.cat([masks, fake_images.detach()], dim=1)
                real_feats_d = discriminator(real_pair)
                fake_feats_d = discriminator(fake_pair_detached)
                loss_d = sum(
                    HingeGANLoss.d_loss(r[-1], f[-1]) for r, f in zip(real_feats_d, fake_feats_d)
                )
            scaler_d.scale(loss_d).backward()
            scaler_d.step(optimizer_d)
            scaler_d.update()
            running["d_loss"] += loss_d.item()
            d_updates_this_epoch += 1

        # --- generator step ---
        optimizer_g.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            fake_pair = torch.cat([masks, fake_images], dim=1)
            # real_feats_g is only ever used as a detached feature-matching
            # target (see feature_matching_loss), so no_grad here avoids
            # building - then immediately discarding - a full backward
            # graph through the discriminator's parameters for the real
            # branch. fake_feats_g must stay grad-tracked: the whole point
            # of this forward pass is the gradient path back into the
            # generator.
            with torch.no_grad():
                real_feats_g = discriminator(real_pair)
            fake_feats_g = discriminator(fake_pair)

            loss_adv = sum(HingeGANLoss.g_loss(f[-1]) for f in fake_feats_g)
            loss_fm = feature_matching_loss(real_feats_g, fake_feats_g, num_D=num_d, n_layers_D=n_layers_d)
            loss_vgg = vgg_loss_fn(fake_images, real_images)

            loss_g = loss_adv + lambda_feat * loss_fm + lambda_vgg * loss_vgg

        scaler_g.scale(loss_g).backward()
        scaler_g.step(optimizer_g)
        scaler_g.update()

        running["g_loss"] += loss_g.item()
        running["adv_loss"] += loss_adv.item()
        running["fm_loss"] += loss_fm.item()
        running["vgg_loss"] += loss_vgg.item()
        global_step += 1

        if global_step % log_every == 0:
            log_dict = {
                "step_g_loss": loss_g.item(), "step_adv_loss": loss_adv.item(),
                "step_fm_loss": loss_fm.item(), "step_vgg_loss": loss_vgg.item(),
            }
            d_loss_str = "n/a (skipped this step)"
            if do_d_update:
                log_dict["step_d_loss"] = loss_d.item()
                d_loss_str = f"{loss_d.item():.4f}"
            mlflow.log_metrics(log_dict, step=global_step)
            print(
                f"  epoch {epoch} step {step}/{len(loader)} | "
                f"g_loss={loss_g.item():.4f} d_loss={d_loss_str} "
                f"(adv={loss_adv.item():.4f} fm={loss_fm.item():.4f} vgg={loss_vgg.item():.4f})"
            )

    n = len(loader)
    epoch_metrics = {k: v / n for k, v in running.items() if k != "d_loss"}
    # d_loss only accumulated on steps where D actually updated - average
    # over d_updates_this_epoch, not n, or this silently understates the
    # true per-update d_loss by a factor of d_update_every.
    epoch_metrics["d_loss"] = running["d_loss"] / max(d_updates_this_epoch, 1)
    return epoch_metrics, global_step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr-g", type=float, default=4e-4)
    parser.add_argument(
        "--lr-d", type=float, default=1e-4,
        help=(
            "Deliberately lower than --lr-g (TTUR-style asymmetric learning "
            "rates). Diagnosed from the 25-epoch smoke test: d_loss fell "
            "monotonically (2.03->1.00) while g_loss climbed monotonically "
            "(16.40->19.04) and vgg_loss stayed flat (~0.51) across all 5 "
            "sampled epochs, with no reversal - the textbook signature of "
            "the discriminator steadily overpowering the generator under "
            "hinge loss (unlike BCE, hinge g_loss=-D(fake).mean() has no "
            "upper saturation point, so it grows without bound as D's "
            "margin widens). Slowing D relative to G is the standard, "
            "cheapest first fix for this specific pattern - see chat "
            "discussion when this was diagnosed."
        ),
    )
    parser.add_argument("--ngf", type=int, default=32)
    parser.add_argument("--ndf", type=int, default=64)
    parser.add_argument("--num-d", type=int, default=2)
    parser.add_argument("--n-layers-d", type=int, default=4)
    parser.add_argument(
        "--d-update-every", type=int, default=2,
        help=(
            "Update the discriminator only every N steps (G still updates "
            "every step). Default 2 - see train_one_epoch docstring/comment: "
            "TTUR (asymmetric lr_g/lr_d) alone did not stop d_loss falling / "
            "g_loss rising monotonically across two separate 25-epoch smoke "
            "tests, so this halves D's effective update rate as the next "
            "escalation. Set to 1 to disable (update D every step, the "
            "original behavior)."
        ),
    )
    parser.add_argument("--dropout-p", type=float, default=0.5)
    parser.add_argument("--lambda-feat", type=float, default=10.0)
    parser.add_argument("--lambda-vgg", type=float, default=10.0)
    parser.add_argument("--vgg-weights-path", type=str, default="/opt/steel-defect-data/pretrained/vgg19-dcbb9e9d.pth")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--class-target-id", type=int, default=2)
    parser.add_argument("--class-target-fraction", type=float, default=0.15)
    parser.add_argument("--splits-dir", type=str, default="data/splits")
    parser.add_argument("--raw-images-dir", type=str, default="data/raw/severstal-steel-defect-detection")
    parser.add_argument("--processed-dir", type=str, default="data/processed")
    parser.add_argument("--checkpoint-dir", type=str, default="experiments/checkpoints/spade")
    parser.add_argument("--samples-dir", type=str, default="experiments/samples/spade")
    parser.add_argument("--experiment-name", type=str, default="spade")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--sample-every", type=int, default=5)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    checkpoint_dir = Path(args.checkpoint_dir)
    samples_dir = Path(args.samples_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)

    train_ds = Pix2PixSteelDataset(
        split_csv=f"{args.splits_dir}/train.csv",
        images_root=args.raw_images_dir,
        masks_root=args.processed_dir,
    )
    val_ds = Pix2PixSteelDataset(
        split_csv=f"{args.splits_dir}/val.csv",
        images_root=args.raw_images_dir,
        masks_root=args.processed_dir,
        horizontal_flip_prob=0.0,
    )

    target_ids = get_class_image_ids(f"{args.raw_images_dir}/train.csv", class_id=args.class_target_id)
    sampler = build_class_weighted_sampler(
        train_ds, target_ids, target_fraction=args.class_target_fraction,
        class_label=f"class_{args.class_target_id}",
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    fixed_val_batch = next(iter(DataLoader(val_ds, batch_size=4, shuffle=True)))

    generator = build_generator(ngf=args.ngf, dropout_p=args.dropout_p).to(device)
    discriminator = build_discriminator(ndf=args.ndf, n_layers=args.n_layers_d, num_D=args.num_d).to(device)
    vgg_loss_fn = VGGLoss(weights_path=Path(args.vgg_weights_path)).to(device)

    optimizer_g = torch.optim.Adam(generator.parameters(), lr=args.lr_g, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr_d, betas=(0.5, 0.999))
    scaler_g = torch.amp.GradScaler("cuda")
    scaler_d = torch.amp.GradScaler("cuda")

    mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run():
        mlflow.log_params(vars(args))

        n_params_g = sum(p.numel() for p in generator.parameters())
        n_params_d = sum(p.numel() for p in discriminator.parameters())
        mlflow.log_params({"generator_params": n_params_g, "discriminator_params": n_params_d})
        print(f"Generator: {n_params_g/1e6:.2f}M params | Discriminator: {n_params_d/1e6:.2f}M params")

        global_step = 0
        for epoch in range(1, args.epochs + 1):
            epoch_metrics, global_step = train_one_epoch(
                generator, discriminator, train_loader, optimizer_g, optimizer_d,
                scaler_g, scaler_d, vgg_loss_fn, args.lambda_feat, args.lambda_vgg,
                args.num_d, args.n_layers_d, args.d_update_every, device,
                args.log_every, epoch, global_step,
            )

            print(
                f"Epoch {epoch}/{args.epochs} | g_loss={epoch_metrics['g_loss']:.4f} "
                f"d_loss={epoch_metrics['d_loss']:.4f} vgg={epoch_metrics['vgg_loss']:.4f}"
            )
            mlflow.log_metrics({f"epoch_{k}": v for k, v in epoch_metrics.items()}, step=epoch)

            if epoch % args.sample_every == 0 or epoch == 1:
                sample_path = samples_dir / f"epoch_{epoch:04d}.png"
                save_sample_grid(generator, fixed_val_batch, device, sample_path)
                mlflow.log_artifact(str(sample_path))
                print(f"  Saved sample grid: {sample_path}")

            if epoch % args.checkpoint_every == 0 or epoch == args.epochs:
                gen_path = checkpoint_dir / f"generator_epoch{epoch:04d}.pth"
                disc_path = checkpoint_dir / f"discriminator_epoch{epoch:04d}.pth"
                torch.save(generator.state_dict(), gen_path)
                torch.save(discriminator.state_dict(), disc_path)
                print(f"  Saved checkpoint: {gen_path}")

        torch.save(generator.state_dict(), checkpoint_dir / "generator_last.pth")
        torch.save(discriminator.state_dict(), checkpoint_dir / "discriminator_last.pth")

    print("Training finished.")


if __name__ == "__main__":
    main()
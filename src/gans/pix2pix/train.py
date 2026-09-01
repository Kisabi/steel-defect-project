"""
Training script for Pix2Pix: mask-conditioned synthetic steel defect generation.

Standard Pix2Pix training recipe (Isola et al.):
- G_loss = adversarial(BCEWithLogits, patch-wise) + lambda_l1 * L1(fake, real)
- D_loss = 0.5 * (BCE(D(real), 1) + BCE(D(fake.detach()), 0))
- Adam, lr=2e-4, betas=(0.5, 0.999) for both G and D - beta1=0.9 (the Adam
  default) makes GAN training visibly unstable; 0.5 is the standard fix.

Uses the class_2-weighted sampler from src/gans/pix2pix/dataset.py (see
progress_log.md for why class_2, not class_3, is the oversampling target).

Logs scalar losses to MLflow every --log-every steps, and saves a
real-mask / fake-image / real-image comparison grid every
--sample-every epochs, so training can be checked qualitatively without
waiting for a full run (useful now that Jupyter is set up - open the
saved PNGs there).

Usage (inside the container, from /workspace):
    python -m src.gans.pix2pix.train \
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

from src.gans.pix2pix.dataset import (
    Pix2PixSteelDataset,
    build_class_weighted_sampler,
    get_class_image_ids,
)
from src.gans.pix2pix.discriminator import build_discriminator
from src.gans.pix2pix.generator import build_generator, init_weights


def denormalize(image_tensor: torch.Tensor) -> torch.Tensor:
    """[-1, 1] -> [0, 1], for saving/visualizing generator output."""
    return (image_tensor.clamp(-1, 1) + 1) / 2


def save_sample_grid(generator, val_batch, device, out_path: Path):
    """Save a (mask-as-RGB, fake, real) comparison grid for a fixed validation batch."""
    generator.eval()
    with torch.no_grad():
        masks = val_batch["mask"].to(device)
        real_images = val_batch["image"].to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            fake_images = generator(masks)

    # collapse the 4-channel one-hot mask into a single-channel visual (argmax class, scaled to [0,1])
    mask_vis = masks.argmax(dim=1, keepdim=True).float() / masks.shape[1]
    mask_vis = mask_vis.repeat(1, 3, 1, 1)  # 1 channel -> 3 for consistent grid stacking

    grid_rows = torch.cat([mask_vis, denormalize(fake_images.float()), denormalize(real_images)], dim=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vutils.save_image(grid_rows, out_path, nrow=masks.shape[0])
    generator.train()


def train_one_epoch(
    generator, discriminator, loader, optimizer_g, optimizer_d,
    scaler_g, scaler_d, adversarial_loss, l1_loss, lambda_l1, device,
    log_every: int, epoch: int, global_step: int,
):
    generator.train()
    discriminator.train()

    running_g_loss = running_d_loss = running_l1 = running_adv = 0.0

    for step, batch in enumerate(loader):
        masks = batch["mask"].to(device)
        real_images = batch["image"].to(device)

        # --- discriminator step ---
        optimizer_d.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            fake_images = generator(masks)

            real_pair = torch.cat([masks, real_images], dim=1)
            fake_pair = torch.cat([masks, fake_images.detach()], dim=1)

            pred_real = discriminator(real_pair)
            pred_fake = discriminator(fake_pair)

            loss_d_real = adversarial_loss(pred_real, torch.ones_like(pred_real))
            loss_d_fake = adversarial_loss(pred_fake, torch.zeros_like(pred_fake))
            loss_d = 0.5 * (loss_d_real + loss_d_fake)

        scaler_d.scale(loss_d).backward()
        scaler_d.step(optimizer_d)
        scaler_d.update()

        # --- generator step ---
        optimizer_g.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            fake_pair = torch.cat([masks, fake_images], dim=1)
            pred_fake_for_g = discriminator(fake_pair)

            loss_adv = adversarial_loss(pred_fake_for_g, torch.ones_like(pred_fake_for_g))
            loss_l1 = l1_loss(fake_images, real_images) * lambda_l1
            loss_g = loss_adv + loss_l1

        scaler_g.scale(loss_g).backward()
        scaler_g.step(optimizer_g)
        scaler_g.update()

        running_g_loss += loss_g.item()
        running_d_loss += loss_d.item()
        running_l1 += loss_l1.item()
        running_adv += loss_adv.item()
        global_step += 1

        if global_step % log_every == 0:
            mlflow.log_metrics(
                {
                    "step_g_loss": loss_g.item(),
                    "step_d_loss": loss_d.item(),
                    "step_l1_loss": loss_l1.item(),
                    "step_adv_loss": loss_adv.item(),
                },
                step=global_step,
            )
            print(
                f"  epoch {epoch} step {step}/{len(loader)} | "
                f"g_loss={loss_g.item():.4f} d_loss={loss_d.item():.4f} "
                f"(l1={loss_l1.item():.4f} adv={loss_adv.item():.4f})"
            )

    n = len(loader)
    return {
        "g_loss": running_g_loss / n,
        "d_loss": running_d_loss / n,
        "l1_loss": running_l1 / n,
        "adv_loss": running_adv / n,
    }, global_step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lambda-l1", type=float, default=100.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--class-target-id", type=int, default=2)
    parser.add_argument("--class-target-fraction", type=float, default=0.15)
    parser.add_argument("--splits-dir", type=str, default="data/splits")
    parser.add_argument("--raw-images-dir", type=str, default="data/raw/severstal-steel-defect-detection")
    parser.add_argument("--processed-dir", type=str, default="data/processed")
    parser.add_argument("--checkpoint-dir", type=str, default="experiments/checkpoints/pix2pix")
    parser.add_argument("--samples-dir", type=str, default="experiments/samples/pix2pix")
    parser.add_argument("--experiment-name", type=str, default="pix2pix")
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
        horizontal_flip_prob=0.0,  # deterministic for visual comparison across epochs
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
    # fixed small validation batch, reused every epoch for a stable qualitative check
    fixed_val_batch = next(iter(DataLoader(val_ds, batch_size=4, shuffle=True)))

    generator = build_generator().to(device)
    discriminator = build_discriminator().to(device)
    generator.apply(init_weights)
    discriminator.apply(init_weights)

    optimizer_g = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    scaler_g = torch.amp.GradScaler("cuda")
    scaler_d = torch.amp.GradScaler("cuda")

    adversarial_loss = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()

    mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run():
        mlflow.log_params(vars(args))

        global_step = 0
        for epoch in range(1, args.epochs + 1):
            epoch_metrics, global_step = train_one_epoch(
                generator, discriminator, train_loader, optimizer_g, optimizer_d,
                scaler_g, scaler_d, adversarial_loss, l1_loss, args.lambda_l1, device,
                args.log_every, epoch, global_step,
            )

            print(
                f"Epoch {epoch}/{args.epochs} | g_loss={epoch_metrics['g_loss']:.4f} "
                f"d_loss={epoch_metrics['d_loss']:.4f} l1={epoch_metrics['l1_loss']:.4f}"
            )
            mlflow.log_metrics(
                {f"epoch_{k}": v for k, v in epoch_metrics.items()}, step=epoch
            )

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

        # always keep a "latest" pair for convenience (e.g. for the sampling/inference script)
        torch.save(generator.state_dict(), checkpoint_dir / "generator_last.pth")
        torch.save(discriminator.state_dict(), checkpoint_dir / "discriminator_last.pth")

    print("Training finished.")


if __name__ == "__main__":
    main()
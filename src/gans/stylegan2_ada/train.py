"""
Training script for the mask-conditioned StyleGAN2-ADA (own implementation).

Recipe (follows StyleGAN2-ADA, adapted to this project):
- Non-saturating logistic loss: G: softplus(-D(fake)), D: softplus(D(fake)) + softplus(-D(real)).
- Lazy R1 regularization every --r1-interval D steps (separate D phase).
- ADA: AdaptiveAugment is applied to every (mask, image) pair that goes into D
  (real and fake), p is steered by r_t = mean(sign(D(real))).
- G EMA (G_ema) is what gets evaluated / saved as generator_epochXXXX.pth.
- Adam lr=--lr, betas=(0, 0.99); D lr/betas are adjusted for lazy regularization.
- fp16 autocast + GradScaler for the main phases; the R1 phase runs in fp32.

Deliberately NOT implemented (scope / memory at 1600x256): path length
regularization, style mixing.

Deviation: R1 is computed w.r.t. the already-augmented real image, which avoids
a double backward through grid_sample (slow / fragile on ROCm).

Uses the same class_2-weighted sampler as Pix2Pix/SPADE (see progress_log.md).

Usage (inside the container, from /workspace):
    python -m src.gans.stylegan2_ada.train --dry-run-limit 20     # smoke check
    python -m src.gans.stylegan2_ada.train --epochs 100 --batch-size 2
"""

import argparse
import copy
import math
import subprocess
from pathlib import Path

import mlflow
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from torch.utils.data import DataLoader
import time

from src.gans.pix2pix.dataset import (
    Pix2PixSteelDataset,
    build_class_weighted_sampler,
    get_class_image_ids,
)
from src.gans.stylegan2_ada.augment import AdaptiveAugment
from src.gans.stylegan2_ada.discriminator import build_discriminator
from src.gans.stylegan2_ada.generator import build_generator


def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def denormalize(image_tensor: torch.Tensor) -> torch.Tensor:
    """[-1, 1] -> [0, 1], for saving/visualizing generator output."""
    return (image_tensor.clamp(-1, 1) + 1) / 2


def mask_to_vis(masks: torch.Tensor) -> torch.Tensor:
    """4-channel one-hot mask -> 3-channel gray image (background 0, class k -> k/4)."""
    ids = torch.arange(1, masks.shape[1] + 1, device=masks.device, dtype=masks.dtype).view(1, -1, 1, 1)
    vis = (masks * ids).sum(dim=1, keepdim=True) / masks.shape[1]
    return vis.repeat(1, 3, 1, 1)


def set_requires_grad(model, flag: bool):
    for p in model.parameters():
        p.requires_grad_(flag)


@torch.no_grad()
def update_ema(g_ema, g, beta: float):
    for p_ema, p in zip(g_ema.parameters(), g.parameters()):
        p_ema.copy_(p.lerp(p_ema, beta))
    for b_ema, b in zip(g_ema.buffers(), g.buffers()):
        b_ema.copy_(b)


@torch.no_grad()
def save_sample_grid(g_ema, val_batch, fixed_z, device, out_path: Path):
    """Save a (mask, fake from G_ema with fixed z, real) comparison grid."""
    masks = val_batch["mask"].to(device)
    real_images = val_batch["image"].to(device)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        fake_images = g_ema(masks, fixed_z)
    rows = torch.cat([mask_to_vis(masks), denormalize(fake_images.float()), denormalize(real_images)], dim=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vutils.save_image(rows, out_path, nrow=masks.shape[0])


def train_one_epoch(G, D, G_ema, opt_g, opt_d, scaler_g, scaler_d, ada, loader,
                    args, device, epoch, global_step):
    G.train()
    D.train()
    sum_g = sum_d = sum_r1 = 0.0
    n_steps = n_r1 = 0
    t_last, step_last = time.time(), global_step
    sum_rt = 0.0

    for step, batch in enumerate(loader):
        if args.dry_run_limit and step >= args.dry_run_limit:
            break
        masks = batch["mask"].to(device, non_blocking=True)
        real = batch["image"].to(device, non_blocking=True)
        bsz = masks.shape[0]

        # ---------------- D main phase ----------------
        set_requires_grad(D, True)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
            fake = G(masks)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            real_logits = D(ada(torch.cat([masks, real], dim=1)))
            fake_logits = D(ada(torch.cat([masks, fake], dim=1)))
            loss_d = F.softplus(fake_logits).mean() + F.softplus(-real_logits).mean()
        opt_d.zero_grad(set_to_none=True)
        scaler_d.scale(loss_d).backward()
        scaler_d.step(opt_d)
        scaler_d.update()
        ada.update(real_logits)
        sum_rt += ada.last_rt

        # ---------------- D lazy R1 phase ----------------
        if args.r1_gamma > 0 and (global_step + 1) % args.r1_interval == 0:
            with torch.no_grad():
                aug_pair = ada(torch.cat([masks, real], dim=1))
            aug_img = aug_pair[:, masks.shape[1]:].detach().requires_grad_(True)
            pair_in = torch.cat([aug_pair[:, : masks.shape[1]].detach(), aug_img], dim=1)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.r1_fp16):
                logits = D(pair_in)
                (grad,) = torch.autograd.grad(logits.sum(), aug_img, create_graph=True)
                r1 = grad.square().sum(dim=(1, 2, 3)).mean()
                loss_r1 = r1 * (args.r1_gamma / 2.0) * args.r1_interval
            opt_d.zero_grad(set_to_none=True)
            scaler_d.scale(loss_r1).backward()
            scaler_d.step(opt_d)
            scaler_d.update()
            sum_r1 += r1.item()
            n_r1 += 1

        # ---------------- G phase ----------------
        set_requires_grad(D, False)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            fake = G(masks)
            fake_logits = D(ada(torch.cat([masks, fake], dim=1)))
            loss_g = F.softplus(-fake_logits).mean()
        opt_g.zero_grad(set_to_none=True)
        scaler_g.scale(loss_g).backward()
        scaler_g.step(opt_g)
        scaler_g.update()
        set_requires_grad(D, True)

        # ---------------- G EMA ----------------
        global_step += 1
        cur_nimg = global_step * bsz
        ema_nimg = args.ema_kimg * 1000.0
        if args.ema_rampup is not None:
            ema_nimg = min(ema_nimg, cur_nimg * args.ema_rampup)
        update_ema(G_ema, G, 0.5 ** (bsz / max(ema_nimg, 1e-8)))

        g_val, d_val = loss_g.item(), loss_d.item()
        if not (math.isfinite(g_val) and math.isfinite(d_val)):
            raise RuntimeError(f"Non-finite loss at step {global_step}: g={g_val} d={d_val}")
        sum_g += g_val
        sum_d += d_val
        n_steps += 1

        if global_step % args.log_every == 0:
            now = time.time()
            s_per_step = (now - t_last) / (global_step - step_last)
            t_last, step_last = now, global_step
            mlflow.log_metrics(
                {"step_g_loss": g_val, "step_d_loss": d_val, "step_ada_p": ada.p,
                 "step_ada_rt": ada.last_rt, "step_seconds": s_per_step},
                step=global_step,
            )
            print(
                f"  epoch {epoch} step {step}/{len(loader)} | g={g_val:.4f} d={d_val:.4f} "
                f"| p={ada.p:.4f} r_t={ada.last_rt:.3f} | {s_per_step:.2f} s/step"
            )

    metrics = {
        "g_loss": sum_g / max(n_steps, 1),
        "d_loss": sum_d / max(n_steps, 1),
        "ada_p": ada.p,
        "ada_rt": sum_rt / max(n_steps, 1),
    }
    if n_r1:
        metrics["r1"] = sum_r1 / n_r1
    return metrics, global_step


def worker_init_fn(worker_id):
    # OpenCV's own thread pool can deadlock inside forked DataLoader workers
    import cv2
    cv2.setNumThreads(0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--r1-gamma", type=float, default=10.0)
    parser.add_argument("--r1-interval", type=int, default=16)
    parser.add_argument("--ada-target", type=float, default=0.6)
    parser.add_argument("--ada-interval", type=int, default=4)
    parser.add_argument("--ada-kimg", type=float, default=100)
    parser.add_argument("--ada-p-init", type=float, default=0.0)
    parser.add_argument("--ema-kimg", type=float, default=None, help="default: 10 * batch_size / 32")
    parser.add_argument("--ema-rampup", type=float, default=0.05)
    parser.add_argument("--z-dim", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--class-target-id", type=int, default=2)
    parser.add_argument("--class-target-fraction", type=float, default=0.15)
    parser.add_argument("--splits-dir", type=str, default="data/splits")
    parser.add_argument("--raw-images-dir", type=str, default="data/raw/severstal-steel-defect-detection")
    parser.add_argument("--processed-dir", type=str, default="data/processed")
    parser.add_argument("--checkpoint-dir", type=str, default="experiments/checkpoints/stylegan2_ada")
    parser.add_argument("--samples-dir", type=str, default="experiments/samples/stylegan2_ada")
    parser.add_argument("--experiment-name", type=str, default="stylegan2_ada")
    parser.add_argument("--run-name", type=str, default="stylegan2-ada")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--sample-every", type=int, default=5)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--resume", type=str, default=None, help="path to state_last.pth")
    parser.add_argument("--dry-run-limit", type=int, default=0, help="stop each epoch after N steps (smoke check)")
    parser.add_argument("--r1-fp16", action="store_true",
                    help="run the R1 phase under fp16 autocast (fp32 double backward is very slow on this ROCm setup)")
    args = parser.parse_args()

    if args.ema_kimg is None:
        args.ema_kimg = 10.0 * args.batch_size / 32.0
    if args.dry_run_limit:
        if not args.resume:
            args.epochs = 1
        args.experiment_name += "-dryrun"

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
        persistent_workers=True, timeout=300, worker_init_fn=worker_init_fn,
    )
    fixed_val_batch = next(iter(DataLoader(
        val_ds, batch_size=4, shuffle=True, generator=torch.Generator().manual_seed(0)
    )))
    fixed_z = torch.randn(4, args.z_dim, device=device, generator=torch.Generator(device=device).manual_seed(0))

    G = build_generator(z_dim=args.z_dim).to(device)
    D = build_discriminator().to(device)
    G_ema = copy.deepcopy(G).eval().requires_grad_(False)

    # lazy regularization: scale D's lr / betas by r1_interval / (r1_interval + 1)
    d_ratio = args.r1_interval / (args.r1_interval + 1)
    opt_g = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.0, 0.99), eps=1e-8)
    opt_d = torch.optim.Adam(
        D.parameters(), lr=args.lr * d_ratio, betas=(0.0, 0.99 ** d_ratio), eps=1e-8
    )
    scaler_g = torch.amp.GradScaler("cuda")
    scaler_d = torch.amp.GradScaler("cuda")
    ada = AdaptiveAugment(
        p_init=args.ada_p_init, target_rt=args.ada_target,
        interval=args.ada_interval, ada_kimg=args.ada_kimg,
    )

    start_epoch, global_step = 1, 0
    if args.resume:
        state = torch.load(args.resume, map_location=device)
        G.load_state_dict(state["G"])
        D.load_state_dict(state["D"])
        G_ema.load_state_dict(state["G_ema"])
        opt_g.load_state_dict(state["opt_g"])
        opt_d.load_state_dict(state["opt_d"])
        ada.load_state_dict(state["ada"])
        start_epoch, global_step = state["epoch"] + 1, state["global_step"]
        print(f"Resumed from {args.resume}: epoch {state['epoch']}, p={ada.p:.4f}")

    mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run(run_name=args.run_name):
        mlflow.set_tags({
            "stage": "train",
            "generator_variant": "stylegan2_ada",
            "git_commit": get_git_commit(),
        })
        mlflow.log_params(vars(args))
        mlflow.log_params({
            "generator_params": sum(p.numel() for p in G.parameters()),
            "discriminator_params": sum(p.numel() for p in D.parameters()),
        })

        for epoch in range(start_epoch, args.epochs + 1):
            epoch_metrics, global_step = train_one_epoch(
                G, D, G_ema, opt_g, opt_d, scaler_g, scaler_d, ada, train_loader,
                args, device, epoch, global_step,
            )
            print(
                f"Epoch {epoch}/{args.epochs} | g={epoch_metrics['g_loss']:.4f} "
                f"d={epoch_metrics['d_loss']:.4f} | p={epoch_metrics['ada_p']:.4f} "
                f"r_t={epoch_metrics['ada_rt']:.3f} r1={epoch_metrics.get('r1', float('nan')):.4f}"
            )
            mlflow.log_metrics({f"epoch_{k}": v for k, v in epoch_metrics.items()}, step=epoch)

            if epoch % args.sample_every == 0 or epoch == 1:
                sample_path = samples_dir / f"epoch_{epoch:04d}.png"
                save_sample_grid(G_ema, fixed_val_batch, fixed_z, device, sample_path)
                mlflow.log_artifact(str(sample_path))
                print(f"  Saved sample grid: {sample_path}")

            if epoch % args.checkpoint_every == 0 or epoch == args.epochs:
                # EMA generator = the weights used for FID / synthetic data generation
                torch.save(G_ema.state_dict(), checkpoint_dir / f"generator_epoch{epoch:04d}.pth")
                # single overwritten file with the full training state, for --resume
                torch.save(
                    {
                        "G": G.state_dict(), "D": D.state_dict(), "G_ema": G_ema.state_dict(),
                        "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(),
                        "ada": ada.state_dict(), "epoch": epoch, "global_step": global_step,
                    },
                    checkpoint_dir / "state_last.pth",
                )
                print(f"  Saved checkpoint: generator_epoch{epoch:04d}.pth")

        torch.save(G_ema.state_dict(), checkpoint_dir / "generator_last.pth")
        torch.save(D.state_dict(), checkpoint_dir / "discriminator_last.pth")

    print("Training finished.")


if __name__ == "__main__":
    main()
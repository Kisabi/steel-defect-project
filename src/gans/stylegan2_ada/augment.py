"""
Own differentiable ADA (Adaptive Discriminator Augmentation) for the
StyleGAN2 discriminator, pure PyTorch (no custom CUDA ops).

The pipeline is applied to the concatenated (mask, image) pair, shape
(B, 7, H, W): channels [0:4] = one-hot mask, channels [4:7] = RGB in [-1, 1].
Geometric ops are applied identically to mask and image (one resampling of
all channels), colour ops touch only the image channels. The same pipeline
with the same p is applied to real AND fake pairs; gradients flow through it
to the generator.

Ops (each applied per-sample with probability p, as in the official ADA):
  geometric: x-flip, y-flip, integer translation, isotropic scale,
             small rotation, anisotropic scale, fractional translation
  colour:    brightness, contrast

Deliberately NOT included (vs. the official 'bgc' pipeline):
- rotate90: would change the 1600x256 aspect ratio.
- hue / saturation / luma flip: low value for near-gray steel images.
- filtering, noise, cutout: not part of 'bgc'.

Special care for the non-square 1600x256 frame:
- The affine matrices are built in an aspect-corrected space (x scaled by
  W/H), otherwise rotation/aniso would shear the image in normalized coords.
- Rotation is tiny by default (rotate_max=0.01 -> about +-1.8 deg): at 6.25:1
  aspect even a few degrees move the strip ends by tens of pixels.
- Reflection padding mirrors the mask together with the image, so they stay
  consistent. The mask is resampled bilinearly, so it becomes slightly soft.

Controller: every `interval` D steps, r_t = mean(sign(D(real))) is compared
to `target_rt`; p moves up/down so that it can span 0..1 in `ada_kimg`
thousand images.

Check:
    python -m src.gans.stylegan2_ada.augment
"""

import math

import torch
import torch.nn.functional as F


class AdaptiveAugment:
    def __init__(
        self,
        p_init=0.0,
        target_rt=0.6,
        interval=4,
        ada_kimg=100,
        mask_ch=4,
        xint_max=0.125,
        scale_std=0.2,
        rotate_max=0.01,  # fraction of pi
        aniso_std=0.2,
        xfrac_std=0.125,
        brightness_std=0.2,
        contrast_std=0.5,
    ):
        self.p = float(p_init)
        self.target_rt = target_rt
        self.interval = interval
        self.ada_kimg = ada_kimg
        self.mask_ch = mask_ch
        self.xint_max = xint_max
        self.scale_std = scale_std
        self.rotate_max = rotate_max
        self.aniso_std = aniso_std
        self.xfrac_std = xfrac_std
        self.brightness_std = brightness_std
        self.contrast_std = contrast_std

        self._sign_sum = 0.0
        self._count = 0
        self._steps = 0
        self.last_rt = 0.0

    # ----- controller -----

    def update(self, real_logits: torch.Tensor):
        """Call once per D step with D's logits on AUGMENTED real pairs, shape (B, 1)."""
        # accumulate on-device, sync with the GPU only once per interval
        self._sign_sum = self._sign_sum + torch.sign(real_logits.detach().float()).sum()
        self._count += real_logits.numel()
        self._steps += 1
        if self._steps % self.interval != 0:
            return

        rt = float(self._sign_sum) / self._count
        self.last_rt = rt
        batch = real_logits.shape[0]
        step = (batch * self.interval) / (self.ada_kimg * 1000.0)
        direction = 1.0 if rt > self.target_rt else -1.0
        self.p = min(max(self.p + direction * step, 0.0), 1.0)
        self._sign_sum, self._count = 0.0, 0

    def state_dict(self):
        return {"p": self.p}

    def load_state_dict(self, state):
        self.p = float(state["p"])

    # ----- augmentation -----

    def __call__(self, pair: torch.Tensor) -> torch.Tensor:
        p = self.p
        if p <= 0.0:
            return pair

        dtype = pair.dtype
        x = pair.float()  # resampling in fp32
        b, _, h, w = x.shape
        dev = x.device
        aspect = w / h

        def gate():
            """Per-sample Bernoulli(p) as float (b,)."""
            return (torch.rand(b, device=dev) < p).float()

        def coin():
            return torch.randint(0, 2, (b,), device=dev).float()

        # --- linear part: R @ diag(sx, sy) ---
        # flips: applied with prob p, direction is a coin flip (effective flip prob p/2)
        flip_x = 1.0 - 2.0 * gate() * coin()
        flip_y = 1.0 - 2.0 * gate() * coin()
        scale = torch.exp2(torch.randn(b, device=dev) * self.scale_std * gate())
        aniso = torch.exp2(torch.randn(b, device=dev) * self.aniso_std * gate())
        angle = (torch.rand(b, device=dev) * 2 - 1) * math.pi * self.rotate_max * gate()

        sx = flip_x * scale * aniso
        sy = flip_y * scale / aniso
        cos, sin = torch.cos(angle), torch.sin(angle)

        # conjugate with the aspect correction S = diag(W/H, 1): M_ij = L_ij * S_j / S_i
        m00 = cos * sx
        m01 = (-sin * sy) / aspect
        m10 = (sin * sx) * aspect
        m11 = cos * sy

        # --- translation, in normalized coords (full image span = 2) ---
        tx_pix = torch.round((torch.rand(b, device=dev) * 2 - 1) * self.xint_max * w) * gate()
        ty_pix = torch.round((torch.rand(b, device=dev) * 2 - 1) * self.xint_max * h) * gate()
        tx_frac = torch.randn(b, device=dev) * self.xfrac_std * gate()
        ty_frac = torch.randn(b, device=dev) * self.xfrac_std * gate()
        tx = 2.0 * (tx_pix / w + tx_frac)
        ty = 2.0 * (ty_pix / h + ty_frac)

        theta = torch.stack(
            [torch.stack([m00, m01, tx], dim=1), torch.stack([m10, m11, ty], dim=1)], dim=1
        )  # (b, 2, 3)
        grid = F.affine_grid(theta, list(x.shape), align_corners=False)
        x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)

        # --- colour ops on the image channels only ---
        mask, img = x[:, : self.mask_ch], x[:, self.mask_ch:]
        bright = (torch.randn(b, device=dev) * self.brightness_std * gate()).view(b, 1, 1, 1)
        contrast = torch.exp2(torch.randn(b, device=dev) * self.contrast_std * gate()).view(b, 1, 1, 1)
        mean = img.mean(dim=(1, 2, 3), keepdim=True)  # contrast around the per-image mean
        img = (img - mean) * contrast + mean + bright

        return torch.cat([mask, img], dim=1).to(dtype)


if __name__ == "__main__":
    pair = torch.randn(2, 7, 256, 1600)

    aug = AdaptiveAugment(p_init=0.0)
    print("p=0 returns the input unchanged:", aug(pair) is pair)

    aug.p = 1.0
    out = aug(pair)
    print("p=1 shape:", tuple(out.shape), "| mean |diff|:", round((out - pair).abs().mean().item(), 4))

    # gradients must flow through the augmentation to the image (and so to G)
    x = pair.clone().requires_grad_(True)
    aug(x).mean().backward()
    print("grad flows:", x.grad is not None and x.grad.abs().sum().item() > 0)

    # controller: D too confident on real (all logits > 0) -> p must grow
    ctrl = AdaptiveAugment(p_init=0.0, ada_kimg=1, interval=4)
    for _ in range(8):
        ctrl.update(torch.ones(2, 1))
    print("p after 8 'overfitting' steps:", round(ctrl.p, 4))  # expected 0.016
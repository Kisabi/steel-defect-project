"""
Mask-conditioned StyleGAN2 generator (own pure-PyTorch implementation).

Differences from the official unconditional StyleGAN2 (see progress_log.md):
- No constant input. A small conv encoder turns the 4-channel one-hot mask
  into the 4x25 seed feature map (same seed resolution as Pix2Pix/SPADE).
- At every resolution the mask is resized (area average) and concatenated to
  the features before each modulated conv, so spatial defect layout is not
  lost (a global style vector w cannot carry it).
- Style w comes from z via the mapping network. A single w is used for all
  layers (no style mixing). Sample diversity comes from z and per-pixel noise,
  not from Dropout as in Pix2Pix/SPADE.
- Upsampling is bilinear interpolation before the conv (see layers.py).
- Output is unbounded (no tanh), as in StyleGAN2; clamp to [-1, 1] when saving.

Resolutions (6 doublings, 25:4 aspect kept everywhere):
    4x25 -> 8x50 -> 16x100 -> 32x200 -> 64x400 -> 128x800 -> 256x1600

Check shapes / params:
    python -m src.gans.stylegan2_ada.generator
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.gans.stylegan2_ada.layers import (
    BiasLeakyReLU,
    EqualizedConv2d,
    EqualizedLinear,
    ModulatedConv2d,
    NoiseInjection,
)


class MappingNetwork(nn.Module):
    """z -> w: pixel-norm on z, then an MLP with reduced learning rate (lr_mul)."""

    def __init__(self, z_dim=512, w_dim=512, num_layers=4, lr_mul=0.01):
        super().__init__()
        dims = [z_dim] + [w_dim] * num_layers
        self.layers = nn.ModuleList(
            [EqualizedLinear(dims[i], dims[i + 1], lr_mul=lr_mul) for i in range(num_layers)]
        )

    def forward(self, z):
        x = z * torch.rsqrt(z.pow(2).mean(dim=1, keepdim=True) + 1e-8)
        for layer in self.layers:
            x = F.leaky_relu(layer(x), 0.2) * math.sqrt(2.0)
        return x


class MaskEncoder(nn.Module):
    """One-hot mask (B, 4, 256, 1600) -> seed features (B, C, 4, 25).

    Six stride-2 convs: 256x1600 -> 4x25 (1600/64 = 25, 256/64 = 4).
    """

    def __init__(self, mask_ch=4, channels=(32, 64, 128, 256, 512, 512)):
        super().__init__()
        layers = []
        in_ch = mask_ch
        for out_ch in channels:
            layers += [
                EqualizedConv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False),
                BiasLeakyReLU(out_ch),
            ]
            in_ch = out_ch
        self.net = nn.Sequential(*layers)
        self.out_channels = in_ch

    def forward(self, mask):
        return self.net(mask)


class StyledConv(nn.Module):
    """concat(mask) -> modulated conv 3x3 -> noise -> bias + LeakyReLU."""

    def __init__(self, in_ch, out_ch, w_dim, mask_ch):
        super().__init__()
        self.conv = ModulatedConv2d(in_ch + mask_ch, out_ch, 3, w_dim)
        self.noise = NoiseInjection()
        self.act = BiasLeakyReLU(out_ch)

    def forward(self, x, mask, w):
        x = torch.cat([x, mask.to(x.dtype)], dim=1)
        x = self.conv(x, w)
        x = self.noise(x)
        return self.act(x)


class ToRGB(nn.Module):
    """1x1 modulated conv (no demodulation) + skip connection from the previous resolution."""

    def __init__(self, in_ch, w_dim):
        super().__init__()
        self.conv = ModulatedConv2d(in_ch, 3, 1, w_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, x, w, skip=None):
        out = self.conv(x, w) + self.bias.to(x.dtype)
        if skip is not None:
            skip = F.interpolate(skip, scale_factor=2, mode="bilinear", align_corners=False)
            out = out + skip.to(out.dtype)
        return out


class SynthesisBlock(nn.Module):
    """One resolution doubling: two styled convs + ToRGB (skip architecture)."""

    def __init__(self, in_ch, out_ch, w_dim, mask_ch):
        super().__init__()
        self.conv_a = StyledConv(in_ch, out_ch, w_dim, mask_ch)
        self.conv_b = StyledConv(out_ch, out_ch, w_dim, mask_ch)
        self.to_rgb = ToRGB(out_ch, w_dim)


class StyleGAN2Generator(nn.Module):
    def __init__(
        self,
        z_dim=512,
        w_dim=512,
        mask_ch=4,
        num_mapping_layers=4,
        # channels[0] = 4x25 level, channels[i] = i-th doubling (6 doublings -> 7 entries)
        channels=(512, 512, 512, 512, 352, 160, 64),
    ):
        super().__init__()
        self.z_dim = z_dim
        self.mask_ch = mask_ch

        self.mapping = MappingNetwork(z_dim, w_dim, num_mapping_layers)
        self.encoder = MaskEncoder(mask_ch, (32, 64, 128, 256, 512, channels[0]))

        self.conv0 = StyledConv(channels[0], channels[0], w_dim, mask_ch)
        self.to_rgb0 = ToRGB(channels[0], w_dim)
        self.blocks = nn.ModuleList(
            [SynthesisBlock(channels[i - 1], channels[i], w_dim, mask_ch) for i in range(1, len(channels))]
        )

    def forward(self, mask, z=None):
        """mask: (B, 4, 256, 1600) one-hot float. z: optional (B, z_dim); sampled if None."""
        if z is None:
            z = torch.randn(mask.shape[0], self.z_dim, device=mask.device, dtype=torch.float32)
        w = self.mapping(z)

        x = self.encoder(mask)  # (B, C0, 4, 25)
        m = F.adaptive_avg_pool2d(mask, x.shape[-2:])
        x = self.conv0(x, m, w)
        img = self.to_rgb0(x, w)

        for block in self.blocks:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            m = F.adaptive_avg_pool2d(mask, x.shape[-2:])
            x = block.conv_a(x, m, w)
            x = block.conv_b(x, m, w)
            img = block.to_rgb(x, w, skip=img)
        return img


def build_generator(**kwargs) -> StyleGAN2Generator:
    return StyleGAN2Generator(**kwargs)


def _count(module) -> float:
    return sum(p.numel() for p in module.parameters()) / 1e6


if __name__ == "__main__":
    gen = build_generator()
    print(f"Generator params: {_count(gen):.2f}M")
    print(f"  mapping: {_count(gen.mapping):.2f}M | encoder: {_count(gen.encoder):.2f}M | "
          f"conv0+to_rgb0: {_count(gen.conv0) + _count(gen.to_rgb0):.2f}M | "
          f"blocks: {_count(gen.blocks):.2f}M")

    labels = torch.randint(0, 5, (1, 256, 1600))
    mask = F.one_hot(labels, 5)[..., 1:].permute(0, 3, 1, 2).float()  # classes 1-4
    with torch.no_grad():
        out = gen(mask)
    print("Output:", tuple(out.shape))  # expected (1, 3, 256, 1600)
"""
StyleGAN2 residual discriminator (own pure-PyTorch implementation).

- Input: the concatenated (mask, image) pair, 7 channels (4 one-hot mask +
  3 RGB), same conditioning scheme as Pix2Pix / SPADE discriminators.
  ADA augmentations are applied to this pair BEFORE the discriminator
  (see augment.py), so mask and image stay spatially consistent.
- 6 residual blocks, each halving the resolution with an anti-aliasing blur:
  256x1600 -> 128x800 -> 64x400 -> 32x200 -> 16x100 -> 8x50 -> 4x25.
- Epilogue: minibatch std -> conv 3x3 -> 1x1 channel reduction -> FC -> logit.
  The 1x1 reduction keeps the FC layer small (4x25 = 100 positions would give
  a ~26M-parameter FC without it).

Check shapes / params:
    python -m src.gans.stylegan2_ada.discriminator
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.gans.stylegan2_ada.layers import (
    BiasLeakyReLU,
    Blur,
    EqualizedConv2d,
    EqualizedLinear,
)


class MinibatchStd(nn.Module):
    """Appends one channel with the average per-group feature std (anti mode-collapse).

    The batch is split into groups of up to `group_size` samples. With the
    small batch sizes used here (2-4) the group is the whole batch.
    """

    def __init__(self, group_size=4, eps=1e-8):
        super().__init__()
        self.group_size = group_size
        self.eps = eps

    def forward(self, x):
        b, c, h, w = x.shape
        g = min(self.group_size, b)
        while b % g != 0:  # largest divisor of b not exceeding group_size
            g -= 1
        y = x.float().reshape(g, b // g, c, h, w)
        y = y - y.mean(dim=0, keepdim=True)
        y = y.pow(2).mean(dim=0)  # (b/g, c, h, w)
        y = (y + self.eps).sqrt().mean(dim=(1, 2, 3))  # (b/g,)
        y = y.reshape(-1, 1, 1, 1).repeat(g, 1, h, w)  # back to batch order
        return torch.cat([x, y.to(x.dtype)], dim=1)


class DiscriminatorBlock(nn.Module):
    """Residual block with 2x downsampling: conv -> blur(stride 2) -> conv, plus a blurred 1x1 skip."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv0 = EqualizedConv2d(in_ch, in_ch, 3, padding=1, bias=False)
        self.act0 = BiasLeakyReLU(in_ch)
        self.blur = Blur(in_ch, stride=2)
        self.conv1 = EqualizedConv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.act1 = BiasLeakyReLU(out_ch)
        self.skip_blur = Blur(in_ch, stride=2)
        self.skip = EqualizedConv2d(in_ch, out_ch, 1, bias=False)

    def forward(self, x):
        y = self.act0(self.conv0(x))
        y = self.act1(self.conv1(self.blur(y)))  # blur first, then conv at half resolution
        s = self.skip(self.skip_blur(x))
        return (y + s) / math.sqrt(2.0)


class StyleGAN2Discriminator(nn.Module):
    def __init__(
        self,
        in_channels=7,
        # channels[0] = from-RGB level, channels[i] = output of i-th block (6 blocks -> 7 entries)
        channels=(64, 128, 256, 512, 512, 512, 512),
        final_size=(4, 25),
        reduce_ch=64,
    ):
        super().__init__()
        self.from_rgb = nn.Sequential(
            EqualizedConv2d(in_channels, channels[0], 1, bias=False),
            BiasLeakyReLU(channels[0]),
        )
        self.blocks = nn.Sequential(
            *[DiscriminatorBlock(channels[i], channels[i + 1]) for i in range(len(channels) - 1)]
        )

        ch = channels[-1]
        self.mbstd = MinibatchStd(group_size=4)
        self.conv = EqualizedConv2d(ch + 1, ch, 3, padding=1, bias=False)
        self.conv_act = BiasLeakyReLU(ch)
        self.reduce = EqualizedConv2d(ch, reduce_ch, 1, bias=False)
        self.reduce_act = BiasLeakyReLU(reduce_ch)
        self.fc = EqualizedLinear(reduce_ch * final_size[0] * final_size[1], ch)
        self.out = EqualizedLinear(ch, 1)

    def forward(self, pair):
        """pair: (B, 7, 256, 1600) = concat(mask one-hot, image). Returns logits (B, 1)."""
        x = self.from_rgb(pair)
        x = self.blocks(x)
        x = self.mbstd(x)
        x = self.conv_act(self.conv(x))
        x = self.reduce_act(self.reduce(x))
        x = F.leaky_relu(self.fc(x.flatten(1)), 0.2) * math.sqrt(2.0)
        return self.out(x)


def build_discriminator(**kwargs) -> StyleGAN2Discriminator:
    return StyleGAN2Discriminator(**kwargs)


if __name__ == "__main__":
    disc = build_discriminator()
    n_params = sum(p.numel() for p in disc.parameters()) / 1e6
    print(f"Discriminator params: {n_params:.2f}M")

    x = torch.randn(2, 7, 256, 1600)
    with torch.no_grad():
        feats = disc.blocks(disc.from_rgb(x))
        print("Features before epilogue:", tuple(feats.shape))  # expected (2, 512, 4, 25)
        print("Logits:", tuple(disc(x).shape))  # expected (2, 1)
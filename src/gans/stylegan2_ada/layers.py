"""
Basic building blocks for the own pure-PyTorch StyleGAN2-ADA implementation.

No custom CUDA ops (upfirdn2d / bias_act) are used: everything is built from
standard torch ops so it runs safely on ROCm.

Contents:
- EqualizedLinear / EqualizedConv2d: equalized learning rate (weights are
  stored ~N(0,1) and scaled at runtime by the He constant).
- BiasLeakyReLU: bias + LeakyReLU(0.2) * sqrt(2) (keeps activation variance).
- Blur: fixed [1,3,3,1] low-pass filter, optional stride 2 (used for
  downsampling in the discriminator).
- ModulatedConv2d: StyleGAN2 modulated/demodulated convolution
  (grouped-conv implementation), optional 2x upsampling.
- NoiseInjection: per-pixel noise with a learned scalar strength.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class EqualizedLinear(nn.Module):
    """Linear layer with equalized learning rate.

    lr_mul < 1 is used for the mapping network (StyleGAN2 uses 0.01).
    """

    def __init__(self, in_features, out_features, bias=True, bias_init=0.0, lr_mul=1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) / lr_mul)
        self.bias = nn.Parameter(torch.full((out_features,), float(bias_init))) if bias else None
        self.scale = lr_mul / math.sqrt(in_features)
        self.lr_mul = lr_mul

    def forward(self, x):
        bias = self.bias * self.lr_mul if self.bias is not None else None
        return F.linear(x, self.weight * self.scale, bias)


class EqualizedConv2d(nn.Module):
    """Plain convolution with equalized learning rate (no modulation)."""

    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_ch, in_ch, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_ch)) if bias else None
        self.scale = 1.0 / math.sqrt(in_ch * kernel_size * kernel_size)
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        return F.conv2d(x, self.weight * self.scale, self.bias,
                        stride=self.stride, padding=self.padding)


class BiasLeakyReLU(nn.Module):
    """Learned per-channel bias + LeakyReLU(0.2), rescaled by sqrt(2)."""

    def __init__(self, channels, negative_slope=0.2):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(channels))
        self.negative_slope = negative_slope
        self.gain = math.sqrt(2.0)

    def forward(self, x):
        x = x + self.bias.view(1, -1, 1, 1)
        return F.leaky_relu(x, self.negative_slope) * self.gain


class Blur(nn.Module):
    """Fixed binomial [1,3,3,1] blur applied per channel (depthwise).

    stride=1 keeps the resolution, stride=2 gives anti-aliased 2x downsampling.
    Assumes even H and W (true for every resolution in this project).
    """

    def __init__(self, channels, stride=1):
        super().__init__()
        k = torch.tensor([1.0, 3.0, 3.0, 1.0])
        k = k[:, None] * k[None, :]
        k = k / k.sum()
        self.register_buffer("kernel", k.expand(channels, 1, 4, 4).contiguous())
        self.channels = channels
        self.stride = stride

    def forward(self, x):
        # pad (1,2) for stride 1 and (1,1) for stride 2 keeps output size = input / stride
        pad = (1, 2, 1, 2) if self.stride == 1 else (1, 1, 1, 1)
        x = F.pad(x, pad, mode="reflect")
        return F.conv2d(x, self.kernel.to(x.dtype), stride=self.stride, groups=self.channels)


class ModulatedConv2d(nn.Module):
    """StyleGAN2 modulated convolution.

    Style w -> per-input-channel scales s. Weights are multiplied by s, then
    (optionally) demodulated so each output channel has unit expected norm.
    Implemented as one grouped conv over the batch (groups = batch size).

    upsample=True: the input is upsampled 2x with bilinear interpolation
    BEFORE the convolution. This is a deliberate deviation from the official
    transposed-conv + FIR filter: interpolation + conv avoids the
    checkerboard artifact seen in the Pix2Pix/SPADE experiments, and here
    capacity is controlled by channel counts, not by the kernel size.
    """

    def __init__(self, in_ch, out_ch, kernel_size, style_dim, demodulate=True, upsample=False):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.demodulate = demodulate
        self.upsample = upsample
        self.weight = nn.Parameter(torch.randn(1, out_ch, in_ch, kernel_size, kernel_size))
        self.scale = 1.0 / math.sqrt(in_ch * kernel_size * kernel_size)
        # bias_init=1 -> styles start at 1, i.e. no modulation at the start of training
        self.affine = EqualizedLinear(style_dim, in_ch, bias_init=1.0)

    def forward(self, x, w):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        b, c, h, wd = x.shape

        # fp32 for the weight math: demodulation can overflow/underflow in fp16 under AMP
        s = self.affine(w).float().view(b, 1, c, 1, 1)
        weight = self.scale * self.weight.float() * s  # (b, out, in, k, k)
        if self.demodulate:
            d = torch.rsqrt(weight.pow(2).sum(dim=(2, 3, 4)) + 1e-8)  # (b, out)
            weight = weight * d.view(b, self.out_ch, 1, 1, 1)

        weight = weight.view(b * self.out_ch, c, self.kernel_size, self.kernel_size)
        x = x.reshape(1, b * c, h, wd)
        x = F.conv2d(x, weight.to(x.dtype), padding=self.kernel_size // 2, groups=b)
        return x.view(b, self.out_ch, h, wd)


class NoiseInjection(nn.Module):
    """Adds per-pixel Gaussian noise scaled by a learned scalar (init 0)."""

    def __init__(self):
        super().__init__()
        self.strength = nn.Parameter(torch.zeros(1))

    def forward(self, x, noise=None):
        if noise is None:
            b, _, h, w = x.shape
            noise = torch.randn(b, 1, h, w, device=x.device, dtype=x.dtype)
        return x + self.strength.to(x.dtype) * noise


if __name__ == "__main__":
    # Quick shape sanity check: python -m src.gans.stylegan2_ada.layers
    x = torch.randn(2, 64, 8, 50)
    w = torch.randn(2, 512)

    mc = ModulatedConv2d(64, 32, 3, style_dim=512, upsample=True)
    y = mc(x, w)
    print("ModulatedConv2d up:", tuple(y.shape))  # expected (2, 32, 16, 100)

    mc_rgb = ModulatedConv2d(32, 3, 1, style_dim=512, demodulate=False)
    print("ToRGB-style 1x1:", tuple(mc_rgb(y, w).shape))  # expected (2, 3, 16, 100)

    print("Blur s=1:", tuple(Blur(64)(x).shape))  # expected (2, 64, 8, 50)
    print("Blur s=2:", tuple(Blur(64, stride=2)(x).shape))  # expected (2, 64, 4, 25)

    print("Noise:", tuple(NoiseInjection()(y).shape))
    print("BiasLeakyReLU:", tuple(BiasLeakyReLU(32)(y).shape))
    print("EqLinear:", tuple(EqualizedLinear(512, 64)(w).shape))
    print("EqConv:", tuple(EqualizedConv2d(64, 16, 3, padding=1)(x).shape))
"""
Multi-scale PatchGAN discriminator (Park et al., 2019 / Wang et al.,
pix2pixHD).

Two identical PatchGAN discriminators (num_D=2, the paper's default),
one operating on the (mask, image) pair at full resolution and one on a
2x-downsampled version (via average pooling, not strided conv, so the
downsampling itself introduces no learnable parameters or artifacts).
This gives the generator gradient signal at two receptive-field scales
simultaneously - useful here given how different class_2's typical
presentation (thin, 1-2px defect lines) is from class_3/4's large blob
defects; a single-scale PatchGAN might bias toward whichever scale
dominates the loss.

Each single-scale discriminator returns its list of intermediate layer
activations (not just the final patch logits), needed for the feature
matching loss in losses.py.

Conditioned on concatenated (mask, image), same 4+3=7 input channels as
the Pix2Pix PatchGAN discriminator - kept as a controlled variable
between the two GANs' discriminators, matching the project's default of
"reuse Pix2Pix conventions unless SPADE forces a difference".
"""

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm

INPUT_NC = 4 + 3  # mask channels + RGB image channels


def _init_conv(conv: nn.Conv2d) -> nn.Conv2d:
    """normal_(0, 0.02) init, applied before spectral_norm wrapping - see
    spade_resblock.py for why this must happen before, not after."""
    nn.init.normal_(conv.weight, 0.0, 0.02)
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0.0)
    return conv


class NLayerDiscriminator(nn.Module):
    """A single-scale PatchGAN, returning intermediate activations.

    Args:
        input_nc: input channel count (7: 4-channel mask + 3-channel image).
        ndf: base channel width (paper default 64).
        n_layers: number of downsampling conv layers (paper default 4,
            giving a 70x70-equivalent-ish patch receptive field - deeper
            than Pix2Pix's 3-layer 70x70 PatchGAN since SPADE's multi-scale
            setup relies on two shallower receptive fields adding up,
            rather than one discriminator alone covering the full range).
    """

    def __init__(self, input_nc: int = INPUT_NC, ndf: int = 64, n_layers: int = 4):
        super().__init__()
        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))  # 2, for kw=4

        self.layers = nn.ModuleList()

        self.layers.append(
            nn.Sequential(
                spectral_norm(_init_conv(nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw))),
                nn.LeakyReLU(0.2, inplace=False),
            )
        )

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            stride = 1 if n == n_layers - 1 else 2
            self.layers.append(
                nn.Sequential(
                    spectral_norm(_init_conv(nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=stride, padding=padw))),
                    nn.InstanceNorm2d(nf, affine=False),
                    nn.LeakyReLU(0.2, inplace=False),
                )
            )

        # final layer -> single-channel patch logits, no norm/activation,
        # no spectral norm (matches paper: only the feature-extracting
        # layers are spectral-normed, not the final logit projection).
        self.layers.append(_init_conv(nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)))

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Returns [input, layer_1_out, layer_2_out, ..., patch_logits] -
        every intermediate activation, needed for feature matching loss."""
        results = [x]
        for layer in self.layers:
            results.append(layer(results[-1]))
        return results[1:]


class MultiscaleDiscriminator(nn.Module):
    """num_D stacked NLayerDiscriminators, each seeing a coarser scale.

    Args:
        input_nc: input channel count (7).
        ndf: base channel width per discriminator.
        n_layers: depth per discriminator.
        num_D: number of scales (paper default 2).
    """

    def __init__(self, input_nc: int = INPUT_NC, ndf: int = 64, n_layers: int = 4, num_D: int = 2):
        super().__init__()
        self.num_D = num_D
        self.discriminators = nn.ModuleList(
            [NLayerDiscriminator(input_nc, ndf, n_layers) for _ in range(num_D)]
        )
        # count_include_pad=False matches the paper's downsampling exactly -
        # otherwise zero-padded border pixels get averaged in and bias the
        # coarse scale's border patches.
        self.downsample = nn.AvgPool2d(kernel_size=3, stride=2, padding=1, count_include_pad=False)

    def forward(self, x: torch.Tensor) -> list[list[torch.Tensor]]:
        """Returns a list of length num_D, each element being that scale's
        full list of intermediate activations (see NLayerDiscriminator)."""
        results = []
        current_input = x
        for i, D in enumerate(self.discriminators):
            results.append(D(current_input))
            if i != self.num_D - 1:
                current_input = self.downsample(current_input)
        return results


def build_discriminator(ndf: int = 64, n_layers: int = 4, num_D: int = 2) -> MultiscaleDiscriminator:
    return MultiscaleDiscriminator(input_nc=INPUT_NC, ndf=ndf, n_layers=n_layers, num_D=num_D)
"""
SPADE ResNet block (Park et al., 2019).

Each block applies two SPADE-normalized 3x3 convs (pre-activation style:
SPADE -> LeakyReLU -> Conv), with a SPADE-normalized 1x1 shortcut conv
whenever the channel count changes between fin and fout. Spectral norm is
applied to every conv, as in the original paper - it stabilizes the
generator against the discriminator's gradients without needing a
separate weight-clipping or gradient-penalty scheme.

Stochasticity decision (see chat discussion - deferred from
progress_log.md, decided when starting SPADE): the original SPADE paper
gets sample diversity from an optional VAE image encoder (z ~ q(z|real
image) at train time, z ~ N(0,1) at inference). We deliberately skip the
VAE branch and instead add Dropout2d to the three bottleneck-adjacent
blocks (head_0, G_middle_0, G_middle_1 in generator.py), mirroring the
Dropout(0.5)-near-the-bottleneck choice already made for the Pix2Pix
U-Net generator. This keeps the same "generator.train() at sampling time
for stochastic variants" pattern that compute_fid.py already relies on
for Pix2Pix's --samples-per-mask flag, so the two GANs stay comparable
under the same evaluation script without special-casing SPADE.

This is a documented deviation from the paper (no VAE encoder, no KL
loss term) - simpler, and consistent with the project's existing
Pix2Pix stochasticity mechanism, at the cost of not being the paper's
"canonical" stochastic SPADE variant.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm

from src.gans.spade.spade_norm import SPADE


class SPADEResnetBlock(nn.Module):
    """One SPADE ResBlock, optionally with dropout for stochastic sampling.

    Args:
        fin: input channel count.
        fout: output channel count. If fin != fout, a learned SPADE-
            normalized 1x1 shortcut conv is added (matches the paper).
        label_nc: mask channel count (4, see dataset.py).
        use_dropout: whether to apply Dropout2d(dropout_p) after the
            first conv, active only in this block. See module docstring
            for which blocks in generator.py enable this.
        dropout_p: dropout probability, only used if use_dropout=True.
    """

    def __init__(
        self,
        fin: int,
        fout: int,
        label_nc: int,
        use_dropout: bool = False,
        dropout_p: float = 0.5,
    ):
        super().__init__()
        self.learned_shortcut = fin != fout
        fmiddle = min(fin, fout)

        # Init BEFORE wrapping in spectral_norm: torch.nn.utils.parametrizations
        # .spectral_norm turns `.weight` into a computed parametrization, so a
        # later blanket `model.apply(init_weights)` (the pattern used for
        # Pix2Pix's plain Conv2d/ConvTranspose2d layers) would silently fail
        # to reach the real underlying weight here. Match the same
        # normal_(0, 0.02) convention as Pix2Pix, just applied at construction
        # time instead of via a post-hoc .apply() pass.
        conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        nn.init.normal_(conv_0.weight, 0.0, 0.02)
        nn.init.constant_(conv_0.bias, 0.0)
        nn.init.normal_(conv_1.weight, 0.0, 0.02)
        nn.init.constant_(conv_1.bias, 0.0)
        self.conv_0 = spectral_norm(conv_0)
        self.conv_1 = spectral_norm(conv_1)

        if self.learned_shortcut:
            conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)
            nn.init.normal_(conv_s.weight, 0.0, 0.02)
            self.conv_s = spectral_norm(conv_s)

        self.norm_0 = SPADE(fin, label_nc)
        self.norm_1 = SPADE(fmiddle, label_nc)
        if self.learned_shortcut:
            self.norm_s = SPADE(fin, label_nc)

        self.use_dropout = use_dropout
        if use_dropout:
            self.dropout = nn.Dropout2d(dropout_p)

    def _shortcut(self, x: torch.Tensor, segmap: torch.Tensor) -> torch.Tensor:
        if self.learned_shortcut:
            return self.conv_s(self.norm_s(x, segmap))
        return x

    @staticmethod
    def _actvn(x: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(x, 2e-1)

    def forward(self, x: torch.Tensor, segmap: torch.Tensor) -> torch.Tensor:
        x_s = self._shortcut(x, segmap)

        dx = self.conv_0(self._actvn(self.norm_0(x, segmap)))
        if self.use_dropout:
            dx = self.dropout(dx)
        dx = self.conv_1(self._actvn(self.norm_1(dx, segmap)))

        return x_s + dx
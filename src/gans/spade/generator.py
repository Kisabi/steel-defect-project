"""
SPADE generator (Park et al., 2019), adapted to native 1600x256 steel
defect images.

No image encoder, no VAE branch (see spade_resblock.py docstring for the
stochasticity decision). The generator starts from the conditioning mask
itself, downsampled to a small spatial seed, and repeatedly upsamples
while re-injecting the full-resolution mask at every SPADE ResBlock.

Resolution plan - chosen to exactly match Pix2Pix's bottleneck, not by
coincidence:

    Pix2Pix's U-Net used 6 downsampling blocks specifically because
    1600x256 isn't a power-of-two square; 6 blocks (factor 2^6 = 64)
    lands on a clean 4x25 bottleneck without fractional dimensions
    (1600/64=25, 256/64=4 exactly - see progress_log.md). SPADE's
    upsampling path is built here to hit that same 4x25 seed size and
    the same 64x total upsampling factor, using the paper's "more"
    upsampling-layer configuration (6 upsample() calls, 7 SPADEResBlocks:
    head_0, G_middle_0, G_middle_1, up_0..up_3), rather than the paper's
    default "normal" config (5 calls, factor 32x) or "most" (7 calls,
    128x, plus an extra block) - both of which would have needed uneven
    interpolation to hit 1600x256 exactly. This lets the FID/quality
    comparison against Pix2Pix cite the same effective receptive field
    depth for both generators as a controlled variable, not an
    incidental difference between architectures.

VRAM note: ngf=64 (paper default) gives 16*ngf=1024 channels at the
4x25/8x50/16x100 stages. Combined with spectral norm's power-iteration
overhead and the SPADE mask-conditioning branches running at every
resolution up to the full 256x1600, this is heavier per-sample than the
Pix2Pix U-Net (29.24M params) despite the shared bottleneck. Given the
~16GB VRAM budget that already constrained Pix2Pix to batch_size=2-4,
do a dry-run forward+backward pass at the target batch size before
committing to a full run - drop ngf to 32 (halves every channel count
above) if it doesn't fit, and report whichever value was actually used
as a stated deviation from the paper default, same as any other
VRAM-driven choice already documented for this thesis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.gans.spade.spade_resblock import SPADEResnetBlock

IMG_HEIGHT = 256
IMG_WIDTH = 1600
LABEL_NC = 4  # one-hot class_1..class_4 mask channels, see dataset.py
UPSAMPLE_FACTOR = 64  # 2^6, matches Pix2Pix's 6-downsampling-block bottleneck
SEED_H = IMG_HEIGHT // UPSAMPLE_FACTOR  # 4
SEED_W = IMG_WIDTH // UPSAMPLE_FACTOR  # 25


class SPADEGenerator(nn.Module):
    """SPADE generator, "more"-upsampling-layer configuration (6 doublings).

    Args:
        label_nc: mask channel count (4).
        ngf: base channel width. Paper default is 64; see module
            docstring for the VRAM trade-off if that doesn't fit.
        dropout_p: dropout probability for the three bottleneck-adjacent
            SPADEResnetBlocks (head_0, G_middle_0, G_middle_1) - this is
            the project's chosen stand-in for the paper's VAE-based
            stochasticity, see spade_resblock.py.
    """

    def __init__(self, label_nc: int = LABEL_NC, ngf: int = 64, dropout_p: float = 0.5):
        super().__init__()
        self.label_nc = label_nc
        self.seed_h = SEED_H
        self.seed_w = SEED_W

        nf = ngf

        # Project the downsampled mask directly to 16*nf channels at the
        # (seed_h, seed_w) seed resolution - no separate learned encoder,
        # no noise vector (see stochasticity decision above).
        self.fc = nn.Conv2d(label_nc, 16 * nf, kernel_size=3, padding=1)
        nn.init.normal_(self.fc.weight, 0.0, 0.02)
        nn.init.constant_(self.fc.bias, 0.0)

        # Bottleneck-adjacent blocks: dropout enabled here only, mirroring
        # the Pix2Pix U-Net's "dropout near the bottleneck, not throughout"
        # choice rather than regularizing every resolution stage.
        self.head_0 = SPADEResnetBlock(16 * nf, 16 * nf, label_nc, use_dropout=True, dropout_p=dropout_p)
        self.G_middle_0 = SPADEResnetBlock(16 * nf, 16 * nf, label_nc, use_dropout=True, dropout_p=dropout_p)
        self.G_middle_1 = SPADEResnetBlock(16 * nf, 16 * nf, label_nc, use_dropout=True, dropout_p=dropout_p)

        self.up_0 = SPADEResnetBlock(16 * nf, 8 * nf, label_nc)
        self.up_1 = SPADEResnetBlock(8 * nf, 4 * nf, label_nc)
        self.up_2 = SPADEResnetBlock(4 * nf, 2 * nf, label_nc)
        self.up_3 = SPADEResnetBlock(2 * nf, 1 * nf, label_nc)

        self.conv_img = nn.Conv2d(nf, 3, kernel_size=3, padding=1)
        nn.init.normal_(self.conv_img.weight, 0.0, 0.02)
        nn.init.constant_(self.conv_img.bias, 0.0)

        # nearest-neighbor upsampling, not ConvTranspose2d - the paper's
        # original choice, and one that sidesteps the checkerboard-artifact
        # rabbit hole already spent on Pix2Pix's decoder (see
        # progress_log.md "Checkerboard artifact fixes" session). Kept as
        # a single reusable module rather than re-instantiating per call.
        self.up = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, segmap: torch.Tensor) -> torch.Tensor:
        """
        Args:
            segmap: (B, label_nc, 256, 1600) one-hot mask, full resolution.
                Each SPADEResnetBlock internally nearest-resizes this down
                to its own working resolution (see spade_norm.py) - only
                the seed projection below needs an explicit resize.

        Returns:
            (B, 3, 256, 1600) generated image in [-1, 1] (tanh output),
            matching dataset.py's image normalization convention.
        """
        x = F.interpolate(segmap, size=(self.seed_h, self.seed_w), mode="nearest")
        x = self.fc(x)  # (B, 16*nf, 4, 25)

        x = self.head_0(x, segmap)  # (B, 16*nf, 4, 25)

        x = self.up(x)  # (B, 16*nf, 8, 50)
        x = self.G_middle_0(x, segmap)

        x = self.up(x)  # (B, 16*nf, 16, 100)
        x = self.G_middle_1(x, segmap)

        x = self.up(x)  # (B, 16*nf, 32, 200)
        x = self.up_0(x, segmap)  # (B, 8*nf, 32, 200)

        x = self.up(x)  # (B, 8*nf, 64, 400)
        x = self.up_1(x, segmap)  # (B, 4*nf, 64, 400)

        x = self.up(x)  # (B, 4*nf, 128, 800)
        x = self.up_2(x, segmap)  # (B, 2*nf, 128, 800)

        x = self.up(x)  # (B, 2*nf, 256, 1600)
        x = self.up_3(x, segmap)  # (B, 1*nf, 256, 1600)

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = torch.tanh(x)

        return x


def build_generator(ngf: int = 64, dropout_p: float = 0.5) -> SPADEGenerator:
    return SPADEGenerator(label_nc=LABEL_NC, ngf=ngf, dropout_p=dropout_p)
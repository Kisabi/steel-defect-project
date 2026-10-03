"""
SPADE normalization (Park et al., 2019 - "Semantic Image Synthesis with
Spatially-Adaptive Normalization").

Standard normalization layers (BatchNorm/InstanceNorm) collapse per-channel
statistics to two scalars (gamma, beta) applied uniformly across the whole
spatial map, which tends to wash out semantic information carried in a
segmentation mask when it's injected only at the input layer. SPADE instead
predicts *spatially-varying* gamma/beta maps directly from the mask, learned
via two small conv branches, and re-injects them at every normalization
layer throughout the generator.

Architectural decision (see progress_log.md, Pix2Pix session): base
normalization here is InstanceNorm2d rather than the paper's default
BatchNorm2d. This is not a paper deviation invented for SPADE - it's the
same fix already applied to the Pix2Pix generator, for the same underlying
reason: batch size is fixed at 2-4 by the ~16GB VRAM budget for native
1600x256 images, which makes BatchNorm's per-batch statistics too noisy to
be a stable base for the affine modulation. The original paper itself
names this configuration explicitly (`spadeinstance3x3` in their config
naming), so this is a supported variant, not an ad hoc change.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SPADE(nn.Module):
    """Spatially-adaptive de-normalization.

    Args:
        norm_nc: number of channels in the feature map being normalized
            (the SPADE ResBlock's input/intermediate channel count).
        label_nc: number of channels in the conditioning mask (4 for our
            one-hot class_1..class_4 mask, see dataset.py).
        nhidden: width of the shared hidden conv branch that produces the
            gamma/beta maps. 128 is the paper's default.
        kernel_size: kernel size for all three conv layers in this module.
            3 is the paper's default ("spade3x3").
    """

    def __init__(self, norm_nc: int, label_nc: int, nhidden: int = 128, kernel_size: int = 3):
        super().__init__()

        # Parameter-free base normalization - see module docstring for why
        # InstanceNorm2d instead of the paper's default BatchNorm2d.
        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)

        padding = kernel_size // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=kernel_size, padding=padding)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=kernel_size, padding=padding)

    def forward(self, x: torch.Tensor, segmap: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, norm_nc, H, W) feature map to normalize.
            segmap: (B, label_nc, H_mask, W_mask) conditioning mask, at
                whatever resolution it was originally provided (nearest-
                resized here to match x's spatial size - the mask is
                categorical, so nearest is the only resize mode that
                doesn't invent fractional class memberships).
        """
        normalized = self.param_free_norm(x)

        segmap = F.interpolate(segmap, size=x.shape[2:], mode="nearest")
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        return normalized * (1 + gamma) + beta
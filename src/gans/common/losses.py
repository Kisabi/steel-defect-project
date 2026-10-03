"""
Loss functions shared across GAN architectures - currently used by SPADE,
written generically enough to reuse for StyleGAN2-ADA later if needed.

SPADE's training recipe (Park et al., 2019) differs from Pix2Pix's
BCE+L1 in three ways, all kept faithful to the paper here:

  1. Hinge loss instead of BCEWithLogits, for both G and D.
  2. Feature matching loss: L1 between real/fake intermediate
     discriminator activations (from MultiscaleDiscriminator's returned
     activation lists), encouraging the generator to match statistics at
     multiple depths rather than only fooling the final patch logits.
  3. VGG perceptual loss: L1 between real/fake VGG19 features at 5
     depths, weighted to favor deeper (more semantic) layers less than
     shallow (more textural) ones - the paper's [1/32, 1/16, 1/8, 1/4, 1]
     schedule.

VGG19 weights: torchvision's pretrained() download requires
download.pytorch.org, which isn't reachable from this project's network
setup. The official ImageNet checkpoint (vgg19-dcbb9e9d.pth) was
downloaded manually and placed at
/opt/steel-defect-data/pretrained/vgg19-dcbb9e9d.pth - this is the exact
same checkpoint torchvision.models.vgg19(pretrained=True) would have
fetched automatically, just loaded from a local path instead. No
architectural or weight difference from the "canonical" perceptual loss
setup - purely a network-access workaround, worth a one-line footnote in
the thesis methodology section for reproducibility (anyone re-running
this needs the same file at the same path, or to edit VGG_WEIGHTS_PATH
below).
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

VGG_WEIGHTS_PATH = Path("/opt/steel-defect-data/pretrained/vgg19-dcbb9e9d.pth")

# torchvision.models.vgg19().features layer indices immediately after each
# reluN_1 activation - the paper's chosen tap points for perceptual loss.
_VGG_SLICE_ENDPOINTS = [2, 7, 12, 21, 30]  # relu1_1, relu2_1, relu3_1, relu4_1, relu5_1
_VGG_LAYER_WEIGHTS = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class HingeGANLoss:
    """Hinge loss for the discriminator and generator, SPADE-style.

    D loss:  E[relu(1 - D(real))] + E[relu(1 + D(fake))]
    G loss: -E[D(fake)]

    Operates on raw patch logits (a single tensor). When the
    discriminator is multi-scale, call this once per scale and average -
    see spade_train_step-style usage in train.py.
    """

    @staticmethod
    def d_loss(pred_real: torch.Tensor, pred_fake: torch.Tensor) -> torch.Tensor:
        loss_real = F.relu(1.0 - pred_real).mean()
        loss_fake = F.relu(1.0 + pred_fake).mean()
        return loss_real + loss_fake

    @staticmethod
    def g_loss(pred_fake: torch.Tensor) -> torch.Tensor:
        return -pred_fake.mean()


def feature_matching_loss(
    real_features: list[list[torch.Tensor]],
    fake_features: list[list[torch.Tensor]],
    num_D: int,
    n_layers_D: int,
) -> torch.Tensor:
    """L1 feature matching loss across every discriminator scale and layer.

    Args:
        real_features / fake_features: MultiscaleDiscriminator's output on
            (mask, real_image) and (mask, fake_image) respectively - a
            list of num_D scales, each a list of that scale's intermediate
            activations (including the final patch logits).
        num_D: number of discriminator scales.
        n_layers_D: number of conv layers per single-scale discriminator
            (needed to reproduce the paper's per-layer weight schedule).

    Note: fake_features must come from a discriminator forward pass with
    gradients enabled (the generator needs gradients through this term);
    real_features should be computed with the real discriminator forward
    but real images are typically detached upstream since only the fake
    side needs a gradient path back into the generator.
    """
    # paper's weighting: deeper/final layers matter less per-layer since
    # there are more of them contributing; normalized by scale count too.
    feat_weight = 4.0 / (n_layers_D + 1)
    d_weight = 1.0 / num_D

    loss = torch.tensor(0.0, device=fake_features[0][0].device)
    for scale_real, scale_fake in zip(real_features, fake_features):
        # exclude the final element (patch logits) - feature matching only
        # applies to intermediate feature layers, not the discriminator's
        # final decision.
        for layer_real, layer_fake in zip(scale_real[:-1], scale_fake[:-1]):
            loss = loss + feat_weight * d_weight * F.l1_loss(layer_fake, layer_real.detach())
    return loss


class VGGLoss(nn.Module):
    """Perceptual loss: L1 between VGG19 features at 5 depths.

    Expects inputs in [-1, 1] (this project's image normalization
    convention, see dataset.py) and internally converts to the [0, 1] +
    ImageNet-mean/std range VGG19 was trained on.
    """

    def __init__(self, weights_path: Path = VGG_WEIGHTS_PATH):
        super().__init__()
        if not weights_path.exists():
            raise FileNotFoundError(
                f"VGG19 weights not found at {weights_path}. Download the official "
                "torchvision checkpoint (vgg19-dcbb9e9d.pth) manually - "
                "download.pytorch.org isn't reachable from this project's network "
                "setup - and place it at this path, or pass a different "
                "weights_path. See this module's docstring."
            )

        vgg = torchvision.models.vgg19(weights=None)
        state_dict = torch.load(weights_path, map_location="cpu")
        vgg.load_state_dict(state_dict)
        vgg_features = vgg.features

        self.slices = nn.ModuleList()
        start = 0
        for end in _VGG_SLICE_ENDPOINTS:
            self.slices.append(vgg_features[start:end])
            start = end

        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        self.register_buffer("mean", _IMAGENET_MEAN)
        self.register_buffer("std", _IMAGENET_STD)
        self.weights = _VGG_LAYER_WEIGHTS

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """[-1, 1] -> ImageNet-normalized [0, 1]-range input VGG19 expects."""
        x = (x.clamp(-1, 1) + 1) / 2
        return (x - self.mean) / self.std

    def forward(self, fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        fake = self._preprocess(fake)
        real = self._preprocess(real.detach())

        loss = torch.tensor(0.0, device=fake.device)
        x_fake, x_real = fake, real
        for weight, slice_module in zip(self.weights, self.slices):
            x_fake = slice_module(x_fake)
            with torch.no_grad():
                x_real = slice_module(x_real)
            loss = loss + weight * F.l1_loss(x_fake, x_real)
        return loss
"""
Pix2Pix discriminator: 70x70 PatchGAN, conditioned on (mask, image).

The discriminator sees the mask and image concatenated channel-wise
(4 + 3 = 7 input channels) - it's a *conditional* GAN discriminator: it
judges whether the image is a real/plausible photo *for that specific
mask*, not just whether it's a plausible steel image in general.

Fully convolutional, so it works on any input spatial size/aspect ratio
without modification - no need to match the generator's depth. Output is
a grid of realism scores, one per patch of the input, rather than a
single real/fake scalar (hence "PatchGAN": each element only "sees" a
local ~70x70 receptive field of the input, which pushes the discriminator
to judge local texture realism - well suited to defect textures - rather
than global image layout).
"""

import torch
import torch.nn as nn

IMG_HEIGHT = 256
IMG_WIDTH = 1600
MASK_CHANNELS = 4
IMAGE_CHANNELS = 3


class NLayerDiscriminator(nn.Module):
    """PatchGAN discriminator with `n_layers` downsampling convolutions."""

    def __init__(self, input_nc: int, ndf: int = 64, n_layers: int = 3, norm_layer=nn.InstanceNorm2d):
        super().__init__()
        kw, padw = 4, 1

        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=False),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=False),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        # final layer: 1-channel realism map, no activation (raw logits - paired with BCEWithLogitsLoss)
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        self.model = nn.Sequential(*sequence)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def build_discriminator(
    mask_channels: int = MASK_CHANNELS,
    image_channels: int = IMAGE_CHANNELS,
    ndf: int = 64,
    n_layers: int = 3,
) -> nn.Module:
    """Build the conditional PatchGAN discriminator.

    Input channels = mask_channels + image_channels, since mask and image
    are concatenated before being fed in (see module docstring).
    """
    return NLayerDiscriminator(input_nc=mask_channels + image_channels, ndf=ndf, n_layers=n_layers)


if __name__ == "__main__":
    from src.gans.pix2pix.generator import init_weights  # reuse the same normal(0, 0.02) init

    discriminator = build_discriminator()
    discriminator.apply(init_weights)

    dummy_mask = torch.randn(2, MASK_CHANNELS, IMG_HEIGHT, IMG_WIDTH)
    dummy_image = torch.randn(2, IMAGE_CHANNELS, IMG_HEIGHT, IMG_WIDTH)
    dummy_input = torch.cat([dummy_mask, dummy_image], dim=1)

    output = discriminator(dummy_input)

    n_params = sum(p.numel() for p in discriminator.parameters())
    print(f"Discriminator params: {n_params / 1e6:.2f}M")
    print(f"Input  shape: {tuple(dummy_input.shape)}")
    print(f"Output shape (patch realism map): {tuple(output.shape)}")
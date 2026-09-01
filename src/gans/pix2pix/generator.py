"""
Pix2Pix generator: U-Net with skip connections, mask -> image.

Built recursively (innermost block first, then wrapped outward), matching
the canonical Pix2Pix/CycleGAN reference implementation - this keeps the
skip-connection wiring correct by construction instead of hand-managing
encoder/decoder feature lists.

num_downs=6 (not the original paper's 8): our images are 1600x256, not
square. 6 downsampling steps divide both dimensions by 2**6=64 exactly
(1600/64=25, 256/64=4), landing on a 4x25 bottleneck with no fractional
sizes and no dimension collapsing to 1px. 8 downs would collapse the
256px height dimension too early and lose vertical spatial structure.

InstanceNorm2d is used instead of BatchNorm2d: expected batch size is
small (2-4, given 16GB VRAM and full 1600x256 resolution), where
BatchNorm's running statistics get noisy. Dropout(0.5) in the blocks
nearest the bottleneck is the model's only source of stochasticity
(standard Pix2Pix convention - no explicit noise vector z; dropout stays
active at inference time too, for sample diversity).
"""

import torch
import torch.nn as nn

IMG_HEIGHT = 256
IMG_WIDTH = 1600
MASK_CHANNELS = 4
IMAGE_CHANNELS = 3


class UnetSkipConnectionBlock(nn.Module):
    """One encoder-decoder shell of the U-Net, wrapping a `submodule` (or none, if innermost)."""

    def __init__(
        self,
        outer_nc: int,
        inner_nc: int,
        input_nc: int | None = None,
        submodule: nn.Module | None = None,
        outermost: bool = False,
        innermost: bool = False,
        norm_layer=nn.InstanceNorm2d,
        use_dropout: bool = False,
    ):
        super().__init__()
        self.outermost = outermost
        if input_nc is None:
            input_nc = outer_nc

        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=False)
        downrelu = nn.LeakyReLU(0.2, inplace=True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(inplace=True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=False)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=False)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            model = down + [submodule] + up
            if use_dropout:
                model = model + [nn.Dropout(0.5)]

        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.outermost:
            return self.model(x)
        return torch.cat([x, self.model(x)], dim=1)  # skip connection


def build_generator(
    input_nc: int = MASK_CHANNELS,
    output_nc: int = IMAGE_CHANNELS,
    num_downs: int = 6,
    ngf: int = 64,
    use_dropout: bool = True,
) -> nn.Module:
    """Build the full U-Net generator by nesting UnetSkipConnectionBlocks inside out.

    Args:
        input_nc: input channels (4 - one-hot mask, classes 1-4).
        output_nc: output channels (3 - RGB image).
        num_downs: number of downsampling steps. 6 for native 1600x256
            input (see module docstring).
        ngf: base number of generator filters; doubles at each
            downsampling step, capped at ngf*8.
        use_dropout: apply Dropout(0.5) in the blocks nearest the
            bottleneck for output stochasticity.
    """
    # innermost block (bottleneck)
    unet_block = UnetSkipConnectionBlock(
        ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=nn.InstanceNorm2d, innermost=True
    )
    # blocks at the bottleneck resolution, with dropout (num_downs - 5 of them)
    for _ in range(num_downs - 5):
        unet_block = UnetSkipConnectionBlock(
            ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
            norm_layer=nn.InstanceNorm2d, use_dropout=use_dropout,
        )
    # progressively narrower blocks going back out to full resolution
    unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=nn.InstanceNorm2d)
    unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=nn.InstanceNorm2d)
    unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=nn.InstanceNorm2d)
    # outermost block: takes the raw mask, outputs the RGB image
    model = UnetSkipConnectionBlock(
        output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=nn.InstanceNorm2d
    )
    return model


def init_weights(module: nn.Module, mean: float = 0.0, std: float = 0.02):
    """Standard Pix2Pix/DCGAN weight init: normal(0, 0.02) for conv/norm layers."""
    classname = module.__class__.__name__
    if hasattr(module, "weight") and ("Conv" in classname or "Linear" in classname):
        nn.init.normal_(module.weight.data, mean, std)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.constant_(module.bias.data, 0.0)
    elif "InstanceNorm2d" in classname or "BatchNorm2d" in classname:
        if module.weight is not None:
            nn.init.normal_(module.weight.data, 1.0, std)
        if module.bias is not None:
            nn.init.constant_(module.bias.data, 0.0)


if __name__ == "__main__":
    # smoke test: verify shapes on the actual 1600x256 resolution before wiring up training
    generator = build_generator()
    generator.apply(init_weights)

    dummy_mask = torch.randn(2, MASK_CHANNELS, IMG_HEIGHT, IMG_WIDTH)
    output = generator(dummy_mask)

    n_params = sum(p.numel() for p in generator.parameters())
    print(f"Generator params: {n_params / 1e6:.2f}M")
    print(f"Input  shape: {tuple(dummy_mask.shape)}")
    print(f"Output shape: {tuple(output.shape)}")
    assert output.shape == (2, IMAGE_CHANNELS, IMG_HEIGHT, IMG_WIDTH), "Output shape mismatch!"
    print("OK - output shape matches expected (2, 3, 256, 1600)")
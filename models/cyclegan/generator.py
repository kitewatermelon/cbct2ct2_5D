"""ResNet Generator for CycleGAN (2D, single-channel)."""
from __future__ import annotations
import torch
import torch.nn as nn
from monai.networks.blocks import ResidualUnit


class ResNetGenerator(nn.Module):
    """CBCT↔CT 변환용 ResNet Generator.

    구조: Conv↓(×2) → ResBlock(×n_blocks) → ConvTranspose↑(×2) → tanh
    입력/출력: (B, in_channels, H, W), 값 범위 [-1, 1]
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        ngf: int = 64,
        n_blocks: int = 9,
    ) -> None:
        super().__init__()

        # Encoder
        enc: list[nn.Module] = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, ngf, 7, bias=False),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True),
            nn.Conv2d(ngf, ngf * 2, 3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.Conv2d(ngf * 2, ngf * 4, 3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(ngf * 4),
            nn.ReLU(True),
        ]

        # ResBlocks (MONAI ResidualUnit, in==out이므로 shortcut = identity)
        res: list[nn.Module] = [
            ResidualUnit(
                spatial_dims=2,
                in_channels=ngf * 4,
                out_channels=ngf * 4,
                kernel_size=3,
                act="RELU",
                norm="INSTANCE",
                bias=False,
            )
            for _ in range(n_blocks)
        ]

        # Decoder
        dec: list[nn.Module] = [
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, stride=2, padding=1,
                               output_padding=1, bias=False),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 3, stride=2, padding=1,
                               output_padding=1, bias=False),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, out_channels, 7),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*enc, *res, *dec)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

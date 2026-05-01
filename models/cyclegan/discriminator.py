"""PatchDiscriminator wrapper for CycleGAN."""
from __future__ import annotations
from monai.networks.nets import PatchDiscriminator


def build_discriminator(in_channels: int = 1) -> PatchDiscriminator:
    """70×70 PatchGAN discriminator (MONAI 구현).

    반환값: PatchDiscriminator 인스턴스.
    forward() 반환: list[Tensor] — [-1] 요소가 패치 맵 (B, 1, H', W')
    """
    return PatchDiscriminator(
        in_channels=in_channels,
        num_layers_d=3,
        spatial_dims=2,
        channels=64,
        out_channels=1,
    )

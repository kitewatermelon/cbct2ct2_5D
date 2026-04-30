"""InceptionV3 helpers for FID computation in eval scripts.

Images are expected to be in [0, 1] normalized range.
(Training-time FID in stage2_vdm.py uses a separate copy that handles [-1, 1].)
"""
from __future__ import annotations
import torch
from torchvision.models import inception_v3


def build_inception(device: torch.device):
    m = inception_v3(pretrained=True, transform_input=False).to(device)
    m.fc = torch.nn.Identity()
    m.eval()
    return m


@torch.no_grad()
def inception_feats(x: torch.Tensor, model) -> torch.Tensor:
    """x: (B, 1, H, W) in [0, 1] → InceptionV3 feature vectors."""
    x = x.clamp(0.0, 1.0).repeat(1, 3, 1, 1)
    return model(x).flatten(1)

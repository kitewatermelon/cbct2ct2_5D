import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import torch
from models.cyclegan.generator import ResNetGenerator

def test_generator_output_shape():
    G = ResNetGenerator(in_channels=1, out_channels=1, ngf=64, n_blocks=9)
    x = torch.randn(2, 1, 128, 128)
    out = G(x)
    assert out.shape == (2, 1, 128, 128), f"expected (2,1,128,128), got {out.shape}"

def test_generator_output_range():
    G = ResNetGenerator(in_channels=1, out_channels=1, ngf=64, n_blocks=9)
    x = torch.randn(2, 1, 128, 128)
    out = G(x)
    assert out.min() >= -1.0 - 1e-5 and out.max() <= 1.0 + 1e-5, \
        f"tanh output out of range: [{out.min():.4f}, {out.max():.4f}]"

from models.cyclegan.discriminator import build_discriminator

def test_discriminator_output_shape():
    D = build_discriminator(in_channels=1)
    x = torch.randn(2, 1, 128, 128)
    out = D(x)
    # PatchDiscriminator는 리스트 반환 — 마지막 요소가 patch map
    patch = out[-1]
    assert patch.shape[0] == 2
    assert patch.shape[1] == 1
    assert patch.ndim == 4, f"expected 4D patch output, got {patch.shape}"

def test_discriminator_patch_is_smaller_than_input():
    D = build_discriminator(in_channels=1)
    x = torch.randn(2, 1, 128, 128)
    patch = D(x)[-1]
    assert patch.shape[2] < 128 and patch.shape[3] < 128

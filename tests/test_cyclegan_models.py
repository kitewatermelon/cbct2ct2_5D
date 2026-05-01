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

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


def test_one_training_step():
    """학습 1 step이 에러 없이 실행되고 loss가 scalar인지 확인."""
    import torch.nn.functional as F
    from models.cyclegan.generator import ResNetGenerator
    from models.cyclegan.discriminator import build_discriminator

    device = torch.device("cpu")
    G_A2B = ResNetGenerator().to(device)
    G_B2A = ResNetGenerator().to(device)
    D_A   = build_discriminator().to(device)
    D_B   = build_discriminator().to(device)

    opt_G = torch.optim.Adam(
        list(G_A2B.parameters()) + list(G_B2A.parameters()),
        lr=2e-4, betas=(0.5, 0.999)
    )
    opt_D = torch.optim.Adam(
        list(D_A.parameters()) + list(D_B.parameters()),
        lr=2e-4, betas=(0.5, 0.999)
    )

    real_A = torch.randn(2, 1, 128, 128, device=device)  # CBCT
    real_B = torch.randn(2, 1, 128, 128, device=device)  # CT

    # Generator step
    opt_G.zero_grad()
    fake_B = G_A2B(real_A)
    fake_A = G_B2A(real_B)
    loss_adv = (
        F.mse_loss(D_B(fake_B)[-1], torch.ones_like(D_B(fake_B)[-1])) +
        F.mse_loss(D_A(fake_A)[-1], torch.ones_like(D_A(fake_A)[-1]))
    )
    rec_A = G_B2A(fake_B)
    rec_B = G_A2B(fake_A)
    loss_cycle = F.l1_loss(rec_A, real_A) + F.l1_loss(rec_B, real_B)
    loss_G = loss_adv + 10.0 * loss_cycle
    loss_G.backward()
    opt_G.step()

    # Discriminator step
    opt_D.zero_grad()
    loss_D = (
        0.5 * (F.mse_loss(D_B(real_B.detach())[-1], torch.ones(2,1,14,14)) +
               F.mse_loss(D_B(fake_B.detach())[-1], torch.zeros(2,1,14,14))) +
        0.5 * (F.mse_loss(D_A(real_A.detach())[-1], torch.ones(2,1,14,14)) +
               F.mse_loss(D_A(fake_A.detach())[-1], torch.zeros(2,1,14,14)))
    )
    loss_D.backward()
    opt_D.step()

    assert loss_G.item() > 0
    assert loss_D.item() > 0

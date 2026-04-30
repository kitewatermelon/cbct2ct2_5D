"""eval_full.py — Stage2 VDM 전체 ablation 평가.

8개 모델을 stage2_vdm.py와 동일한 val split에서 평가.
출력: eval_results/raw_metrics.csv, summary_stats.csv, boxplot_*.png
"""
from __future__ import annotations
import argparse, math, pathlib
import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from monai.utils import set_determinism
from tqdm.auto import tqdm

from data.synthrad2025 import SynthRad2025, build_transforms

# Try to import from stage2_vdm, but allow graceful failure for testing
try:
    from stage2_vdm import (
        build_vqvae, load_frozen_vqvae, _prepare_cond,
        sample_conditional, build_vdm,
    )
except ImportError:
    pass  # will be added in Task 3

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_CONFIGS: list[dict] = [
    {"key": "uvit_n1_cpr4", "backbone": "uvit", "n": 1, "cpr": 4},
    {"key": "uvit_n3_cpr4", "backbone": "uvit", "n": 3, "cpr": 4},
    {"key": "uvit_n5_cpr4", "backbone": "uvit", "n": 5, "cpr": 4},
    {"key": "uvit_n7_cpr4", "backbone": "uvit", "n": 7, "cpr": 4},
    {"key": "uvit_n9_cpr4", "backbone": "uvit", "n": 9, "cpr": 4},
    {"key": "uvit_n5_cpr2", "backbone": "uvit", "n": 5, "cpr": 2},
    {"key": "uvit_n5_cpr8", "backbone": "uvit", "n": 5, "cpr": 8},
    {"key": "unet_n5_cpr4", "backbone": "unet", "n": 5, "cpr": 4},
]

# ---------------------------------------------------------------------------
# Metric utilities
# ---------------------------------------------------------------------------

def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute PSNR (Peak Signal-to-Noise Ratio).

    Args:
        pred: Predicted tensor
        target: Target tensor

    Returns:
        PSNR value in dB. Returns 100.0 for perfect predictions (MSE < 1e-12).
    """
    mse = F.mse_loss(pred.float(), target.float()).item()
    if mse < 1e-12:
        return 100.0
    return 20.0 * math.log10(1.0 / math.sqrt(mse))


def compute_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute SSIM (Structural Similarity Index Measure).

    Uses standard SSIM formula with 11x11 Gaussian kernel.

    Args:
        pred: Predicted tensor
        target: Target tensor

    Returns:
        SSIM value in range [-1, 1]
    """
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    pred, target = pred.float(), target.float()
    mu_p = F.avg_pool2d(pred,   11, 1, 5)
    mu_t = F.avg_pool2d(target, 11, 1, 5)
    mu_p2, mu_t2, mu_pt = mu_p * mu_p, mu_t * mu_t, mu_p * mu_t
    sp = F.avg_pool2d(pred * pred,     11, 1, 5) - mu_p2
    st = F.avg_pool2d(target * target, 11, 1, 5) - mu_t2
    spt = F.avg_pool2d(pred * target,  11, 1, 5) - mu_pt
    ssim_map = ((2*mu_pt+C1)*(2*spt+C2)) / ((mu_p2+mu_t2+C1)*(sp+st+C2))
    return ssim_map.mean().item()


def compute_mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute Mean Squared Error.

    Args:
        pred: Predicted tensor
        target: Target tensor

    Returns:
        MSE value
    """
    return F.mse_loss(pred.float(), target.float()).item()


# ---------------------------------------------------------------------------
# Data split (stage2_vdm.py와 동일한 로직)
# ---------------------------------------------------------------------------

def build_val_split(
    data_root: str, n: int, val_ratio: float = 0.2, seed: int = 42,
) -> list[int]:
    """val_ds의 원본 full_ds 내 인덱스 목록 반환."""
    full_ds = SynthRad2025(
        root=f"{data_root}/dataset/train/n{n}",
        modality=["cbct", "ct"],
        anatomy=["AB", "HN", "TH"],
        transform=build_transforms(["cbct", "ct"], spatial_size=(128, 128), augment=False),
    )
    n_val   = int(len(full_ds) * val_ratio)
    n_train = len(full_ds) - n_val
    _, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )
    return list(val_ds.indices)


def build_subj_anatomy_map(data_root: str, n: int) -> dict[str, str]:
    """subj_id → anatomy 매핑 딕셔너리 반환."""
    full_ds = SynthRad2025(
        root=f"{data_root}/dataset/train/n{n}",
        modality=["cbct", "ct"],
        anatomy=["AB", "HN", "TH"],
        transform=None,
    )
    return {d.name: d.parent.name for d in full_ds.subject_dirs}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

EMBEDDING_DIM   = 1
NUM_EMBEDDINGS  = 2048
SPATIAL_SIZE    = 128

def load_model_for_eval(
    cfg: dict,
    data_root: str,
    ckpt_base: str,
    vqvae_base: str,
    device: torch.device,
    val_ratio: float = 0.2,
    seed: int = 42,
    batch_size: int = 8,
    num_workers: int = 4,
) -> dict:
    from stage2_vdm import build_vqvae, load_frozen_vqvae, build_vdm

    n, cpr, backbone = cfg["n"], cfg["cpr"], cfg["backbone"]
    key = cfg["key"]

    # ── VQ-VAE ───────────────────────────────────────────────────────────
    vqvae_kw = dict(out_channels=1, compress_ratio=cpr,
                    embedding_dim=EMBEDDING_DIM, num_embeddings=NUM_EMBEDDINGS)
    ct_ae   = load_frozen_vqvae(
        f"{vqvae_base}/1_vqvae_ct_n{n}_cpr{cpr}_img128/best.pt",
        device, in_channels=n, **vqvae_kw,
    )
    cbct_ae = load_frozen_vqvae(
        f"{vqvae_base}/1_vqvae_cbct_n{n}_cpr{cpr}_img128/best.pt",
        device, in_channels=n, **vqvae_kw,
    )

    # ── latent shape ─────────────────────────────────────────────────────
    with torch.no_grad():
        dummy = torch.zeros(1, n, SPATIAL_SIZE, SPATIAL_SIZE, device=device)
        latent_shape = tuple(ct_ae.encode_stage_2_inputs(dummy).shape[1:])

    # ── Val split + anatomy map ───────────────────────────────────────────
    subj_anatomy = build_subj_anatomy_map(data_root, n)
    full_ds = SynthRad2025(
        root=f"{data_root}/dataset/train/n{n}",
        modality=["cbct", "ct"], anatomy=["AB", "HN", "TH"],
        transform=build_transforms(["cbct", "ct"], spatial_size=(128, 128), augment=False),
    )
    n_val   = int(len(full_ds) * val_ratio)
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )

    # ── scale_factor (훈련과 동일: shuffle=True, seed=42 첫 배치) ─────────
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
        generator=torch.Generator().manual_seed(seed),
    )
    with torch.no_grad():
        first_batch = next(iter(train_loader))
        _z = ct_ae.encode_stage_2_inputs(first_batch["ct"].to(device))
        scale_factor = 1.0 / _z.flatten().std().item()

    # ── VDM ──────────────────────────────────────────────────────────────
    vdm = build_vdm(backbone, n, cpr, latent_shape, SPATIAL_SIZE).to(device)
    ckpt = torch.load(
        f"{ckpt_base}/1_vdm_{backbone}_n{n}_cpr{cpr}_img128/latest.pt",
        map_location=device, weights_only=False,
    )
    vdm.load_state_dict(ckpt["model_state_dict"], strict=False)
    vdm.eval()
    for p in vdm.parameters():
        p.requires_grad_(False)

    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )

    return dict(
        ct_ae=ct_ae, cbct_ae=cbct_ae, vdm=vdm,
        scale_factor=scale_factor, val_ds=val_ds,
        val_loader=val_loader, subj_anatomy=subj_anatomy,
        latent_shape=latent_shape,
    )

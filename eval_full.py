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

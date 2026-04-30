import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import math
import torch
from eval_full import MODEL_CONFIGS, compute_psnr, compute_ssim, compute_mse

def test_model_configs_count():
    assert len(MODEL_CONFIGS) == 8

def test_model_configs_keys():
    keys = {c["key"] for c in MODEL_CONFIGS}
    assert keys == {
        "uvit_n1_cpr4", "uvit_n3_cpr4", "uvit_n5_cpr4",
        "uvit_n7_cpr4", "uvit_n9_cpr4",
        "uvit_n5_cpr2", "uvit_n5_cpr8",
        "unet_n5_cpr4",
    }

def test_compute_psnr_perfect():
    x = torch.ones(1, 1, 4, 4) * 0.5
    assert compute_psnr(x, x) == 100.0

def test_compute_psnr_known():
    pred   = torch.zeros(1, 1, 4, 4)
    target = torch.ones(1, 1, 4, 4)
    # mse=1.0 → psnr = 20*log10(1/1) = 0.0
    assert abs(compute_psnr(pred, target) - 0.0) < 1e-4

def test_compute_ssim_range():
    x = torch.rand(1, 1, 32, 32)
    y = torch.rand(1, 1, 32, 32)
    s = compute_ssim(x, y)
    assert -1.0 <= s <= 1.0

def test_compute_mse_perfect():
    x = torch.ones(1, 1, 4, 4)
    assert compute_mse(x, x) == 0.0

def test_compute_mse_known():
    pred   = torch.zeros(1, 1, 2, 2)
    target = torch.ones(1, 1, 2, 2) * 2.0
    assert abs(compute_mse(pred, target) - 4.0) < 1e-6


from eval_full import build_val_split

def test_val_split_reproducible():
    """같은 seed로 두 번 split하면 동일한 indices가 나와야 한다."""
    import os
    data_root = os.environ.get("DATA_ROOT", "/home/dministrator/s2025")
    root = f"{data_root}/dataset/train/n5"
    if not pathlib.Path(root).exists():
        import pytest; pytest.skip("data not available")

    indices_a = build_val_split(data_root=data_root, n=5, val_ratio=0.2, seed=42)
    indices_b = build_val_split(data_root=data_root, n=5, val_ratio=0.2, seed=42)
    assert indices_a == indices_b

def test_val_split_ratio():
    import os
    data_root = os.environ.get("DATA_ROOT", "/home/dministrator/s2025")
    root = f"{data_root}/dataset/train/n5"
    if not pathlib.Path(root).exists():
        import pytest; pytest.skip("data not available")

    indices = build_val_split(data_root=data_root, n=5, val_ratio=0.2, seed=42)
    from data.synthrad2025 import SynthRad2025
    full = SynthRad2025(root=root, modality=["cbct", "ct"], anatomy=["AB","HN","TH"],
                        transform=None)
    expected_n_val = int(len(full) * 0.2)
    assert len(indices) == expected_n_val

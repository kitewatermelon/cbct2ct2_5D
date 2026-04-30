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

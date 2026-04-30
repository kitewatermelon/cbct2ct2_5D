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


from eval_full import load_model_for_eval

def test_load_model_for_eval_keys():
    import os
    data_root = os.environ.get("DATA_ROOT", "/home/dministrator/s2025")
    ckpt_base = "checkpoints_임베딩1/stage2_vdm"
    vqvae_base = "checkpoints_임베딩1/stage1_vqvae"
    if not pathlib.Path(ckpt_base).exists():
        import pytest; pytest.skip("checkpoints not available")

    result = load_model_for_eval(
        cfg={"key":"uvit_n5_cpr4","backbone":"uvit","n":5,"cpr":4},
        data_root=data_root,
        ckpt_base=ckpt_base,
        vqvae_base=vqvae_base,
        device=torch.device("cpu"),
    )
    assert "ct_ae" in result
    assert "cbct_ae" in result
    assert "vdm" in result
    assert "scale_factor" in result
    assert "val_ds" in result
    assert "subj_anatomy" in result
    assert isinstance(result["scale_factor"], float)


from eval_full import evaluate_model

def test_evaluate_model_output_schema():
    """evaluate_model이 올바른 컬럼을 가진 DataFrame을 반환해야 한다."""
    import os
    data_root = os.environ.get("DATA_ROOT", "/home/dministrator/s2025")
    ckpt_base  = "checkpoints_임베딩1/stage2_vdm"
    vqvae_base = "checkpoints_임베딩1/stage1_vqvae"
    if not pathlib.Path(ckpt_base).exists():
        import pytest; pytest.skip("checkpoints not available")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg = {"key": "uvit_n5_cpr4", "backbone": "uvit", "n": 5, "cpr": 4}
    loaded = load_model_for_eval(cfg, data_root, ckpt_base, vqvae_base, device)
    rows = evaluate_model(cfg, loaded, device, n_sample_steps=10)

    import pandas as pd
    df = pd.DataFrame(rows)
    assert set(df.columns) >= {"model", "anatomy", "subj_id", "psnr", "ssim", "mse"}
    assert len(df) == len(loaded["val_ds"])
    assert df["anatomy"].isin(["AB", "HN", "TH"]).all()
    assert (df["psnr"] > 0).all()
    assert (df["ssim"].between(-1, 1)).all()
    assert (df["mse"] >= 0).all()


from eval_full import save_results
import tempfile, os

def test_save_results_creates_files():
    import pandas as pd
    rows = [
        {"model":"uvit_n1_cpr4","anatomy":"AB","subj_id":"s001","psnr":30.0,"ssim":0.90,"mse":0.001},
        {"model":"uvit_n1_cpr4","anatomy":"HN","subj_id":"s002","psnr":32.0,"ssim":0.91,"mse":0.0009},
        {"model":"uvit_n5_cpr4","anatomy":"AB","subj_id":"s001","psnr":33.0,"ssim":0.92,"mse":0.0008},
        {"model":"uvit_n5_cpr4","anatomy":"HN","subj_id":"s002","psnr":34.0,"ssim":0.93,"mse":0.0007},
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        save_results(rows, output_dir=tmpdir)
        assert os.path.exists(os.path.join(tmpdir, "raw_metrics.csv"))
        assert os.path.exists(os.path.join(tmpdir, "summary_stats.csv"))
        df = pd.read_csv(os.path.join(tmpdir, "raw_metrics.csv"))
        assert len(df) == 4
        summary = pd.read_csv(os.path.join(tmpdir, "summary_stats.csv"))
        assert "psnr_mean" in summary.columns
        assert "ssim_std" in summary.columns

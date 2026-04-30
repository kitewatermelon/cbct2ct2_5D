"""eval_full.py — Stage2 VDM 전체 ablation 평가.

8개 모델을 stage2_vdm.py와 동일한 val split에서 평가.
출력: eval_results/raw_metrics.csv, summary_stats.csv, boxplot_*.png
"""
from __future__ import annotations
import argparse, pathlib
import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from monai.metrics import PSNRMetric, SSIMMetric
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

_psnr_metric = PSNRMetric(max_val=1.0)
_ssim_metric = SSIMMetric(data_range=1.0, spatial_dims=2)


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    _psnr_metric.reset()
    _psnr_metric(pred.float(), target.float())
    return _psnr_metric.aggregate().item()


def compute_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    _ssim_metric.reset()
    _ssim_metric(pred.float(), target.float())
    return _ssim_metric.aggregate().item()


def compute_mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return F.mse_loss(pred.float(), target.float(), reduction="sum").item() / target.numel()


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


# ---------------------------------------------------------------------------
# Per-sample inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_model(
    cfg: dict,
    loaded: dict,
    device: torch.device,
    n_sample_steps: int = 200,
) -> list[dict]:
    from stage2_vdm import _prepare_cond, sample_conditional

    ct_ae        = loaded["ct_ae"]
    cbct_ae      = loaded["cbct_ae"]
    vdm          = loaded["vdm"]
    scale_factor = loaded["scale_factor"]
    val_loader   = loaded["val_loader"]
    subj_anatomy = loaded["subj_anatomy"]
    backbone     = cfg["backbone"]
    n            = cfg["n"]
    mid          = n // 2
    key          = cfg["key"]

    rows = []
    for batch in tqdm(val_loader, desc=key, leave=False):
        ct_img   = batch["ct"].to(device)
        cbct_img = batch["cbct"].to(device)
        subj_ids = batch["subj_id"]

        z_cond = cbct_ae.encode_stage_2_inputs(cbct_img) * scale_factor
        cond   = _prepare_cond(z_cond, backbone)

        sampled_z = sample_conditional(vdm, cond, n_sample_steps, device)
        ct_gen    = ct_ae.decode_stage_2_outputs(sampled_z / scale_factor).float()
        ct_gt     = ct_img[:, mid:mid+1].float()

        for i in range(ct_gen.shape[0]):
            sid = subj_ids[i]
            rows.append({
                "model"   : key,
                "anatomy" : subj_anatomy.get(sid, "UNKNOWN"),
                "subj_id" : sid,
                "psnr"    : compute_psnr(ct_gen[i:i+1], ct_gt[i:i+1]),
                "ssim"    : compute_ssim(ct_gen[i:i+1], ct_gt[i:i+1]),
                "mse"     : compute_mse(ct_gen[i:i+1],  ct_gt[i:i+1]),
            })
    return rows


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_results(rows: list[dict], output_dir: str) -> None:
    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    df.to_csv(out / "raw_metrics.csv", index=False)

    summary = (
        df.groupby(["model", "anatomy"])[["psnr", "ssim", "mse"]]
        .agg(["mean", "std", "median"])
    )
    summary.columns = ["_".join(c) for c in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(out / "summary_stats.csv", index=False)
    print(f"[저장] {out/'raw_metrics.csv'}  ({len(df)}행)")
    print(f"[저장] {out/'summary_stats.csv'}")


def save_boxplots(df: "pd.DataFrame", output_dir: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    out    = pathlib.Path(output_dir)
    anatms = sorted(df["anatomy"].unique())
    models = df["model"].unique().tolist()
    colors = {"AB": "#4C72B0", "HN": "#DD8452", "TH": "#55A868"}

    for metric, ylabel, title in [
        ("psnr", "PSNR (dB)", "PSNR by Model"),
        ("ssim", "SSIM",      "SSIM by Model"),
        ("mse",  "MSE",       "MSE by Model"),
    ]:
        fig, ax = plt.subplots(figsize=(12, 5))
        x_positions = range(len(models))
        width = 0.25
        for ai, anat in enumerate(anatms):
            sub = df[df["anatomy"] == anat]
            data_per_model = [sub[sub["model"] == m][metric].values for m in models]
            positions = [x + (ai - 1) * width for x in x_positions]
            bp = ax.boxplot(
                data_per_model, positions=positions, widths=width * 0.8,
                patch_artist=True, manage_ticks=False,
            )
            for patch in bp["boxes"]:
                patch.set_facecolor(colors.get(anat, "gray"))
                patch.set_alpha(0.7)

        ax.set_xticks(list(x_positions))
        ax.set_xticklabels(models, rotation=30, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        handles = [plt.Rectangle((0,0),1,1, fc=colors.get(a,"gray"), alpha=0.7)
                   for a in anatms]
        ax.legend(handles, anatms, title="Anatomy")
        fig.tight_layout()
        fig.savefig(out / f"boxplot_{metric}.png", dpi=150)
        plt.close(fig)
        print(f"[저장] {out}/boxplot_{metric}.png")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser(description="Stage2 VDM 전체 ablation 평가")
    p.add_argument("--data_root",       type=str,   default="/home/dministrator/s2025")
    p.add_argument("--ckpt_base",       type=str,   default="checkpoints_임베딩1/stage2_vdm")
    p.add_argument("--vqvae_base",      type=str,   default="checkpoints_임베딩1/stage1_vqvae")
    p.add_argument("--output_dir",      type=str,   default="eval_results")
    p.add_argument("--device",          type=int,   default=0)
    p.add_argument("--n_sample_steps",  type=int,   default=200)
    p.add_argument("--batch_size",      type=int,   default=8)
    p.add_argument("--num_workers",     type=int,   default=4)
    p.add_argument("--val_ratio",       type=float, default=0.2)
    p.add_argument("--seed",            type=int,   default=42)
    return p.parse_args()


def main():
    args   = get_args()
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    set_determinism(seed=args.seed)

    all_rows: list[dict] = []
    for cfg in MODEL_CONFIGS:
        print(f"\n>>> {cfg['key']}")
        try:
            loaded = load_model_for_eval(
                cfg=cfg,
                data_root=args.data_root,
                ckpt_base=args.ckpt_base,
                vqvae_base=args.vqvae_base,
                device=device,
                val_ratio=args.val_ratio,
                seed=args.seed,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
            rows = evaluate_model(cfg, loaded, device, n_sample_steps=args.n_sample_steps)
            all_rows.extend(rows)
            df_m = pd.DataFrame(rows)
            for anat in ["AB", "HN", "TH"]:
                sub = df_m[df_m["anatomy"] == anat]
                if sub.empty:
                    continue
                print(f"    [{anat}] PSNR={sub['psnr'].mean():.2f}±{sub['psnr'].std():.2f}"
                      f"  SSIM={sub['ssim'].mean():.4f}  MSE={sub['mse'].mean():.5f}  n={len(sub)}")
        except Exception as e:
            print(f"    [건너뜀] {e}")

    if not all_rows:
        print("평가 결과 없음.")
        return

    df = pd.DataFrame(all_rows)
    save_results(all_rows, args.output_dir)
    save_boxplots(df, args.output_dir)
    print(f"\n완료: {args.output_dir}/")


if __name__ == "__main__":
    main()

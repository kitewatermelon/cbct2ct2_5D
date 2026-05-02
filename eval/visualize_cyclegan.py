"""visualize_cyclegan.py — CycleGAN 정성 분석: 부위별 5개 샘플 시각화.

각 anatomy(AB/HN/TH)에 대해 PSNR 분포에서 균등하게 5개 샘플을 선택하고,
CBCT / CT GT / CycleGAN 생성 이미지를 나란히 표시합니다.

Usage:
    python eval/visualize_cyclegan.py
    python eval/visualize_cyclegan.py --n_samples 5 --output_dir eval_results/viz_cyclegan
"""
from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import torch
from torch.utils.data import random_split

from data.synthrad2025 import SynthRad2025, build_transforms
from utils.mha import load_mha

ANATOMIES = ["AB", "HN", "TH"]
ANATOMY_NAMES = {"AB": "Abdomen", "HN": "Head & Neck", "TH": "Thorax"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pick_samples(
    df_anat: pd.DataFrame,
    n: int,
) -> list[str]:
    """PSNR 분포에서 균등하게 n개 subject 선택 (best/worst 포함)."""
    df_sorted = df_anat.sort_values("psnr").reset_index(drop=True)
    total = len(df_sorted)
    if total <= n:
        return df_sorted["subj_id"].tolist()
    # 분위수 기반 균등 선택
    indices = np.linspace(0, total - 1, n, dtype=int)
    return df_sorted.loc[indices, "subj_id"].tolist()


def build_subj_lookup(data_root: str, val_ratio: float, seed: int) -> dict[str, dict]:
    """subj_id → {'cbct': np.ndarray, 'ct': np.ndarray} 룩업 테이블."""
    tf = build_transforms(["cbct", "ct"], (128, 128), augment=False)
    full_ds = SynthRad2025(
        root=f"{data_root}/dataset/train/n1",
        modality=["cbct", "ct"],
        anatomy=ANATOMIES,
        transform=tf,
    )
    n_val = int(len(full_ds) * val_ratio)
    n_train = len(full_ds) - n_val
    _, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )
    lookup: dict[str, dict] = {}
    for idx in val_ds.indices:
        sample = full_ds[idx]
        sid = sample["subj_id"]
        lookup[sid] = {
            "cbct": sample["cbct"][0].numpy(),   # (H, W)
            "ct":   sample["ct"][0].numpy(),
        }
    return lookup


# ---------------------------------------------------------------------------
# Per-anatomy figure
# ---------------------------------------------------------------------------

def plot_anatomy(
    anat: str,
    subj_ids: list[str],
    lookup: dict[str, dict],
    gen_dir: pathlib.Path,
    metrics: dict[str, dict],
    out_path: pathlib.Path,
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> None:
    n = len(subj_ids)
    fig = plt.figure(figsize=(4 * 3 + 1, 4 * n + 0.8), constrained_layout=False)
    fig.suptitle(
        f"CycleGAN 정성 분석 — {ANATOMY_NAMES[anat]} ({anat})",
        fontsize=14, fontweight="bold", y=0.995,
    )

    gs = gridspec.GridSpec(
        n, 4,
        figure=fig,
        wspace=0.04, hspace=0.18,
        left=0.01, right=0.99,
        top=0.97, bottom=0.02,
    )

    col_titles = ["CBCT (입력)", "CT GT", "CycleGAN 생성", "오차 |GT−Gen|"]

    for col, title in enumerate(col_titles):
        ax = fig.add_subplot(gs[0, col])
        ax.set_title(title, fontsize=11, pad=4)
        ax.axis("off")

    for row, sid in enumerate(subj_ids):
        data    = lookup[sid]
        cbct    = data["cbct"]
        ct_gt   = data["ct"]
        ct_gen  = load_mha(gen_dir / f"{sid}.mha")

        err     = np.abs(ct_gt - ct_gen)
        m       = metrics.get(sid, {})
        psnr    = m.get("psnr", float("nan"))
        ssim    = m.get("ssim", float("nan"))

        images  = [cbct, ct_gt, ct_gen, err]
        cmaps   = ["gray", "gray", "gray", "hot"]
        vmins   = [vmin,  vmin,   vmin,   0.0]
        vmaxs   = [vmax,  vmax,   vmax,   0.3]

        for col, (img, cmap, vn, vx) in enumerate(zip(images, cmaps, vmins, vmaxs)):
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(img, cmap=cmap, vmin=vn, vmax=vx, interpolation="nearest")
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(sid, fontsize=7, rotation=0, labelpad=60, va="center")
            if col == 2 and row == 0:
                pass  # title already set above
            if col == 2:
                ax.set_title(
                    f"PSNR {psnr:.2f} dB  SSIM {ssim:.4f}",
                    fontsize=7, pad=2, color="#1a1aff",
                )

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  저장: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CycleGAN 정성 분석 시각화")
    p.add_argument("--data_root",  type=str, default="/home/dministrator/s2025")
    p.add_argument("--gen_dir",    type=str, default="eval_results/gen")
    p.add_argument("--metrics_csv",type=str, default="eval_results/raw_metrics.csv")
    p.add_argument("--output_dir", type=str, default="eval_results/viz_cyclegan")
    p.add_argument("--n_samples",  type=int, default=5,
                   help="부위별 샘플 수 (PSNR 분위수 균등 선택)")
    p.add_argument("--val_ratio",  type=float, default=0.2)
    p.add_argument("--seed",       type=int,   default=42)
    return p.parse_args()


def main() -> None:
    args    = get_args()
    gen_dir = pathlib.Path(args.gen_dir) / "cyclegan"
    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # metrics 로드
    df = pd.read_csv(args.metrics_csv)
    df = df[df["model"] == "cyclegan"]
    metrics: dict[str, dict] = {
        row["subj_id"]: {"psnr": row["psnr"], "ssim": row["ssim"]}
        for _, row in df.iterrows()
    }

    # val 데이터 로드
    print("val split 로드 중...")
    lookup = build_subj_lookup(args.data_root, args.val_ratio, args.seed)

    for anat in ANATOMIES:
        print(f"\n[{ANATOMY_NAMES[anat]}]")
        df_anat  = df[df["anatomy"] == anat]
        subj_ids = pick_samples(df_anat, args.n_samples)

        for sid in subj_ids:
            m = metrics.get(sid, {})
            print(f"  {sid}  PSNR={m.get('psnr', float('nan')):.2f}  SSIM={m.get('ssim', float('nan')):.4f}")

        plot_anatomy(
            anat       = anat,
            subj_ids   = subj_ids,
            lookup     = lookup,
            gen_dir    = gen_dir,
            metrics    = metrics,
            out_path   = out_dir / f"cyclegan_{anat}.png",
        )

    print(f"\n완료: {out_dir}/")


if __name__ == "__main__":
    main()

"""best_case_analysis.py — n5_cpr4 vs n1_cpr4 최적 케이스 분석.

raw_metrics.csv (eval_full.py 출력)에서 uvit_n5_cpr4와 uvit_n1_cpr4의
PSNR/SSIM/MSE를 열방향(전체 샘플 × 두 모델) min-max 정규화한 뒤
combined score = psnr_norm + ssim_norm + (1 - mse_norm) 를 비교.

score_n5 - score_n1 이 가장 큰 샘플을 부위별 top-3 출력 및 시각화:
  컬럼: CBCT | gen(n1) | gen(n5) | GT CT

Usage:
    python best_case_analysis.py \\
        --metrics_csv  eval_results/raw_metrics.csv \\
        --gen_dir      eval_results/gen \\
        --data_root    /home/dministrator/s2025 \\
        --output_dir   eval_results/best_case
"""
from __future__ import annotations
import argparse, pathlib
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from monai.utils import set_determinism
from tqdm.auto import tqdm

from eval.eval_full import build_val_loader
from utils.mha import load_mha

MODEL_N5 = "uvit_n5_cpr4"
MODEL_N1 = "uvit_n1_cpr4"
N5, N1   = 5, 1


# ---------------------------------------------------------------------------
# Score computation
# ---------------------------------------------------------------------------

def build_score_df(metrics_csv: str) -> pd.DataFrame:
    df = pd.read_csv(metrics_csv)
    df = df[df["model"].isin([MODEL_N5, MODEL_N1])].copy()

    # min-max 정규화: 두 모델의 전체 샘플을 합쳐서 열방향으로
    for col in ["psnr", "ssim", "mse"]:
        lo, hi = df[col].min(), df[col].max()
        df[f"{col}_norm"] = (df[col] - lo) / (hi - lo + 1e-12)

    # combined score (MSE는 낮을수록 좋으므로 반전)
    df["score"] = df["psnr_norm"] + df["ssim_norm"] + (1.0 - df["mse_norm"])
    return df


def find_top_cases(score_df: pd.DataFrame, top_k: int = 3) -> pd.DataFrame:
    """부위별로 score_n5 - score_n1 가 큰 top_k 샘플 반환."""
    n5 = score_df[score_df["model"] == MODEL_N5][
        ["subj_id", "anatomy", "psnr", "ssim", "mse", "psnr_norm", "ssim_norm", "mse_norm", "score"]
    ].rename(columns=lambda c: f"n5_{c}" if c not in ("subj_id", "anatomy") else c)

    n1 = score_df[score_df["model"] == MODEL_N1][
        ["subj_id", "score", "psnr", "ssim", "mse"]
    ].rename(columns=lambda c: f"n1_{c}" if c != "subj_id" else c)

    merged = n5.merge(n1, on="subj_id")
    merged["delta"] = merged["n5_score"] - merged["n1_score"]

    top_rows = (
        merged.sort_values("delta", ascending=False)
              .groupby("anatomy", group_keys=False)
              .head(top_k)
    )
    return top_rows.sort_values(["anatomy", "delta"], ascending=[True, False])


# ---------------------------------------------------------------------------
# Dataset lookup helpers
# ---------------------------------------------------------------------------

def collect_samples(loader, target_sids: set[str], mid: int) -> dict[str, dict]:
    """target_sids에 해당하는 샘플만 {sid: {"cbct": np, "gt": np}} 로 반환."""
    out: dict[str, dict] = {}
    for batch in tqdm(loader, desc="loading GT/CBCT", leave=False):
        for i, sid in enumerate(batch["subj_id"]):
            if sid in target_sids and sid not in out:
                out[sid] = {
                    "cbct": batch["cbct"][i, mid].numpy(),   # (H, W)  mid slice
                    "gt"  : batch["ct"][i, mid].numpy(),
                }
        if len(out) == len(target_sids):
            break
    return out


def load_gen(gen_dir: pathlib.Path, model_key: str, sid: str) -> np.ndarray:
    return load_mha(gen_dir / model_key / f"{sid}.mha")


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def visualize_cases(
    top_cases: pd.DataFrame,
    samples_n5: dict,
    samples_n1: dict,
    gen_dir: pathlib.Path,
    output_dir: pathlib.Path,
) -> None:
    """부위별로 한 PNG씩 저장. 열: CBCT | gen_n1 | gen_n5 | GT CT"""
    cmap = "gray"

    for anat, grp in top_cases.groupby("anatomy"):
        n_rows = len(grp)
        fig, axes = plt.subplots(n_rows, 4, figsize=(14, 3.5 * n_rows))
        if n_rows == 1:
            axes = axes[np.newaxis, :]  # ensure 2-D

        col_titles = ["CBCT (input)", f"gen {MODEL_N1}", f"gen {MODEL_N5}", "GT CT"]
        for ax, title in zip(axes[0], col_titles):
            ax.set_title(title, fontsize=10)

        for row_idx, (_, case) in enumerate(grp.iterrows()):
            sid = case["subj_id"]

            cbct_img = samples_n5[sid]["cbct"]
            gt_img   = samples_n5[sid]["gt"]
            gen_n5   = load_gen(gen_dir, MODEL_N5, sid)
            # n1 uses n=1 so mid=0, cbct/gt slice selection differs but gen is same 2D image
            gen_n1   = load_gen(gen_dir, MODEL_N1, sid)

            imgs = [cbct_img, gen_n1, gen_n5, gt_img]
            vmin = min(i.min() for i in [gen_n5, gen_n1, gt_img])
            vmax = max(i.max() for i in [gen_n5, gen_n1, gt_img])

            for col_idx, img in enumerate(imgs):
                ax = axes[row_idx, col_idx]
                ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
                ax.axis("off")

            # row annotation
            axes[row_idx, 0].set_ylabel(
                f"{sid}\nΔ={case['delta']:.3f}\n"
                f"n5 PSNR={case['n5_psnr']:.2f} SSIM={case['n5_ssim']:.4f}\n"
                f"n1 PSNR={case['n1_psnr']:.2f} SSIM={case['n1_ssim']:.4f}",
                fontsize=7, rotation=0, labelpad=90, va="center",
            )

        fig.suptitle(f"Best cases ({MODEL_N5} vs {MODEL_N1}) — anatomy: {anat}", fontsize=11)
        fig.tight_layout()
        path = output_dir / f"best_cases_{anat}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[저장] {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser(description="n5_cpr4 vs n1_cpr4 최적 케이스 분석")
    p.add_argument("--metrics_csv",  type=str, default="eval_results/raw_metrics.csv")
    p.add_argument("--gen_dir",      type=str, default="eval_results/gen")
    p.add_argument("--data_root",    type=str, default="/home/dministrator/s2025")
    p.add_argument("--output_dir",   type=str, default="eval_results/best_case")
    p.add_argument("--top_k",        type=int, default=3)
    p.add_argument("--val_ratio",    type=float, default=0.2)
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--batch_size",   type=int, default=8)
    p.add_argument("--num_workers",  type=int, default=4)
    return p.parse_args()


def main():
    args = get_args()
    set_determinism(seed=args.seed)
    out = pathlib.Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    gen_dir = pathlib.Path(args.gen_dir)

    # ── Score 계산 ────────────────────────────────────────────────────────
    score_df  = build_score_df(args.metrics_csv)
    top_cases = find_top_cases(score_df, top_k=args.top_k)

    print("\n=== 부위별 top-{} 케이스 (score_n5 - score_n1 기준) ===".format(args.top_k))
    print(top_cases[[
        "anatomy", "subj_id",
        "n5_psnr", "n5_ssim", "n5_mse", "n5_score",
        "n1_psnr", "n1_ssim", "n1_mse", "n1_score",
        "delta",
    ]].to_string(index=False))

    csv_path = out / "top_cases.csv"
    top_cases.to_csv(csv_path, index=False)
    print(f"\n[저장] {csv_path}")

    # ── GT / CBCT 로딩 (n5 기준 mid slice) ───────────────────────────────
    target_sids = set(top_cases["subj_id"])
    loader_n5 = build_val_loader(
        args.data_root, n=N5,
        val_ratio=args.val_ratio, seed=args.seed,
        batch_size=args.batch_size, num_workers=args.num_workers,
    )
    mid_n5    = N5 // 2
    samples   = collect_samples(loader_n5, target_sids, mid=mid_n5)

    missing = target_sids - set(samples)
    if missing:
        print(f"[경고] 다음 subj_id를 dataset에서 찾지 못했습니다: {missing}")

    # ── 시각화 ───────────────────────────────────────────────────────────
    visualize_cases(top_cases, samples, samples, gen_dir, out)
    print(f"\n완료: {out}/")


if __name__ == "__main__":
    main()

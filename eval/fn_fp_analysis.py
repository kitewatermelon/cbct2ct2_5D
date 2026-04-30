"""fn_fp_analysis.py — Tissue-level FP/FN analysis from generated CT images.

HU 범위(hu_analysis.py의 TISSUES 정의)로 픽셀을 tissue로 분류한 뒤,
GT vs generated 간 불일치를 FP/FN rate로 정량화.

정의:
  FN rate (tissue T) = GT에서 T인 픽셀 중 gen에서 T가 아닌 비율  (miss rate)
  FP rate (tissue T) = gen에서 T인 픽셀 중 GT에서 T가 아닌 비율  (false discovery rate)
  Precision          = TP / (TP + FP)
  Recall             = TP / (TP + FN)  [= 1 - FN rate]

Outputs (in --output_dir):
  fp_fn_stats.csv      — 모델 × tissue × 지표 테이블
  fp_fn_barplot.png    — FP rate / FN rate grouped barplot

Usage:
    python fn_fp_analysis.py \\
        --gen_dir   eval_results/gen \\
        --data_root /home/dministrator/s2025 \\
        --output_dir eval_results/fp_fn
"""
from __future__ import annotations
import argparse, json, pathlib
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from monai.utils import set_determinism
from tqdm.auto import tqdm

from eval.eval_full import MODEL_CONFIGS, build_val_loader
from utils.hu import TISSUES, TISSUE_ORDER, to_hu, classify_tissue
from utils.mha import load_mha


# ---------------------------------------------------------------------------
# Per-model FP/FN computation
# ---------------------------------------------------------------------------

def compute_fp_fn(
    cfg: dict,
    gen_dir: pathlib.Path,
    data_root: str,
    val_ratio: float = 0.2,
    seed: int = 42,
    batch_size: int = 4,
    num_workers: int = 4,
) -> pd.DataFrame:
    key       = cfg["key"]
    n         = cfg["n"]
    mid       = n // 2
    model_dir = gen_dir / key

    meta_path = model_dir / "meta.json"
    with open(meta_path) as f:
        subj_anatomy: dict[str, str] = json.load(f)

    val_loader = build_val_loader(
        data_root, n, val_ratio=val_ratio, seed=seed,
        batch_size=batch_size, num_workers=num_workers,
    )

    # tissue별 누적 카운터: {tissue_idx: [TP, FP, FN, TN]}
    counts = {idx: np.zeros(4, dtype=np.int64) for idx in range(len(TISSUES))}

    for batch in tqdm(val_loader, desc=key, leave=False):
        ct_img   = batch["ct"]
        subj_ids = batch["subj_id"]
        gt_norm  = ct_img[:, mid:mid+1].float()   # (B, 1, H, W)

        for i, sid in enumerate(subj_ids):
            gen_arr = load_mha(model_dir / f"{sid}.mha")   # (H, W) [0,1]
            gen_hu = to_hu(gen_arr)
            gt_hu  = to_hu(gt_norm[i, 0].numpy())

            gen_cls = classify_tissue(gen_hu)      # (H, W)  int8
            gt_cls  = classify_tissue(gt_hu)

            for t_idx in range(len(TISSUES)):
                pred_t = gen_cls == t_idx
                true_t = gt_cls  == t_idx
                counts[t_idx][0] += int(( pred_t &  true_t).sum())  # TP
                counts[t_idx][1] += int(( pred_t & ~true_t).sum())  # FP
                counts[t_idx][2] += int((~pred_t &  true_t).sum())  # FN
                counts[t_idx][3] += int((~pred_t & ~true_t).sum())  # TN

    rows = []
    for t_idx, name in enumerate(TISSUE_ORDER):
        TP, FP, FN, TN = counts[t_idx]
        total_pred_t = TP + FP      # gen에서 T로 예측한 픽셀 수
        total_true_t = TP + FN      # GT에서 실제 T인 픽셀 수
        rows.append({
            "model"    : key,
            "tissue"   : name,
            "TP"       : int(TP),
            "FP"       : int(FP),
            "FN"       : int(FN),
            "TN"       : int(TN),
            "fp_rate"  : FP / total_pred_t if total_pred_t > 0 else float("nan"),
            "fn_rate"  : FN / total_true_t if total_true_t > 0 else float("nan"),
            "precision": TP / total_pred_t if total_pred_t > 0 else float("nan"),
            "recall"   : TP / total_true_t if total_true_t > 0 else float("nan"),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_fp_fn(df: pd.DataFrame, output_dir: pathlib.Path) -> None:
    models   = df["model"].unique().tolist()
    tissues  = TISSUE_ORDER
    n_m      = len(models)
    x        = np.arange(len(tissues))
    width    = 0.8 / n_m

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    for ax, (metric, label, color_base) in zip(
        axes,
        [
            ("fp_rate", "FP rate  (false discovery rate)", "#E57373"),
            ("fn_rate", "FN rate  (miss rate)",             "#64B5F6"),
        ],
    ):
        colors = plt.cm.Set2(np.linspace(0, 0.8, n_m))
        for mi, model in enumerate(models):
            sub    = df[df["model"] == model].set_index("tissue")
            vals   = [sub.loc[t, metric] if t in sub.index else 0.0 for t in tissues]
            offset = (mi - n_m / 2 + 0.5) * width
            ax.bar(x + offset, vals, width=width * 0.9,
                   label=model, color=colors[mi], alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(tissues, rotation=20, ha="right")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7, loc="upper right")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

    fig.suptitle("Tissue-level FP / FN rates by Model", fontsize=12)
    fig.tight_layout()
    path = output_dir / "fp_fn_barplot.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[저장] {path}")


def plot_fp_fn_heatmap(df: pd.DataFrame, output_dir: pathlib.Path) -> None:
    """모델 × tissue 히트맵 (FP rate / FN rate 각각)."""
    for metric, title in [("fp_rate", "FP rate"), ("fn_rate", "FN rate")]:
        pivot = df.pivot(index="model", columns="tissue", values=metric)
        pivot = pivot[TISSUE_ORDER]   # 열 순서 고정

        fig, ax = plt.subplots(figsize=(len(TISSUE_ORDER) * 1.5 + 1, len(pivot) * 0.7 + 1))
        im = ax.imshow(pivot.values, vmin=0, vmax=1, cmap="RdYlGn_r", aspect="auto")
        ax.set_xticks(range(len(TISSUE_ORDER)))
        ax.set_xticklabels(TISSUE_ORDER, rotation=25, ha="right")
        ax.set_yticks(range(len(pivot)))
        ax.set_yticklabels(pivot.index, fontsize=8)
        ax.set_title(f"{title} heatmap (lower = better)")

        for r in range(pivot.shape[0]):
            for c in range(pivot.shape[1]):
                val = pivot.values[r, c]
                ax.text(c, r, f"{val:.2%}", ha="center", va="center",
                        fontsize=7, color="black")

        fig.colorbar(im, ax=ax, fraction=0.02)
        fig.tight_layout()
        path = output_dir / f"fp_fn_heatmap_{metric}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"[저장] {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser(description="Tissue-level FP/FN analysis from .mha gen files")
    p.add_argument("--gen_dir",    type=str, default="eval_results/gen")
    p.add_argument("--data_root",  type=str, default="/home/dministrator/s2025")
    p.add_argument("--output_dir", type=str, default="eval_results/fp_fn")
    p.add_argument("--val_ratio",  type=float, default=0.2)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--batch_size", type=int,   default=4)
    p.add_argument("--num_workers",type=int,   default=4)
    p.add_argument("--keys", nargs="+", default=None,
                   help="분석할 모델 key 목록 (기본: 전체 8개)")
    return p.parse_args()


def main():
    args = get_args()
    set_determinism(seed=args.seed)
    out     = pathlib.Path(args.output_dir)
    gen_dir = pathlib.Path(args.gen_dir)
    out.mkdir(parents=True, exist_ok=True)

    configs = MODEL_CONFIGS
    if args.keys:
        configs = [c for c in MODEL_CONFIGS if c["key"] in args.keys]
    print(f"분석 대상 모델: {[c['key'] for c in configs]}")

    all_dfs: list[pd.DataFrame] = []
    for cfg in configs:
        print(f"\n>>> {cfg['key']}")
        try:
            df = compute_fp_fn(
                cfg, gen_dir, args.data_root,
                val_ratio=args.val_ratio, seed=args.seed,
                batch_size=args.batch_size, num_workers=args.num_workers,
            )
            all_dfs.append(df)
            print(df[["tissue", "fp_rate", "fn_rate", "precision", "recall"]].to_string(index=False))
        except Exception as e:
            print(f"  [건너뜀] {e}")

    if not all_dfs:
        print("결과 없음.")
        return

    full_df = pd.concat(all_dfs, ignore_index=True)
    csv_path = out / "fp_fn_stats.csv"
    full_df.to_csv(csv_path, index=False)
    print(f"\n[저장] {csv_path}")

    plot_fp_fn(full_df, out)
    plot_fp_fn_heatmap(full_df, out)
    print(f"\n완료: {out}/")


if __name__ == "__main__":
    main()

"""ttest_analysis.py — 모델 간 pairwise paired t-test.

raw_metrics.csv를 읽어 anatomy(AB/HN/TH) × metric(psnr/ssim/mse) 별로
모든 모델 쌍의 paired t-test를 수행하고 결과를 정리한다.

같은 subj_id가 모든 모델에 존재하므로 subj_id로 매칭한 paired t-test를 사용.

출력:
  ttest_results.csv       — 전체 pairwise 결과 (long format)
  ttest_pvalue_heatmap_<anatomy>_<metric>.png — p-value 히트맵 (anatomy × metric)
"""
from __future__ import annotations
import argparse, itertools, pathlib
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


METRICS   = ["psnr", "ssim", "mse"]
ANATOMIES = ["AB", "HN", "TH"]
ALPHA     = 0.05


# ---------------------------------------------------------------------------
# T-test
# ---------------------------------------------------------------------------

def run_ttest(df: pd.DataFrame) -> pd.DataFrame:
    """모든 anatomy × metric × model pair에 대해 paired t-test 수행."""
    models = df["model"].unique().tolist()
    pairs  = list(itertools.combinations(models, 2))

    rows = []
    for anatomy in ANATOMIES:
        sub = df[df["anatomy"] == anatomy]

        # subj_id × model pivot (각 metric별)
        for metric in METRICS:
            pivot = (
                sub.pivot_table(index="subj_id", columns="model", values=metric)
                   .dropna(axis=0)   # 두 모델 모두 있는 subject만
            )

            for model_a, model_b in pairs:
                if model_a not in pivot.columns or model_b not in pivot.columns:
                    continue

                a_vals = pivot[model_a].values
                b_vals = pivot[model_b].values
                n      = len(a_vals)

                t_stat, p_val = stats.ttest_rel(a_vals, b_vals)
                mean_diff = (a_vals - b_vals).mean()

                rows.append({
                    "anatomy"   : anatomy,
                    "metric"    : metric,
                    "model_a"   : model_a,
                    "model_b"   : model_b,
                    "n"         : n,
                    "mean_a"    : a_vals.mean(),
                    "mean_b"    : b_vals.mean(),
                    "mean_diff" : mean_diff,   # a - b
                    "t_stat"    : round(t_stat, 4),
                    "p_value"   : p_val,
                    "significant": p_val < ALPHA,
                })

    result = pd.DataFrame(rows)
    # Bonferroni correction (모델 수 × anatomy 수 × metric 수)
    n_tests = len(rows)
    result["p_bonferroni"] = (result["p_value"] * n_tests).clip(upper=1.0)
    result["sig_bonferroni"] = result["p_bonferroni"] < ALPHA
    return result


# ---------------------------------------------------------------------------
# Heatmap (p-value matrix per anatomy × metric)
# ---------------------------------------------------------------------------

def plot_pvalue_heatmaps(result: pd.DataFrame, output_dir: pathlib.Path):
    models = sorted(set(result["model_a"].tolist() + result["model_b"].tolist()))
    n = len(models)

    for anatomy in ANATOMIES:
        for metric in METRICS:
            sub = result[(result["anatomy"] == anatomy) & (result["metric"] == metric)]
            if sub.empty:
                continue

            # build symmetric p-value matrix
            mat = pd.DataFrame(np.nan, index=models, columns=models)
            for _, row in sub.iterrows():
                mat.loc[row["model_a"], row["model_b"]] = row["p_value"]
                mat.loc[row["model_b"], row["model_a"]] = row["p_value"]
            np.fill_diagonal(mat.values, 1.0)

            fig, ax = plt.subplots(figsize=(8, 6))
            mask = np.zeros_like(mat.values, dtype=bool)
            mask[np.triu_indices_from(mask, k=1)] = True  # upper triangle

            sns.heatmap(
                mat.astype(float),
                annot=True, fmt=".3f",
                cmap="RdYlGn_r",
                vmin=0, vmax=0.1,
                linewidths=0.5,
                ax=ax,
                mask=mask,
                cbar_kws={"label": "p-value"},
            )
            ax.set_title(f"Paired t-test p-values  |  {anatomy}  |  {metric.upper()}")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)

            # mark significance threshold
            ax.text(
                0.01, 0.01,
                f"p < {ALPHA} = significant  |  green = more significant",
                transform=ax.transAxes, fontsize=7, color="gray",
            )

            fig.tight_layout()
            path = output_dir / f"ttest_pvalue_{anatomy}_{metric}.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            print(f"[저장] {path}")


# ---------------------------------------------------------------------------
# Summary table (per anatomy: which pairs are significant?)
# ---------------------------------------------------------------------------

def print_summary(result: pd.DataFrame):
    for anatomy in ANATOMIES:
        print(f"\n{'='*60}")
        print(f"  Anatomy: {anatomy}")
        print(f"{'='*60}")
        for metric in METRICS:
            sub = result[
                (result["anatomy"] == anatomy) &
                (result["metric"]  == metric)
            ].sort_values("p_value")
            sig = sub[sub["significant"]]
            print(f"\n  [{metric.upper()}]  유의한 쌍: {len(sig)}/{len(sub)}")
            if not sig.empty:
                for _, r in sig.iterrows():
                    direction = ">" if r["mean_diff"] > 0 else "<"
                    print(f"    {r['model_a']} {direction} {r['model_b']}"
                          f"  Δ={r['mean_diff']:+.4f}"
                          f"  t={r['t_stat']:.3f}  p={r['p_value']:.4f}"
                          f"{'*' if r['sig_bonferroni'] else ''}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser(description="Pairwise paired t-test from raw_metrics.csv")
    p.add_argument("--input",      type=str, default="eval_results/raw_metrics.csv")
    p.add_argument("--output_dir", type=str, default="eval_results/ttest")
    p.add_argument("--no_heatmap", action="store_true", help="히트맵 생략")
    return p.parse_args()


def main():
    args = get_args()
    out  = pathlib.Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    print(f"로드: {args.input}  ({len(df)}행, {df['model'].nunique()}개 모델)")

    result = run_ttest(df)

    csv_path = out / "ttest_results.csv"
    result.to_csv(csv_path, index=False)
    print(f"[저장] {csv_path}  ({len(result)}쌍)")

    print_summary(result)

    if not args.no_heatmap:
        try:
            import seaborn  # noqa: F401
            plot_pvalue_heatmaps(result, out)
        except ImportError:
            print("[경고] seaborn 없음 — 히트맵 생략 (pip install seaborn)")

    print(f"\n완료: {out}/")
    print("  * = Bonferroni 보정 후에도 유의")


if __name__ == "__main__":
    main()

"""verify_preprocessed.py — 전처리 결과 시각화 검증 스크립트.

사용법:
    python data/verify_preprocessed.py \
        --preprocessed_root ../s2025/dataset/preprocessed \
        --output_dir verify_output \
        [--n_slices 5] \
        [--n_subjects 3] \
        [--seed 0]

출력:
    verify_output/
        overview.png         — 여러 피험자 × modality 그리드
        n_slices_compare.png — n1/n3/n5/n7/n9 같은 피험자 비교
        val_check.png        — val middle slice 확인
        stats.txt            — 픽셀값 통계 요약
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# 유틸
# ---------------------------------------------------------------------------

def load_npy(path: pathlib.Path) -> np.ndarray:
    return np.load(str(path))  # (n_slices, H, W)


def center_slice(arr: np.ndarray) -> np.ndarray:
    """(n, H, W) → 중앙 슬라이스 (H, W)."""
    return arr[arr.shape[0] // 2]


def list_subjects(split_dir: pathlib.Path, n_slices: int) -> List[pathlib.Path]:
    d = split_dir / f"n{n_slices}"
    if not d.exists():
        return []
    return sorted(p for p in d.iterdir() if p.is_dir())


def available_modalities(subj_dir: pathlib.Path) -> List[str]:
    mods = []
    for m in ("cbct", "ct", "mask"):
        if any(subj_dir.glob(f"{m}_*.npy")):
            mods.append(m)
    return mods


def pick_file(subj_dir: pathlib.Path, modality: str) -> Optional[pathlib.Path]:
    files = sorted(subj_dir.glob(f"{modality}_*.npy"))
    if not files:
        return None
    return files[len(files) // 2]   # middle


# ---------------------------------------------------------------------------
# Plot 1 — overview: subjects × modalities (중앙 슬라이스)
# ---------------------------------------------------------------------------

def plot_overview(
    preprocessed_root: pathlib.Path,
    n_slices: int,
    n_subjects: int,
    split: str,
    seed: int,
    output_path: pathlib.Path,
) -> None:
    split_dir = preprocessed_root / split
    subjects = list_subjects(split_dir, n_slices)
    if not subjects:
        print(f"[skip] {split_dir}/n{n_slices} 없음")
        return

    rng = random.Random(seed)
    chosen = rng.sample(subjects, min(n_subjects, len(subjects)))
    modalities = available_modalities(chosen[0])

    n_rows = len(chosen)
    n_cols = len(modalities)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 3, n_rows * 3 + 0.5),
                             squeeze=False)
    fig.suptitle(f"Overview — split={split}  n_slices={n_slices}", fontsize=13)

    cmap_map = {"cbct": "gray", "ct": "gray", "mask": "hot"}

    for r, subj_dir in enumerate(chosen):
        for c, mod in enumerate(modalities):
            ax = axes[r][c]
            fpath = pick_file(subj_dir, mod)
            if fpath is None:
                ax.axis("off")
                continue
            arr = load_npy(fpath)
            img = center_slice(arr)
            vmin, vmax = (0, 1) if mod != "mask" else (0, arr.max())
            ax.imshow(img, cmap=cmap_map.get(mod, "gray"), vmin=vmin, vmax=vmax)
            ax.set_title(f"{subj_dir.name}\n{mod}  slice={fpath.stem.split('_')[1]}",
                         fontsize=7)
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[saved] {output_path}")


# ---------------------------------------------------------------------------
# Plot 2 — n_slices 비교: 같은 피험자, n1/n3/n5/n7/n9 각 슬라이스 윈도우
# ---------------------------------------------------------------------------

def plot_n_slices_compare(
    preprocessed_root: pathlib.Path,
    split: str,
    modality: str,
    seed: int,
    output_path: pathlib.Path,
) -> None:
    n_list = [n for n in [1, 3, 5, 7, 9]
              if (preprocessed_root / split / f"n{n}").exists()]
    if not n_list:
        print("[skip] n_slices 디렉토리 없음")
        return

    # 공통 피험자 찾기
    sets = [set(p.name for p in list_subjects(preprocessed_root / split, n))
            for n in n_list]
    common = sorted(set.intersection(*sets))
    if not common:
        print("[skip] 공통 피험자 없음")
        return

    rng = random.Random(seed)
    subj_id = rng.choice(common)

    # n별 최대 슬라이스 수
    max_n = max(n_list)
    n_cols = len(n_list)

    fig, axes = plt.subplots(max_n, n_cols,
                             figsize=(n_cols * 3, max_n * 3 + 0.5),
                             squeeze=False)
    fig.suptitle(f"n_slices 비교 — {subj_id}  modality={modality}  split={split}",
                 fontsize=12)

    for c, n in enumerate(n_list):
        subj_dir = preprocessed_root / split / f"n{n}" / subj_id
        fpath = pick_file(subj_dir, modality)
        if fpath is None:
            for r in range(max_n):
                axes[r][c].axis("off")
            continue
        arr = load_npy(fpath)   # (n, H, W)

        for r in range(max_n):
            ax = axes[r][c]
            if r < n:
                ax.imshow(arr[r], cmap="gray", vmin=0, vmax=1)
                label = "center" if r == n // 2 else f"[{r - n//2:+d}]"
                ax.set_title(label, fontsize=7)
            else:
                ax.axis("off")
            if r == 0:
                ax.set_xlabel(f"n={n}", fontsize=9)
                ax.xaxis.set_label_position("top")
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[saved] {output_path}")


# ---------------------------------------------------------------------------
# Plot 3 — val middle-slice 확인
# ---------------------------------------------------------------------------

def plot_val_check(
    preprocessed_root: pathlib.Path,
    n_slices: int,
    n_subjects: int,
    seed: int,
    output_path: pathlib.Path,
) -> None:
    subjects = list_subjects(preprocessed_root / "val", n_slices)
    if not subjects:
        print("[skip] val 디렉토리 없음")
        return

    rng = random.Random(seed)
    chosen = rng.sample(subjects, min(n_subjects, len(subjects)))
    modalities = available_modalities(chosen[0])

    fig, axes = plt.subplots(len(chosen), len(modalities),
                             figsize=(len(modalities) * 3, len(chosen) * 3 + 0.5),
                             squeeze=False)
    fig.suptitle(f"Val middle-slice check  n_slices={n_slices}", fontsize=12)

    cmap_map = {"cbct": "gray", "ct": "gray", "mask": "hot"}

    for r, subj_dir in enumerate(chosen):
        files = sorted(subj_dir.glob("cbct_*.npy"))
        n_files = len(files)
        for c, mod in enumerate(modalities):
            ax = axes[r][c]
            mfiles = sorted(subj_dir.glob(f"{mod}_*.npy"))
            if not mfiles:
                ax.axis("off")
                continue
            arr = load_npy(mfiles[0])
            img = center_slice(arr)
            vmin, vmax = (0, 1) if mod != "mask" else (0, arr.max())
            ax.imshow(img, cmap=cmap_map.get(mod, "gray"), vmin=vmin, vmax=vmax)
            ax.set_title(f"{subj_dir.name} | {mod}\n"
                         f"file count: {n_files}",
                         fontsize=7)
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[saved] {output_path}")


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def compute_stats(
    preprocessed_root: pathlib.Path,
    n_slices: int,
    n_samples: int,
    seed: int,
    output_path: pathlib.Path,
) -> None:
    lines = []
    for split in ("train", "val"):
        subjects = list_subjects(preprocessed_root / split, n_slices)
        if not subjects:
            continue
        rng = random.Random(seed)
        chosen = rng.sample(subjects, min(n_samples, len(subjects)))
        modalities = available_modalities(chosen[0]) if chosen else []

        lines.append(f"\n=== {split.upper()}  n_slices={n_slices} ===")
        lines.append(f"총 피험자: {len(subjects)}명  (샘플링: {len(chosen)}명)")

        # val의 경우 피험자당 파일 수 확인 (middle only면 1개여야 함)
        file_counts = [len(list(s.glob("cbct_*.npy"))) for s in subjects[:20]]
        lines.append(f"피험자당 cbct 파일 수: min={min(file_counts)} "
                     f"max={max(file_counts)} mean={np.mean(file_counts):.1f}")

        for mod in modalities:
            vals = []
            for subj_dir in chosen:
                fpath = pick_file(subj_dir, mod)
                if fpath:
                    arr = load_npy(fpath)
                    vals.append(arr)
            if not vals:
                continue
            stack = np.concatenate([v.ravel() for v in vals])
            lines.append(
                f"  {mod:6s}: min={stack.min():.4f}  max={stack.max():.4f}  "
                f"mean={stack.mean():.4f}  std={stack.std():.4f}  "
                f"shape={vals[0].shape}"
            )

    text = "\n".join(lines)
    print(text)
    output_path.write_text(text)
    print(f"[saved] {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser(description="전처리 결과 시각화 검증")
    p.add_argument("--preprocessed_root", type=str,
                   default="../s2025/dataset/preprocessed")
    p.add_argument("--output_dir", type=str, default="verify_output")
    p.add_argument("--n_slices",   type=int, default=5)
    p.add_argument("--modality",   type=str, default="cbct",
                   choices=["cbct", "ct", "mask"])
    p.add_argument("--n_subjects", type=int, default=4,
                   help="overview/val_check에 표시할 피험자 수")
    p.add_argument("--n_stats_samples", type=int, default=20,
                   help="통계 계산에 사용할 피험자 수")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = get_args()
    root    = pathlib.Path(args.preprocessed_root)
    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # split_seed42.json 메타 출력
    for json_path in sorted(root.glob("split_seed*.json")):
        meta = json.loads(json_path.read_text())
        print(f"\n[meta] {json_path.name}")
        if "meta" in meta:
            for k, v in meta["meta"].items():
                print(f"  {k}: {v}")
        print(f"  train: {len(meta.get('train', []))}명 | "
              f"val: {len(meta.get('val', []))}명")

    print("\n--- 1. Overview (train) ---")
    plot_overview(
        root, args.n_slices, args.n_subjects,
        split="train", seed=args.seed,
        output_path=out_dir / "overview_train.png",
    )

    print("\n--- 2. Overview (val) ---")
    plot_overview(
        root, args.n_slices, args.n_subjects,
        split="val", seed=args.seed,
        output_path=out_dir / "overview_val.png",
    )

    print("\n--- 3. n_slices 비교 ---")
    plot_n_slices_compare(
        root, split="train", modality=args.modality,
        seed=args.seed,
        output_path=out_dir / "n_slices_compare.png",
    )

    print("\n--- 4. Val middle-slice 확인 ---")
    plot_val_check(
        root, args.n_slices, args.n_subjects,
        seed=args.seed,
        output_path=out_dir / "val_check.png",
    )

    print("\n--- 5. 통계 ---")
    compute_stats(
        root, args.n_slices, args.n_stats_samples,
        seed=args.seed,
        output_path=out_dir / "stats.txt",
    )

    print(f"\n완료. 결과: {out_dir}/")


if __name__ == "__main__":
    main()

"""preprocess.py — SynthRad2025 슬라이스 전처리 스크립트.

실행 (최초 1회):
    python data/preprocess.py \
        --data_root ../s2025/dataset \
        --output_root ../s2025/dataset/preprocessed \
        --n_slices 1 3 5 7 9 \
        --anatomy AB HN TH \
        --spatial_size 128 \
        --offset 10 \
        --val_ratio 0.2 \
        --seed 42 \
        --num_workers 8

출력 구조:
    {output_root}/
        split_seed{seed}.json          <- 환자 ID 목록 + 메타
        train/
            n1/AB1000001/cbct_0045.npy  # (1, H, W) float32
            n1/AB1000001/ct_0045.npy
            n1/AB1000001/mask_0045.npy
            n5/AB1000001/cbct_0045.npy  # (5, H, W) float32
            ...
        val/
            n1/AB1000002/cbct_0089.npy  # middle slice만
            ...

재실행 안전: 이미 존재하는 파일은 skip (--overwrite로 강제 재실행).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import pathlib
from typing import Dict, List, Tuple

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import zoom


# ---------------------------------------------------------------------------
# 볼륨 로드
# ---------------------------------------------------------------------------

def load_volume(subj_dir: pathlib.Path, modalities: List[str],
                hu_min: float, hu_max: float) -> Dict[str, np.ndarray]:
    """subj_dir 안의 .mha 파일을 읽어 정규화된 ndarray dict 반환.

    Returns:
        {modality: ndarray (D, H, W) float32}
    """
    result: Dict[str, np.ndarray] = {}
    mask_arr = sitk.GetArrayFromImage(
        sitk.ReadImage(str(subj_dir / "mask.mha"))
    ).astype(np.float32)
    result["mask"] = mask_arr

    for m in modalities:
        arr = sitk.GetArrayFromImage(
            sitk.ReadImage(str(subj_dir / f"{m}.mha"))
        ).astype(np.float32)
        if m == "ct":
            arr = np.clip(arr, hu_min, hu_max)
            arr = (arr - hu_min) / (hu_max - hu_min)
        elif m == "cbct":
            lo, hi = arr.min(), arr.max()
            if hi > lo:
                arr = (arr - lo) / (hi - lo)
        arr = arr * mask_arr  # apply mask
        result[m] = arr

    return result


# ---------------------------------------------------------------------------
# 슬라이스 윈도우 추출 + resize
# ---------------------------------------------------------------------------

def extract_and_resize(
    arr: np.ndarray,   # (D, H, W)
    center: int,
    n_slices: int,
    spatial_size: int,
    is_mask: bool,
) -> np.ndarray:
    """center 기준 n_slices 윈도우 추출 후 (n_slices, spatial_size, spatial_size) 반환."""
    D, H, W = arr.shape
    half = n_slices // 2
    window = np.zeros((n_slices, H, W), dtype=np.float32)
    for i, abs_idx in enumerate(range(center - half, center - half + n_slices)):
        if 0 <= abs_idx < D:
            window[i] = arr[abs_idx]

    if H == spatial_size and W == spatial_size:
        return window

    zoom_h = spatial_size / H
    zoom_w = spatial_size / W
    order = 0 if is_mask else 1  # 0=nearest, 1=bilinear
    return zoom(window, (1, zoom_h, zoom_w), order=order).astype(np.float32)


# ---------------------------------------------------------------------------
# 피험자 단위 처리 함수 (worker)
# ---------------------------------------------------------------------------

def process_subject(
    args_tuple: Tuple,
) -> int:
    """단일 피험자의 모든 n_slices × 슬라이스를 전처리해 저장.

    Returns:
        저장된 슬라이스 수.
    """
    (subj_dir, split, n_slices_list, output_root,
     modalities, spatial_size, offset, hu_min, hu_max,
     is_val, overwrite) = args_tuple

    subj_id = subj_dir.name
    saved = 0

    try:
        vol = load_volume(subj_dir, modalities, hu_min, hu_max)
    except Exception as e:
        print(f"[skip] {subj_id}: {e}")
        return 0

    D = vol["mask"].shape[0]
    valid_start = offset
    valid_end   = D - offset  # exclusive

    if valid_end <= valid_start:
        print(f"[skip] {subj_id}: depth {D} too small for offset {offset}")
        return 0

    if is_val:
        # middle slice 하나만
        center_candidates = [(valid_start + valid_end) // 2]
    else:
        center_candidates = list(range(valid_start, valid_end))

    all_mods = modalities + ["mask"]

    for n in n_slices_list:
        out_dir = pathlib.Path(output_root) / split / f"n{n}" / subj_id
        out_dir.mkdir(parents=True, exist_ok=True)

        for center in center_candidates:
            for m in all_mods:
                out_path = out_dir / f"{m}_{center:04d}.npy"
                if out_path.exists() and not overwrite:
                    continue
                is_mask = (m == "mask")
                window = extract_and_resize(
                    vol[m], center, n, spatial_size, is_mask
                )
                np.save(str(out_path), window)
                saved += 1

    return saved


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser(description="SynthRad2025 슬라이스 전처리")
    p.add_argument("--data_root", type=str, required=True,
                   help="synthRAD2025_Task2_Train / _Train_D 의 상위 디렉토리")
    p.add_argument("--output_root", type=str, required=True,
                   help="전처리 결과를 저장할 루트 디렉토리")
    p.add_argument("--n_slices", type=int, nargs="+", default=[1, 3, 5, 7, 9])
    p.add_argument("--modality", type=str, nargs="+", default=["cbct", "ct"],
                   help="저장할 모달리티 (mask는 항상 저장됨)")
    p.add_argument("--anatomy", type=str, nargs="+", default=["AB", "HN", "TH"])
    p.add_argument("--spatial_size", type=int, default=128,
                   help="저장 해상도 (default: 128)")
    p.add_argument("--offset", type=int, default=10,
                   help="볼륨 앞뒤로 제외할 슬라이스 수 (default: 10)")
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hu_min", type=float, default=-1024.0)
    p.add_argument("--hu_max", type=float, default=3071.0)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--overwrite", action="store_true",
                   help="이미 존재하는 파일도 덮어씀")
    return p.parse_args()


def collect_subjects(data_root: str, anatomy: List[str]) -> List[pathlib.Path]:
    root = pathlib.Path(data_root)
    data_dirs = [
        root / "synthRAD2025_Task2_Train"  / "Task2",
        root / "synthRAD2025_Task2_Train_D" / "Task2",
    ]
    subjects = []
    for data_dir in data_dirs:
        if not data_dir.exists():
            print(f"[warn] {data_dir} 없음, 건너뜀")
            continue
        for anat in [a.upper() for a in anatomy]:
            anat_path = data_dir / anat
            if not anat_path.exists():
                print(f"[warn] {anat_path} 없음")
                continue
            subjects.extend(sorted(p for p in anat_path.iterdir() if p.is_dir()))
    return subjects


def main():
    args = get_args()
    output_root = pathlib.Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # 1. 피험자 수집
    subjects = collect_subjects(args.data_root, args.anatomy)
    print(f"[preprocess] 총 피험자: {len(subjects)}명")

    # 2. 환자 단위 train/val 분리 (seed 고정)
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(subjects))
    n_val = max(1, int(len(subjects) * args.val_ratio))
    val_indices  = set(idx[:n_val].tolist())
    train_indices = [i for i in range(len(subjects)) if i not in val_indices]
    val_indices   = [i for i in range(len(subjects)) if i in val_indices]

    train_subjects = [subjects[i] for i in train_indices]
    val_subjects   = [subjects[i] for i in val_indices]
    print(f"[preprocess] train: {len(train_subjects)}명 | val: {len(val_subjects)}명")

    # 3. split_seed{seed}.json 저장
    split_path = output_root / f"split_seed{args.seed}.json"
    split_info = {
        "train": [s.name for s in train_subjects],
        "val":   [s.name for s in val_subjects],
        "meta": {
            "seed":         args.seed,
            "val_ratio":    args.val_ratio,
            "spatial_size": args.spatial_size,
            "offset":       args.offset,
            "n_slices":     args.n_slices,
            "modality":     args.modality,
            "anatomy":      args.anatomy,
        },
    }
    with open(split_path, "w") as f:
        json.dump(split_info, f, indent=2, ensure_ascii=False)
    print(f"[preprocess] split 정보 저장: {split_path}")

    # 4. 병렬 전처리
    tasks = []
    for subj_dir in train_subjects:
        tasks.append((
            subj_dir, "train", args.n_slices, str(output_root),
            args.modality, args.spatial_size, args.offset,
            args.hu_min, args.hu_max, False, args.overwrite,
        ))
    for subj_dir in val_subjects:
        tasks.append((
            subj_dir, "val", args.n_slices, str(output_root),
            args.modality, args.spatial_size, args.offset,
            args.hu_min, args.hu_max, True, args.overwrite,
        ))

    total_saved = 0
    if args.num_workers > 1:
        with mp.Pool(args.num_workers) as pool:
            for n in pool.imap_unordered(process_subject, tasks):
                total_saved += n
                if total_saved % 10000 == 0 and total_saved > 0:
                    print(f"  저장됨: {total_saved:,}개")
    else:
        for i, task in enumerate(tasks):
            n = process_subject(task)
            total_saved += n
            if (i + 1) % 10 == 0:
                print(f"  피험자 {i+1}/{len(tasks)} 완료 | 저장: {total_saved:,}개")

    print(f"[preprocess] 완료. 총 저장 파일: {total_saved:,}개")
    print(f"[preprocess] 출력 루트: {output_root}")


if __name__ == "__main__":
    main()

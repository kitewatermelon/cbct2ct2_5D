"""dataset.py — SynthRad2025 슬라이스 단위 데이터셋 클래스.

변경사항 (볼륨 단위 → 슬라이스 단위):
  - __getitem__이 (subject, slice_idx) 쌍으로 슬라이스 하나를 반환
  - n_slices 윈도우: 중심 슬라이스 기준 앞뒤 n//2 장 추출, 경계는 제로 패딩
  - 케이스 단위 train/val split 헬퍼(split_by_subject) 제공 → data leakage 방지
"""
from __future__ import annotations

import pathlib
from typing import Callable, Dict, List, Optional, Tuple, Union

import bisect
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from monai.transforms import (
    Compose,
    RandAdjustContrastd,
    RandAffined,
)
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def build_transforms(
    modality: List[str],
    spatial_size: Tuple[int, int] = (128, 128),
    augment: bool = False,
) -> Callable:
    """슬라이스 단위 transform.

    입력 dict의 각 키는 (n_slices, H, W) 텐서.
    resize_hw → (n_slices, tH, tW) 로 변환.
    """
    image_keys = [m for m in modality if m in ("cbct", "ct")]
    all_keys   = image_keys + ["mask"]

    def resize_hw(data: dict) -> dict:
        for key in all_keys:
            if key not in data:
                continue
            img = data[key]
            if not isinstance(img, torch.Tensor):
                img = torch.as_tensor(img)

            mode  = "nearest" if key == "mask" else "bilinear"
            align = None      if key == "mask" else False

            if img.ndim == 2:
                img = img.unsqueeze(0)          # (H,W) → (1,H,W)

            # img: (n_slices, H, W)
            resized = []
            for d in range(img.shape[0]):
                sl   = img[d].unsqueeze(0).unsqueeze(0).float()   # (1,1,H,W)
                sl_r = F.interpolate(sl, size=spatial_size,
                                     mode=mode, align_corners=align)
                resized.append(sl_r.squeeze(0).squeeze(0))        # (tH,tW)
            data[key] = torch.stack(resized, dim=0)               # (n,tH,tW)
        return data

    transforms = [resize_hw]

    if augment:
        transforms.append(
            RandAdjustContrastd(keys=image_keys, prob=0.5, gamma=(0.5, 1.5))
        )
        transforms.append(
            RandAffined(
                keys=all_keys,
                prob=0.5,
                rotate_range   =[(-0.0873, 0.0873)] * 3,
                translate_range=[(-1, 1)]            * 3,
                scale_range    =[(-0.05,  0.05)]     * 3,
                mode           =["bilinear" if k != "mask" else "nearest"
                                 for k in all_keys],
                padding_mode   ="zeros",
            )
        )

    return Compose(transforms)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SynthRad2025(Dataset):
    """SynthRad2025 슬라이스 단위 데이터셋.

    케이스(볼륨)를 모두 메모리에 로드한 뒤,
    슬라이스 인덱스 테이블을 구성해 __getitem__에서 슬라이스 하나를 반환.

    Args:
        root:        데이터 루트 경로 또는 피험자 디렉토리 리스트.
        modality:    로드할 모달리티 리스트, e.g. ["cbct", "ct"].
        anatomy:     탐색할 anatomy 목록, e.g. ["AB", "HN", "TH"].
        n_slices:    중심 슬라이스 기준 윈도우 크기 (홀수 권장).
                     경계는 제로 패딩.
        transform:   슬라이스 단위 transform (dict → dict).
        hu_min:      CT HU 클리핑 하한.
        hu_max:      CT HU 클리핑 상한.
        apply_mask:  True면 마스크 외부를 0으로 채움.

    __getitem__ 반환:
        {
            "subj_id"  : str,
            "slice_idx": int,                       # 볼륨 내 원본 슬라이스 인덱스
            "cbct"     : FloatTensor (n_slices, H, W),
            "ct"       : FloatTensor (n_slices, H, W),   # modality에 포함 시
            "mask"     : FloatTensor (n_slices, H, W),
        }
    """

    def __init__(
        self,
        root: Union[str, pathlib.Path, List[Union[str, pathlib.Path]]],
        modality: List[str],
        anatomy: Optional[List[str]] = None,
        n_slices: int = 1,
        transform: Optional[Callable] = None,
        hu_min: float = -1024.0,
        hu_max: float = 3071.0,
        apply_mask: bool = True,
        slice_margin: int = 10,
        cache_subjects: int = 32,
    ) -> None:
        """
        Args:
            slice_margin:    볼륨 앞뒤로 제외할 슬라이스 수. 기본 10.
            cache_subjects:  worker당 메모리에 보관할 최대 볼륨 수. 0 = 캐시 비활성화.
        """
        self.modality       = modality
        self.n_slices       = n_slices
        self.transform      = transform
        self.hu_min         = hu_min
        self.hu_max         = hu_max
        self.apply_mask     = apply_mask
        self.slice_margin   = slice_margin
        self._cache_max     = cache_subjects
        self._volume_cache: Dict[int, Dict[str, torch.Tensor]] = {}

        subject_dirs = self._resolve_subject_dirs(root, anatomy)
        self.subject_dirs, skipped = self._validate(subject_dirs)

        if skipped:
            print(f"[SynthRad2025] {skipped}개 피험자 건너뜀 (파일 누락)")

        # subject별 누적 슬라이스 수 (bisect용) — 인덱스 테이블 없이 O(1) 조회
        # _subject_offsets[i] = subject 0~i-1 의 유효 슬라이스 합계
        m = self.slice_margin
        self._subject_offsets: List[int] = [0]
        self._subject_depths:  List[int] = []
        for subj_idx, d in enumerate(self.subject_dirs):
            depth = self._read_depth(d / "mask.mha")
            n_valid = max(0, depth - 2 * m)
            self._subject_depths.append(depth)
            self._subject_offsets.append(self._subject_offsets[-1] + n_valid)

        self._middle_only   = False
        self._middle_slices = None
        total_slices = self._subject_offsets[-1]
        print(f"[SynthRad2025] 유효 피험자: {len(self.subject_dirs)}개 | "
              f"총 슬라이스: {total_slices}개 "
              f"(앞뒤 {self.slice_margin}장 제외) | "
              f"n_slices: {self.n_slices} | 모달리티: {self.modality}")

    # ------------------------------------------------------------------
    # 내부 유틸
    # ------------------------------------------------------------------

    def _resolve_subject_dirs(self, root, anatomy):
        # 리스트면 각 경로에 대해 재귀 호출 후 합산
        if isinstance(root, list):
            dirs = []
            for r in root:
                dirs.extend(self._resolve_subject_dirs(r, anatomy))
            return dirs
        root = pathlib.Path(root)
        if anatomy:
            dirs = []
            for anat in [a.upper() for a in anatomy]:
                anat_path = root / anat
                if anat_path.exists():
                    dirs.extend(sorted(p for p in anat_path.iterdir() if p.is_dir()))
                else:
                    print(f"[SynthRad2025] 경고: {anat_path} 없음")
            return dirs
        return sorted(p for p in root.iterdir() if p.is_dir())

    def _validate(self, dirs):
        valid, skipped = [], 0
        for d in dirs:
            has_mod  = all((d / f"{m}.mha").exists() for m in self.modality)
            has_mask = (d / "mask.mha").exists()
            if has_mod and has_mask:
                valid.append(d)
            else:
                skipped += 1
        return valid, skipped

    @staticmethod
    def _read_depth(path: pathlib.Path) -> int:
        """볼륨 전체를 읽지 않고 depth(슬라이스 수)만 반환."""
        reader = sitk.ImageFileReader()
        reader.SetFileName(str(path))
        reader.ReadImageInformation()
        # SimpleITK: (W, H, D) 순서
        return reader.GetSize()[2]

    def _load_mha(self, path, modality: str) -> torch.Tensor:
        arr = sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(np.float32)
        if modality == "ct":
            arr = np.clip(arr, self.hu_min, self.hu_max)
            arr = (arr - self.hu_min) / (self.hu_max - self.hu_min)
        elif modality == "cbct":
            lo, hi = arr.min(), arr.max()
            if hi > lo:
                arr = (arr - lo) / (hi - lo)
        else:  # mask
            pass
        return torch.from_numpy(arr)  # (D, H, W)

    def _load_volume(self, d: pathlib.Path) -> Dict[str, torch.Tensor]:
        vol: Dict[str, torch.Tensor] = {}
        vol["mask"] = self._load_mha(d / "mask.mha", "mask")
        for m in self.modality:
            img = self._load_mha(d / f"{m}.mha", m)
            vol[m] = img * vol["mask"] if self.apply_mask else img
        return vol

    def _get_volume(self, subj_idx: int) -> Dict[str, torch.Tensor]:
        if self._cache_max == 0:
            return self._load_volume(self.subject_dirs[subj_idx])
        if subj_idx not in self._volume_cache:
            if len(self._volume_cache) >= self._cache_max:
                self._volume_cache.pop(next(iter(self._volume_cache)))
            self._volume_cache[subj_idx] = self._load_volume(self.subject_dirs[subj_idx])
        return self._volume_cache[subj_idx]

    def _extract_window(
        self,
        vol: torch.Tensor,   # (D, H, W)
        center: int,
        n: int,
    ) -> torch.Tensor:
        """center 슬라이스 기준 n장 윈도우 추출. 경계는 제로 패딩."""
        D, H, W = vol.shape
        half    = n // 2
        result  = torch.zeros(n, H, W, dtype=vol.dtype)
        for i, abs_idx in enumerate(range(center - half, center - half + n)):
            if 0 <= abs_idx < D:
                result[i] = vol[abs_idx]
            # else: 이미 0으로 초기화됨 (제로 패딩)
        return result  # (n, H, W)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._subject_offsets[-1]

    def __getitem__(self, idx: int) -> dict:
        # O(log N): bisect로 subject 찾기
        subj_idx = bisect.bisect_right(self._subject_offsets, idx) - 1
        if getattr(self, "_middle_only", False):
            # val: 케이스당 중간 슬라이스 1장
            center = self._middle_slices[subj_idx]
        else:
            local_idx = idx - self._subject_offsets[subj_idx]
            center    = self.slice_margin + local_idx
        vol = self._get_volume(subj_idx)

        data: dict = {
            "subj_id"  : self.subject_dirs[subj_idx].name,
            "slice_idx": center,
            "mask"     : self._extract_window(vol["mask"], center, self.n_slices),
        }
        for m in self.modality:
            data[m] = self._extract_window(vol[m], center, self.n_slices)

        if self.transform:
            data = self.transform(data)
        return data

    # ------------------------------------------------------------------
    # 케이스 단위 split 헬퍼
    # ------------------------------------------------------------------

    def split_by_subject(
        self,
        val_ratio: float = 0.2,
        seed: int = 42,
        val_middle_only: bool = False,
    ) -> Tuple["SynthRad2025", "SynthRad2025"]:
        """케이스 단위로 train/val 분리 → data leakage 방지.

        Args:
            val_middle_only: True면 val에서 케이스당 중간 슬라이스 1장만 사용.
                             평가 속도 향상에 유용.

        Returns:
            train_dataset, val_dataset  (각각 SynthRad2025 인스턴스)
        """
        rng = np.random.default_rng(seed)
        n   = len(self.subject_dirs)
        idx = rng.permutation(n)

        n_val  = max(1, int(n * val_ratio))
        val_i  = set(idx[:n_val].tolist())
        train_i = [i for i in range(n) if i not in val_i]
        val_i   = [i for i in range(n) if i in val_i]

        def _subset(indices: List[int], middle_only: bool = False) -> "SynthRad2025":
            ds = SynthRad2025.__new__(SynthRad2025)
            ds.modality       = self.modality
            ds.n_slices       = self.n_slices
            ds.transform      = self.transform
            ds.hu_min         = self.hu_min
            ds.hu_max         = self.hu_max
            ds.apply_mask     = self.apply_mask
            ds.slice_margin   = self.slice_margin
            ds.subject_dirs   = [self.subject_dirs[i] for i in indices]
            m = self.slice_margin
            ds._subject_depths = [self._subject_depths[i] for i in indices]
            ds._middle_only    = middle_only
            if middle_only:
                # val: 케이스당 중간 슬라이스 1장 → offset 1씩 증가
                ds._subject_offsets = list(range(len(indices) + 1))
                ds._middle_slices   = [
                    m + max(0, self._subject_depths[i] - 2 * m) // 2
                    for i in indices
                ]
            else:
                ds._subject_offsets = [0]
                for d in ds._subject_depths:
                    ds._subject_offsets.append(
                        ds._subject_offsets[-1] + max(0, d - 2 * m)
                    )
                ds._middle_slices = None
            return ds

        return _subset(train_i, middle_only=False), _subset(val_i, middle_only=val_middle_only)


# ---------------------------------------------------------------------------
# split_dataset 독립 함수 (기존 코드와의 호환용)
# ---------------------------------------------------------------------------

def split_dataset(
    root: Union[str, pathlib.Path],
    modality: List[str],
    anatomy: Optional[List[str]] = None,
    n_slices: int = 1,
    val_ratio: float = 0.2,
    seed: int = 42,
    train_transform: Optional[Callable] = None,
    val_transform: Optional[Callable] = None,
    hu_min: float = -1024.0,
    hu_max: float = 3071.0,
    apply_mask: bool = True,
    val_middle_only: bool = True,
    slice_margin: int = 10,
) -> Tuple[SynthRad2025, SynthRad2025]:
    """케이스 단위 train/val split 후 각각 다른 transform 적용.

    Args:
        val_middle_only: True(기본값)면 val은 케이스당 중간 슬라이스 1장만 사용.
                         전체 슬라이스 평가가 필요하면 False로 변경.

    Usage:
        train_ds, val_ds = split_dataset(
            root     = "/data/prof2/mai/s2025/dataset/synthRAD2025_Task2_Train",
            modality = ["cbct", "ct"],
            anatomy  = ["AB", "HN", "TH"],
            n_slices = 9,
            val_ratio = 0.2,
            seed      = 42,
            train_transform = build_transforms(["cbct","ct"], augment=True),
            val_transform   = build_transforms(["cbct","ct"], augment=False),
        )
    """
    # transform 없이 전체 로드 (케이스 split 용)
    full_ds = SynthRad2025(
        root=root, modality=modality, anatomy=anatomy,
        n_slices=n_slices, transform=None,
        hu_min=hu_min, hu_max=hu_max,
        apply_mask=apply_mask,
        slice_margin=slice_margin,
    )

    train_ds, val_ds = full_ds.split_by_subject(
        val_ratio=val_ratio, seed=seed, val_middle_only=val_middle_only
    )
    train_ds.transform = train_transform
    val_ds.transform   = val_transform

    print(f"[split_dataset] train: {len(train_ds)}슬라이스 "
          f"({len(train_ds.subject_dirs)}케이스) | "
          f"val: {len(val_ds)}슬라이스 "
          f"({len(val_ds.subject_dirs)}케이스, "
          f"{'middle slice only' if val_middle_only else 'all slices'})")
    return train_ds, val_ds


# ---------------------------------------------------------------------------
# 테스트
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 두 데이터셋 모두 Task2/ 하위에 AB/HN/TH 구조
    ROOT = [
        "/data/prof2/mai/s2025/dataset/synthRAD2025_Task2_Train/Task2",
        "/data/prof2/mai/s2025/dataset/synthRAD2025_Task2_Train_D/Task2",
    ]

    train_ds, val_ds = split_dataset(
        root             = ROOT,
        modality         = ["cbct", "ct"],
        anatomy          = ["AB", "HN", "TH"],
        n_slices         = 9,
        val_ratio        = 0.2,
        seed             = 42,
        val_middle_only  = True,   # val은 케이스당 중간 슬라이스 1장만
        train_transform  = build_transforms(["cbct", "ct"],
                                            spatial_size=(128, 128),
                                            augment=True),
        val_transform    = build_transforms(["cbct", "ct"],
                                            spatial_size=(128, 128),
                                            augment=False),

    )

    sample = train_ds[0]
    print("subj_id  :", sample["subj_id"])
    print("slice_idx:", sample["slice_idx"])
    print("cbct     :", sample["cbct"].shape,
          sample["cbct"].min().item(), sample["cbct"].max().item())
    print("ct       :", sample["ct"].shape,
          sample["ct"].min().item(), sample["ct"].max().item())
    print("mask     :", sample["mask"].shape)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    batch = next(iter(train_loader))
    print("batch cbct:", batch["cbct"].shape)   # (32, 9, 128, 128)
    print("batch ct  :", batch["ct"].shape)     # (32, 9, 128, 128)
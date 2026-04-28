"""preprocessed_dataset.py — 전처리된 슬라이스 파일 기반 Dataset.

data/preprocess.py 로 미리 저장된 .npy 슬라이스를 읽어
__getitem__마다 파일 하나만 np.load → DataLoader stall 완전 제거.

사용 예:
    from data.preprocessed_dataset import PreprocessedDataset, build_preprocessed_transforms

    train_ds = PreprocessedDataset(
        preprocessed_root = "/data/prof2/mai/s2025/dataset/preprocessed",
        split     = "train",
        n_slices  = 9,
        modality  = ["cbct", "ct"],
        transform = build_preprocessed_transforms(augment=True),
    )
    val_ds = PreprocessedDataset(
        preprocessed_root = "/data/prof2/mai/s2025/dataset/preprocessed",
        split     = "val",
        n_slices  = 9,
        modality  = ["cbct", "ct"],
        transform = build_preprocessed_transforms(augment=False),
    )
"""
from __future__ import annotations

import pathlib
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from monai.transforms import Compose, RandAdjustContrastd, RandAffined
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Transforms (augment only — resize는 이미 전처리 시 완료됨)
# ---------------------------------------------------------------------------

def build_preprocessed_transforms(
    modality: List[str] = ("cbct", "ct"),
    augment: bool = False,
) -> Optional[Callable]:
    """전처리 완료 슬라이스용 transform (resize 없음, augment만)."""
    image_keys = [m for m in modality if m in ("cbct", "ct")]
    all_keys   = image_keys + ["mask"]

    if not augment:
        return None

    return Compose([
        RandAdjustContrastd(keys=image_keys, prob=0.5, gamma=(0.5, 1.5)),
        RandAffined(
            keys=all_keys,
            prob=0.5,
            rotate_range   =[(-0.0873, 0.0873)] * 3,
            translate_range=[(-1, 1)]            * 3,
            scale_range    =[(-0.05,  0.05)]     * 3,
            mode           =["bilinear" if k != "mask" else "nearest"
                             for k in all_keys],
            padding_mode   ="zeros",
        ),
    ])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PreprocessedDataset(Dataset):
    """전처리된 슬라이스 .npy 파일 기반 Dataset.

    Args:
        preprocessed_root: data/preprocess.py 의 --output_root 경로.
        split:             "train" | "val"
        n_slices:          1 | 3 | 5 | 7 | 9
        modality:          로드할 모달리티 리스트, e.g. ["cbct"] or ["cbct", "ct"].
                           mask는 항상 로드됨.
        transform:         슬라이스 단위 augmentation (dict → dict).
                           build_preprocessed_transforms() 사용 권장.

    __getitem__ 반환:
        {
            "subj_id"  : str,
            "slice_idx": int,
            "cbct"     : FloatTensor (n_slices, H, W),   # modality에 포함 시
            "ct"       : FloatTensor (n_slices, H, W),   # modality에 포함 시
            "mask"     : FloatTensor (n_slices, H, W),
        }
    """

    def __init__(
        self,
        preprocessed_root: str | pathlib.Path,
        split: str,
        n_slices: int,
        modality: List[str],
        transform: Optional[Callable] = None,
    ) -> None:
        self.preprocessed_root = pathlib.Path(preprocessed_root)
        self.split    = split
        self.n_slices = n_slices
        self.modality = modality
        self.transform = transform

        split_dir = self.preprocessed_root / split / f"n{n_slices}"
        if not split_dir.exists():
            raise FileNotFoundError(
                f"{split_dir} 없음. data/preprocess.py 를 먼저 실행하세요."
            )

        # cbct 파일 기준으로 샘플 목록 구성 (modality 중 하나를 anchor로 사용)
        anchor = modality[0] if modality else "cbct"
        self._samples: List[Tuple[pathlib.Path, str, int]] = []
        # (subj_dir, subj_id, center_idx)

        for subj_dir in sorted(split_dir.iterdir()):
            if not subj_dir.is_dir():
                continue
            subj_id = subj_dir.name
            for f in sorted(subj_dir.glob(f"{anchor}_*.npy")):
                # 파일명: cbct_0045.npy → center=45
                try:
                    center = int(f.stem.split("_")[1])
                except (IndexError, ValueError):
                    continue
                self._samples.append((subj_dir, subj_id, center))

        if len(self._samples) == 0:
            raise RuntimeError(
                f"{split_dir} 에서 {anchor}_*.npy 파일을 찾을 수 없습니다."
            )

        n_subjects = len({s[1] for s in self._samples})
        print(f"[PreprocessedDataset] split={split} n_slices={n_slices} | "
              f"피험자: {n_subjects}명 | 슬라이스: {len(self._samples):,}개")

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict:
        subj_dir, subj_id, center = self._samples[idx]

        data: dict = {
            "subj_id"  : subj_id,
            "slice_idx": center,
        }

        # mask 항상 로드
        mask_path = subj_dir / f"mask_{center:04d}.npy"
        data["mask"] = torch.from_numpy(np.load(str(mask_path)))

        # 요청 modality 로드
        for m in self.modality:
            npy_path = subj_dir / f"{m}_{center:04d}.npy"
            data[m] = torch.from_numpy(np.load(str(npy_path)))

        if self.transform:
            data = self.transform(data)

        return data

    @property
    def subject_dirs(self) -> List[str]:
        """stage1_full.py 의 len(ds.subject_dirs) 호환용."""
        return sorted({s[1] for s in self._samples})

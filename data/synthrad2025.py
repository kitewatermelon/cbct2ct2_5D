"""dataset.py — SynthRad2025 단일 데이터셋 클래스."""
from __future__ import annotations
 
import pathlib
from typing import Callable, List, Optional, Tuple, Union  # Tuple/Optional: build_transforms에서 사용
 
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
 
def build_transforms(
    modality: List[str],
    spatial_size: Tuple[int, int] = (128, 128),
    augment: bool = False,
) -> Callable:
    """논문 전처리 파이프라인에 맞는 transform 반환.
 
    전처리 순서 (논문 §2.x 기준):
      1. HU 클리핑 [-1024, 3071]  → Dataset._load() 에서 처리
      2. Body mask 적용           → Dataset.__getitem__() 에서 처리
      3. [0, 1] 정규화            → Dataset._load() 에서 처리
      4. Resize H×W → spatial_size (기본 128×128, D 축 유지)
      5. (train only) RandAdjustContrast: gamma ∈ [0.5, 1.5]
      6. (train only) RandAffine: ±5° rotation / ±1 px translation / ±5% scale
 
    Args:
        modality:     이미지 키 리스트, e.g. ["cbct", "ct"].
        spatial_size: 리사이즈 목표 (H, W). 기본 (128, 128).
        augment:      True면 augmentation 추가 (train 전용).
 
    Returns:
        dict → dict transform (Compose).
    """
    image_keys = [m for m in modality if m in ("cbct", "ct")]
    all_keys   = image_keys + ["mask"]
 
    def resize_hw(data: dict) -> dict:
        """볼륨 (D, H, W)의 H, W를 spatial_size로 리사이즈.
        이미지: trilinear, 마스크: nearest.
        """
        for key in all_keys:
            if key not in data:
                continue
            img = data[key]
            if not isinstance(img, torch.Tensor):
                img = torch.as_tensor(img)
            mode       = "nearest"         if key == "mask" else "trilinear"
            align      = None              if key == "mask" else False
            img_5d     = img.unsqueeze(0).unsqueeze(0).float()   # (1,1,D,H,W)
            target     = (img.shape[0], *spatial_size)           # (D, tH, tW)
            resized    = F.interpolate(img_5d, size=target,
                                       mode=mode, align_corners=align)
            data[key]  = resized.squeeze(0).squeeze(0)
        return data
 
    transforms = [resize_hw]
 
    if augment:
        # RandAdjustContrastd: 이미지 키에만 적용, mask 제외
        transforms.append(
            RandAdjustContrastd(
                keys=image_keys,
                prob=0.5,
                gamma=(0.5, 1.5),
            )
        )
        # RandAffined: 이미지 + 마스크 동시 적용 (공간 일관성 유지)
        # ±5° ≈ ±0.0873 rad / ±1 px translation / ±5% scale
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
 
 
class SynthRad2025(Dataset):
    """SynthRad2025 피험자별 3D 볼륨 데이터셋.
 
    Args:
        root:        데이터 루트 경로 (anatomy 하위 디렉토리 포함) 또는
                     피험자 디렉토리 경로 리스트.
        modality:    로드할 모달리티 리스트, e.g. ["cbct", "ct"].
        anatomy:     root 경로 방식일 때 탐색할 anatomy 목록,
                     e.g. ["AB", "HN", "TH"]. None이면 전체 하위 디렉토리 탐색.
        transform:   호출 가능한 transform (dict → dict).
        hu_min:      CT HU 클리핑 하한.
        hu_max:      CT HU 클리핑 상한.
        apply_mask:  True면 마스크 외부 영역을 0으로 채움.
 
    각 아이템:
        {
            "subj_id": str,
            "cbct"  : FloatTensor (D, H, W),   # modality에 포함된 경우
            "ct"    : FloatTensor (D, H, W),   # modality에 포함된 경우
            "mask"  : FloatTensor (D, H, W),
        }
    """
 
    def __init__(
        self,
        root: Union[str, pathlib.Path, List[Union[str, pathlib.Path]]],
        modality: List[str],
        anatomy: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
        hu_min: float = -1024.0,
        hu_max: float = 3071.0,
        apply_mask: bool = True,
    ) -> None:
        self.modality = modality
        self.transform = transform
        self.hu_min = hu_min
        self.hu_max = hu_max
        self.apply_mask = apply_mask
 
        subject_dirs = self._resolve_subject_dirs(root, anatomy)
        self.subject_dirs, skipped = self._validate(subject_dirs)
 
        if skipped:
            print(f"[SynthRad2025] {skipped}개 피험자 건너뜀 (파일 누락)")
        print(f"[SynthRad2025] 유효 피험자: {len(self.subject_dirs)}개 | "
              f"모달리티: {self.modality}")
  
    def _resolve_subject_dirs(
        self,
        root: Union[str, pathlib.Path, List],
        anatomy: Optional[List[str]],
    ) -> List[pathlib.Path]:
        """root가 리스트면 그대로 반환, 경로면 anatomy 하위 디렉토리 탐색."""
        if isinstance(root, list):
            return [pathlib.Path(d) for d in root]
 
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
 
    def _validate(
        self, dirs: List[pathlib.Path]
    ) -> Tuple[List[pathlib.Path], int]:
        """필요한 파일이 모두 있는 디렉토리만 걸러냄."""
        valid, skipped = [], 0
        for d in dirs:
            has_mod  = all((d / f"{m}.mha").exists() for m in self.modality)
            has_mask = (d / "mask.mha").exists()
            if has_mod and has_mask:
                valid.append(d)
            else:
                skipped += 1
        return valid, skipped

    def _load(self, path: Union[str, pathlib.Path], modality: str) -> torch.Tensor:
        """mha → float32 Tensor.
        CT:   HU 클리핑 [-1024, 3071] 후 [0, 1] 정규화.
        CBCT: 클리핑 없이 volume 단위 min-max 정규화.
        """
        arr = sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(np.float32)
        if modality == "ct":
            arr = np.clip(arr, self.hu_min, self.hu_max)
            arr = (arr - self.hu_min) / (self.hu_max - self.hu_min)
        else:  # cbct
            lo, hi = arr.min(), arr.max()
            if hi > lo:
                arr = (arr - lo) / (hi - lo)
        return torch.from_numpy(arr)
 
    def _load_mask(self, path: Union[str, pathlib.Path]) -> torch.Tensor:
        arr = sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(np.float32)
        return torch.from_numpy(arr)

    def __len__(self) -> int:
        return len(self.subject_dirs)
 
    def __getitem__(self, idx: int) -> dict:
        d = self.subject_dirs[idx]
        mask = self._load_mask(d / "mask.mha")
 
        data: dict = {"subj_id": d.name, "mask": mask}
        for m in self.modality:
            img = self._load(d / f"{m}.mha", modality=m)
            data[m] = img * mask if self.apply_mask else img
 
        if self.transform:
            data = self.transform(data)
        return data

if __name__ == "__main__":
    ROOT = "/home/dministrator/s2025"
    train_ds = SynthRad2025(
        root=f"{ROOT}/dataset/train/n9",
        modality= ["cbct", "ct"],
        anatomy=["AB", "HN", "TH"],
        transform=build_transforms( ["cbct", "ct"], spatial_size=(128, 128), augment=True),
    )
    val_ds = SynthRad2025(
        root=f"{ROOT}/dataset/val/n9",
        modality= ["cbct"],
        anatomy=["AB", "HN", "TH"],
        transform=build_transforms(["cbct"], spatial_size=(128, 128), augment=False),
    )

    sample = train_ds[0]
    print(sample["subj_id"])
    print("cbct:", sample["cbct"].shape, sample["cbct"].min().item(), sample["cbct"].max().item())
    print("ct:", sample["ct"].shape, sample["ct"].min().item(), sample["ct"].max().item())
    print("mask:", sample["mask"].shape)

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True,  num_workers=0, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=2, shuffle=False, num_workers=0)
    batch = next(iter(train_loader))
    print("batch cbct:", batch["cbct"].shape)
"""HU conversion and tissue classification utilities shared across eval scripts."""
from __future__ import annotations
import numpy as np
import torch

HU_MIN = -1024.0
HU_MAX =  3071.0

TISSUES: dict[str, tuple[float | None, float | None]] = {
    "Air"         : (None, -700),
    "Lung"        : (-700, -200),
    "Fat"         : (-200,  -50),
    "Soft tissue" : ( -50,  150),
    "Bone"        : ( 150, None),
}
TISSUE_ORDER = list(TISSUES.keys())


def to_hu(x):
    """[0, 1] normalized → HU values. Accepts numpy arrays or torch tensors."""
    return x * (HU_MAX - HU_MIN) + HU_MIN


def tissue_mask(hu, lo: float | None, hi: float | None):
    """Boolean mask for pixels within [lo, hi) HU range. Numpy or tensor input."""
    if isinstance(hu, np.ndarray):
        mask = np.ones(hu.shape, dtype=bool)
        if lo is not None:
            mask &= hu >= lo
        if hi is not None:
            mask &= hu < hi
    else:
        mask = torch.ones_like(hu, dtype=torch.bool)
        if lo is not None:
            mask &= hu >= lo
        if hi is not None:
            mask &= hu < hi
    return mask


def classify_tissue(hu: np.ndarray) -> np.ndarray:
    """Assign tissue index (0–4) to each pixel. Returns int8 array."""
    out = np.full(hu.shape, -1, dtype=np.int8)
    for idx, (lo, hi) in enumerate(TISSUES.values()):
        out[tissue_mask(hu, lo, hi)] = idx
    return out


def _hu_errors(pred: np.ndarray, gt: np.ndarray) -> dict:
    """MAE, MAE std, bias를 반환 (RMSE 제외)."""
    err  = pred - gt
    abse = np.abs(err)
    return {
        "MAE"     : float(np.mean(abse)),
        "MAE_std" : float(np.std(abse)),
        "bias"    : float(np.mean(err)),
    }
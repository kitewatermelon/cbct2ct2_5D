"""SimpleITK .mha read/write helpers shared across eval scripts."""
from __future__ import annotations
import pathlib
import numpy as np
import SimpleITK as sitk


def save_mha(arr: np.ndarray, path: pathlib.Path) -> None:
    """Save (H, W) float32 array as .mha file."""
    sitk.WriteImage(sitk.GetImageFromArray(arr.astype(np.float32)), str(path))


def load_mha(path: pathlib.Path) -> np.ndarray:
    """Load .mha file and return (H, W) float32 numpy array."""
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(np.float32)

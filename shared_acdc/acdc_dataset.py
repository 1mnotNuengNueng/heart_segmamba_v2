"""
Utilities for loading the ACDC dataset and applying the paper-based
cardiac MRI augmentation.

Expected dependencies:
- numpy
- nibabel
- optionally torch for the Dataset wrapper
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple
import re

import numpy as np

try:
    import nibabel as nib
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "This module requires nibabel. Install it with `pip install nibabel`."
    ) from exc

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover
    torch = None

    class Dataset:  # type: ignore[override]
        """Fallback base class when torch is not installed."""

        pass

from shared_acdc.cardiac_mri_paper_augmentation import (
    CardiacMRIPaperAugmentor,
    PaperAugmentationParams,
    infer_acdc_phase,
)


def load_nifti_array(path: str | Path, dtype: np.dtype = np.float32) -> np.ndarray:
    """Load a NIfTI file into a numpy array."""
    return np.asarray(nib.load(str(path)).get_fdata(), dtype=dtype)


def zscore_normalize(image: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalize a slice or volume with z-score normalization."""
    mean = float(image.mean())
    std = float(image.std())
    return ((image - mean) / max(std, eps)).astype(np.float32)


def minmax_normalize(image: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalize a slice or volume into [0, 1]."""
    min_value = float(image.min())
    max_value = float(image.max())
    return ((image - min_value) / max(max_value - min_value, eps)).astype(np.float32)


def extract_frame_number(path: str | Path) -> Optional[int]:
    """Extract the ACDC frame number from a filename."""
    match = re.search(r"_frame(\d+)", Path(path).stem)
    if match is None:
        return None
    return int(match.group(1))


@dataclass
class ACDCCase:
    patient_id: str
    image_path: Path
    mask_path: Path
    info_cfg_path: Path
    phase: Optional[str]
    frame_number: Optional[int]


@dataclass
class ACDCSliceSample:
    patient_id: str
    phase: Optional[str]
    frame_number: Optional[int]
    slice_index: int
    image_path: Path
    mask_path: Path


def collect_acdc_cases(split_dir: str | Path) -> List[ACDCCase]:
    """
    Collect ED/ES labeled cases from an ACDC split directory.

    Example split_dir:
        ACDC/ACDC/database/training
    """
    split_dir = Path(split_dir)
    cases: List[ACDCCase] = []

    for patient_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
        info_cfg_path = patient_dir / "Info.cfg"
        if not info_cfg_path.exists():
            continue

        for image_path in sorted(patient_dir.glob("*_frame*.nii.gz")):
            if image_path.name.endswith("_gt.nii.gz"):
                continue

            mask_path = image_path.with_name(image_path.name.replace(".nii.gz", "_gt.nii.gz"))
            if not mask_path.exists():
                continue

            phase = infer_acdc_phase(image_path=image_path, info_cfg_path=info_cfg_path)
            if phase not in {"ED", "ES"}:
                continue

            cases.append(
                ACDCCase(
                    patient_id=patient_dir.name,
                    image_path=image_path,
                    mask_path=mask_path,
                    info_cfg_path=info_cfg_path,
                    phase=phase,
                    frame_number=extract_frame_number(image_path),
                )
            )

    return cases


def expand_cases_to_slices(
    cases: Sequence[ACDCCase],
    drop_empty_slices: bool = True,
) -> List[ACDCSliceSample]:
    """Expand case-level entries into slice-level samples."""
    samples: List[ACDCSliceSample] = []

    for case in cases:
        mask_volume = load_nifti_array(case.mask_path, dtype=np.int16)
        if mask_volume.ndim != 3:
            raise ValueError(f"Expected 3D volume for {case.mask_path}, got shape {mask_volume.shape}")

        num_slices = mask_volume.shape[2]
        for slice_index in range(num_slices):
            if drop_empty_slices and not np.any(mask_volume[:, :, slice_index] > 0):
                continue

            samples.append(
                ACDCSliceSample(
                    patient_id=case.patient_id,
                    phase=case.phase,
                    frame_number=case.frame_number,
                    slice_index=slice_index,
                    image_path=case.image_path,
                    mask_path=case.mask_path,
                )
            )

    return samples


class ACDCSliceDataset(Dataset):
    """
    PyTorch-style dataset for 2D ACDC slices.

    Each item returns:
    - image: shape [1, H, W] if torch is installed, else numpy [1, H, W]
    - mask: shape [H, W]
    - meta: dict with patient/frame/slice/phase/statistics
    """

    def __init__(
        self,
        split_dir: str | Path,
        augment: bool = False,
        augmentation_params: Optional[PaperAugmentationParams] = None,
        normalization: Optional[str] = "zscore",
        drop_empty_slices: bool = True,
        image_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        mask_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.split_dir = Path(split_dir)
        self.cases = collect_acdc_cases(self.split_dir)
        self.samples = expand_cases_to_slices(self.cases, drop_empty_slices=drop_empty_slices)
        self.augment = augment
        self.normalization = normalization
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.augmentor = (
            CardiacMRIPaperAugmentor(params=augmentation_params, seed=seed) if augment else None
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _normalize(self, image: np.ndarray) -> np.ndarray:
        if self.normalization is None:
            return image.astype(np.float32)
        if self.normalization == "zscore":
            return zscore_normalize(image)
        if self.normalization == "minmax":
            return minmax_normalize(image)
        raise ValueError(f"Unsupported normalization mode: {self.normalization}")

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image_volume = load_nifti_array(sample.image_path, dtype=np.float32)
        mask_volume = load_nifti_array(sample.mask_path, dtype=np.int16)

        image = image_volume[:, :, sample.slice_index]
        mask = mask_volume[:, :, sample.slice_index]

        stats: Dict[str, float] = {}
        if self.augment and self.augmentor is not None:
            image, mask, stats = self.augmentor.augment(image=image, mask=mask, phase=sample.phase)

        image = self._normalize(image)

        if self.image_transform is not None:
            image = self.image_transform(image)
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)

        image = np.expand_dims(image.astype(np.float32), axis=0)
        mask = mask.astype(np.int64)

        meta = {
            "patient_id": sample.patient_id,
            "phase": sample.phase,
            "frame_number": sample.frame_number,
            "slice_index": sample.slice_index,
            "image_path": str(sample.image_path),
            "mask_path": str(sample.mask_path),
            **stats,
        }

        if torch is not None:
            return (
                torch.from_numpy(image),
                torch.from_numpy(mask),
                meta,
            )
        return image, mask, meta


__all__ = [
    "ACDCCase",
    "ACDCSliceDataset",
    "ACDCSliceSample",
    "collect_acdc_cases",
    "expand_cases_to_slices",
    "extract_frame_number",
    "load_nifti_array",
    "minmax_normalize",
    "zscore_normalize",
]

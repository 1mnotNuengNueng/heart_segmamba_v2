from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import nibabel as nib
import numpy as np
from scipy import ndimage

from shared_acdc.acdc_dataset import ACDCCase, collect_acdc_cases
from shared_acdc.export_acdc_to_nnunet import export_acdc_to_nnunet_raw
from .settings import (
    ACDC_TEST_DIR,
    ACDC_TRAIN_DIR,
    DEFAULT_TARGET_SHAPE,
    DEFAULT_VAL_RATIO,
    NNUNET_RAW_DIR,
    SHARED_PREPROCESSED_DIR,
)


def _resize_volume(volume: np.ndarray, output_shape: Sequence[int], order: int) -> np.ndarray:
    zoom = [o / i for o, i in zip(output_shape, volume.shape)]
    return ndimage.zoom(volume, zoom=zoom, order=order)


def _normalize(image: np.ndarray, mode: str) -> np.ndarray:
    if mode == "zscore":
        mean = float(image.mean())
        std = float(image.std())
        return ((image - mean) / max(std, 1e-8)).astype(np.float32)
    if mode == "minmax":
        min_value = float(image.min())
        max_value = float(image.max())
        return ((image - min_value) / max(max_value - min_value, 1e-8)).astype(np.float32)
    if mode == "none":
        return image.astype(np.float32)
    raise ValueError(f"Unsupported normalization mode: {mode}")


def _patient_splits(train_cases: Sequence[ACDCCase], val_ratio: float) -> Tuple[List[str], List[str]]:
    patient_ids = sorted({case.patient_id for case in train_cases})
    val_count = max(1, int(round(len(patient_ids) * val_ratio)))
    val_patients = patient_ids[-val_count:]
    train_patients = patient_ids[:-val_count]
    return train_patients, val_patients


def _save_case(case: ACDCCase, destination: Path, target_shape: Sequence[int], normalization: str) -> str:
    destination.mkdir(parents=True, exist_ok=True)

    image_nii = nib.load(str(case.image_path))
    mask_nii = nib.load(str(case.mask_path))

    image = np.asarray(image_nii.get_fdata(), dtype=np.float32)
    mask = np.asarray(mask_nii.get_fdata(), dtype=np.int16)

    image = np.transpose(image, (2, 0, 1))
    mask = np.transpose(mask, (2, 0, 1))

    resized_image = _resize_volume(image, target_shape, order=1)
    resized_mask = _resize_volume(mask, target_shape, order=0).astype(np.int16)
    resized_image = _normalize(resized_image, normalization)

    case_id = f"{case.patient_id}_{case.phase.lower()}_frame{case.frame_number:02d}"
    np.savez_compressed(
        destination / f"{case_id}.npz",
        image=resized_image[None].astype(np.float32),
        label=resized_mask.astype(np.int16),
    )

    metadata = {
        "case_id": case_id,
        "patient_id": case.patient_id,
        "phase": case.phase,
        "frame_number": int(case.frame_number) if case.frame_number is not None else None,
        "image_path": str(case.image_path),
        "mask_path": str(case.mask_path),
        "original_shape_hwd": [int(x) for x in image_nii.shape],
        "preprocessed_shape_dhw": [int(x) for x in target_shape],
        "affine": [[float(v) for v in row] for row in image_nii.affine.tolist()],
    }
    (destination / f"{case_id}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return case_id


def preprocess_acdc_shared(
    output_dir: Path = SHARED_PREPROCESSED_DIR,
    target_shape: Sequence[int] = DEFAULT_TARGET_SHAPE,
    normalization: str = "zscore",
    val_ratio: float = DEFAULT_VAL_RATIO,
    export_nnunet: bool = True,
) -> Dict[str, object]:
    train_cases = collect_acdc_cases(ACDC_TRAIN_DIR)
    test_cases = collect_acdc_cases(ACDC_TEST_DIR)
    train_patients, val_patients = _patient_splits(train_cases, val_ratio=val_ratio)

    split_dirs = {
        "train": output_dir / "train",
        "val": output_dir / "val",
        "test": output_dir / "test",
    }
    for path in split_dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    train_ids: List[str] = []
    val_ids: List[str] = []
    test_ids: List[str] = []

    for case in train_cases:
        if case.patient_id in val_patients:
            val_ids.append(_save_case(case, split_dirs["val"], target_shape, normalization))
        else:
            train_ids.append(_save_case(case, split_dirs["train"], target_shape, normalization))

    for case in test_cases:
        test_ids.append(_save_case(case, split_dirs["test"], target_shape, normalization))

    split_json = {
        "train_patients": train_patients,
        "val_patients": val_patients,
        "train_cases": train_ids,
        "val_cases": val_ids,
        "test_cases": test_ids,
        "target_shape_dhw": list(target_shape),
        "normalization": normalization,
    }
    (output_dir / "splits.json").write_text(json.dumps(split_json, indent=2), encoding="utf-8")

    nnunet_summary = None
    if export_nnunet:
        nnunet_summary = export_acdc_to_nnunet_raw(
            train_split_dir=ACDC_TRAIN_DIR,
            test_split_dir=ACDC_TEST_DIR,
            output_dataset_dir=NNUNET_RAW_DIR,
        )

    return {
        "output_dir": str(output_dir),
        "train_cases": len(train_ids),
        "val_cases": len(val_ids),
        "test_cases": len(test_ids),
        "nnunet_export": nnunet_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=SHARED_PREPROCESSED_DIR)
    parser.add_argument("--target-shape", type=int, nargs=3, default=DEFAULT_TARGET_SHAPE)
    parser.add_argument("--normalization", type=str, default="zscore", choices=["zscore", "minmax", "none"])
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO)
    parser.add_argument("--skip-nnunet-export", action="store_true")
    args = parser.parse_args()

    summary = preprocess_acdc_shared(
        output_dir=args.output_dir,
        target_shape=tuple(args.target_shape),
        normalization=args.normalization,
        val_ratio=args.val_ratio,
        export_nnunet=not args.skip_nnunet_export,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

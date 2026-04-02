"""
Export ACDC into an nnU-Net raw dataset layout.

Reference for dataset.json keys verified against nnU-Net v2's
`generate_dataset_json.py` in the official repository:
https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/dataset_conversion/generate_dataset_json.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from shared_acdc.acdc_dataset import ACDCCase, collect_acdc_cases


def _ensure_dirs(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def _case_id(case: ACDCCase) -> str:
    phase = case.phase.lower() if case.phase is not None else "unknown"
    frame = f"{case.frame_number:02d}" if case.frame_number is not None else "xx"
    return f"{case.patient_id}_{phase}_frame{frame}"


def _copy_case_to_nnunet_raw(case: ACDCCase, images_dir: Path, labels_dir: Path) -> str:
    case_id = _case_id(case)
    shutil.copy2(case.image_path, images_dir / f"{case_id}_0000.nii.gz")
    shutil.copy2(case.mask_path, labels_dir / f"{case_id}.nii.gz")
    return case_id


def _dataset_json(
    dataset_name: str,
    num_training_cases: int,
    description: str,
    citation: str,
    license_text: str,
) -> Dict[str, object]:
    return {
        "name": dataset_name,
        "description": description,
        "channel_names": {
            "0": "MRI",
        },
        "labels": {
            "background": 0,
            "RV": 1,
            "MYO": 2,
            "LV": 3,
        },
        "numTraining": num_training_cases,
        "file_ending": ".nii.gz",
        "citation": citation,
        "licence": license_text,
    }


def export_acdc_to_nnunet_raw(
    train_split_dir: str | Path,
    test_split_dir: str | Path,
    output_dataset_dir: str | Path,
    dataset_name: str = "Dataset701_ACDC",
    description: str = "ACDC exported to nnU-Net raw format from the local source dataset.",
    citation: str = "Please cite the original ACDC dataset and the augmentation paper when used.",
    license_text: str = "See ACDC dataset license terms in the source repository.",
) -> Dict[str, object]:
    """
    Export ACDC ED/ES volumes to an nnU-Net raw directory layout.

    Output structure:
    - DatasetXXX_Name/
      - imagesTr/
      - labelsTr/
      - imagesTs/
      - labelsTs/  # included for convenience, though nnU-Net raw primarily needs labelsTr
      - dataset.json
    """
    train_cases: Sequence[ACDCCase] = collect_acdc_cases(train_split_dir)
    test_cases: Sequence[ACDCCase] = collect_acdc_cases(test_split_dir)

    output_dataset_dir = Path(output_dataset_dir)
    images_tr = output_dataset_dir / "imagesTr"
    labels_tr = output_dataset_dir / "labelsTr"
    images_ts = output_dataset_dir / "imagesTs"
    labels_ts = output_dataset_dir / "labelsTs"
    _ensure_dirs([images_tr, labels_tr, images_ts, labels_ts])

    train_ids: List[str] = []
    for case in train_cases:
        train_ids.append(_copy_case_to_nnunet_raw(case, images_tr, labels_tr))

    test_ids: List[str] = []
    for case in test_cases:
        test_ids.append(_copy_case_to_nnunet_raw(case, images_ts, labels_ts))

    dataset_json = _dataset_json(
        dataset_name=dataset_name,
        num_training_cases=len(train_cases),
        description=description,
        citation=citation,
        license_text=license_text,
    )
    (output_dataset_dir / "dataset.json").write_text(
        json.dumps(dataset_json, indent=2),
        encoding="utf-8",
    )

    summary = {
        "dataset_dir": str(output_dataset_dir),
        "num_training_cases": len(train_cases),
        "num_test_cases": len(test_cases),
        "train_case_ids": train_ids,
        "test_case_ids": test_ids,
    }
    return summary


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    train_dir = repo_root / "ACDC" / "ACDC" / "database" / "training"
    test_dir = repo_root / "ACDC" / "ACDC" / "database" / "testing"
    output_dir = repo_root / "nnUNet_raw" / "Dataset701_ACDC"

    summary = export_acdc_to_nnunet_raw(
        train_split_dir=train_dir,
        test_split_dir=test_dir,
        output_dataset_dir=output_dir,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

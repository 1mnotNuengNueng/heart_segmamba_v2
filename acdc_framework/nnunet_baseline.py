from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Dict

import nibabel as nib
import numpy as np

from shared_acdc.acdc_dataset import collect_acdc_cases
from .metrics import evaluate_multiclass, save_metrics_report
from .preprocess import preprocess_acdc_shared
from .settings import ACDC_TEST_DIR, NNUNET_RAW_DIR, REPO_ROOT


def _case_id(case) -> str:
    return f"{case.patient_id}_{case.phase.lower()}_frame{case.frame_number:02d}"


def prepare_nnunet_environment() -> Dict[str, str]:
    raw_root = NNUNET_RAW_DIR.parent
    preprocessed_root = REPO_ROOT / "nnUNet_preprocessed"
    results_root = REPO_ROOT / "nnUNet_results"

    os.environ["nnUNet_raw"] = str(raw_root)
    os.environ["nnUNet_preprocessed"] = str(preprocessed_root)
    os.environ["nnUNet_results"] = str(results_root)

    preprocessed_root.mkdir(parents=True, exist_ok=True)
    results_root.mkdir(parents=True, exist_ok=True)

    return {
        "nnUNet_raw": str(raw_root),
        "nnUNet_preprocessed": str(preprocessed_root),
        "nnUNet_results": str(results_root),
    }


def preprocess_nnunet(run_planning: bool = True) -> Dict[str, object]:
    shared_summary = preprocess_acdc_shared(export_nnunet=True)
    env_summary = prepare_nnunet_environment()
    command = ["nnUNetv2_plan_and_preprocess", "-d", "701", "--verify_dataset_integrity"]

    if run_planning:
        subprocess.run(command, check=True)

    return {
        "shared_preprocess": shared_summary,
        "nnunet_env": env_summary,
        "plan_command": " ".join(command),
    }


def train_nnunet(configuration: str = "3d_fullres", fold: str = "0", trainer: str | None = None) -> Dict[str, object]:
    env_summary = prepare_nnunet_environment()
    command = ["nnUNetv2_train", "701", configuration, fold]
    if trainer:
        command.extend(["-tr", trainer])
    subprocess.run(command, check=True)
    return {"nnunet_env": env_summary, "train_command": " ".join(command)}


def predict_nnunet(input_dir: Path | None = None, output_dir: Path | None = None, configuration: str = "3d_fullres", fold: str = "0", checkpoint: str = "checkpoint_best.pth") -> Dict[str, object]:
    env_summary = prepare_nnunet_environment()
    input_dir = input_dir or (NNUNET_RAW_DIR / "imagesTs")
    output_dir = output_dir or (REPO_ROOT / "acdc_runs" / "nnunet" / "predictions" / "test")
    output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        "nnUNetv2_predict",
        "-i",
        str(input_dir),
        "-o",
        str(output_dir),
        "-d",
        "701",
        "-c",
        configuration,
        "-f",
        fold,
        "-chk",
        checkpoint,
    ]
    subprocess.run(command, check=True)
    return {"nnunet_env": env_summary, "predict_command": " ".join(command), "output_dir": str(output_dir)}


def evaluate_nnunet_predictions(prediction_dir: Path | None = None) -> Dict[str, object]:
    prediction_dir = prediction_dir or (REPO_ROOT / "acdc_runs" / "nnunet" / "predictions" / "test")
    metrics_dir = REPO_ROOT / "acdc_runs" / "nnunet" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    gt_map = {_case_id(case): case.mask_path for case in collect_acdc_cases(ACDC_TEST_DIR)}

    case_metrics = {}
    mean_dice_values = []
    mean_hd95_values = []

    for pred_path in sorted(prediction_dir.glob("*.nii.gz")):
        case_id = pred_path.name.replace(".nii.gz", "")
        if case_id not in gt_map:
            continue
        pred = np.asarray(nib.load(pred_path).get_fdata(), dtype=np.int16)
        gt = np.asarray(nib.load(gt_map[case_id]).get_fdata(), dtype=np.int16)
        metrics = evaluate_multiclass(pred, gt)
        case_metrics[case_id] = metrics
        mean_dice_values.append(float(metrics["mean_dice"]))
        mean_hd95_values.append(float(metrics["mean_hd95"]))

    summary = {
        "model": "nnunet",
        "split": "test",
        "num_cases": len(case_metrics),
        "mean_dice": float(np.mean(mean_dice_values)) if mean_dice_values else 0.0,
        "mean_hd95": float(np.mean(mean_hd95_values)) if mean_hd95_values else 0.0,
        "cases": case_metrics,
    }
    save_metrics_report(summary, metrics_dir / "test_metrics.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["preprocess", "train", "predict", "evaluate"])
    parser.add_argument("--configuration", default="3d_fullres")
    parser.add_argument("--fold", default="0")
    parser.add_argument("--trainer", default=None)
    parser.add_argument("--checkpoint", default="checkpoint_best.pth")
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    if args.action == "preprocess":
        summary = preprocess_nnunet(run_planning=True)
    elif args.action == "train":
        summary = train_nnunet(configuration=args.configuration, fold=args.fold, trainer=args.trainer)
    elif args.action == "predict":
        summary = predict_nnunet(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            configuration=args.configuration,
            fold=args.fold,
            checkpoint=args.checkpoint,
        )
    else:
        summary = evaluate_nnunet_predictions(prediction_dir=args.output_dir)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

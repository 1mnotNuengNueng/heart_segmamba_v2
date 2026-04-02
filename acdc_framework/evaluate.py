from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import nibabel as nib
import numpy as np

from .challenge_evaluation import evaluate_acdc_challenge_from_prediction_dir
from .metrics import evaluate_multiclass, save_metrics_report
from .settings import model_run_dir


def _case_id_from_prediction_path(pred_path: Path) -> str:
    return pred_path.name.replace(".nii.gz", "")


def evaluate_predictions(model_name: str, split: str = "test") -> Dict[str, object]:
    run_dir = model_run_dir(model_name)
    pred_dir = run_dir / "predictions" / split
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    case_metrics: Dict[str, object] = {}
    mean_dice_values: List[float] = []
    mean_hd95_values: List[float] = []
    mean_ap50_values: List[float] = []
    mean_ap50_95_values: List[float] = []

    for pred_path in sorted(pred_dir.glob("*.nii.gz")):
        case_id = _case_id_from_prediction_path(pred_path)
        metadata_path = pred_dir / f"{case_id}.json"
        meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        gt = np.asarray(nib.load(meta["mask_path"]).get_fdata(), dtype=np.int16)
        pred = np.asarray(nib.load(pred_path).get_fdata(), dtype=np.int16)

        metrics = evaluate_multiclass(pred, gt)
        case_metrics[case_id] = metrics
        mean_dice_values.append(float(metrics["mean_dice"]))
        mean_hd95_values.append(float(metrics["mean_hd95"]))
        mean_ap50_values.append(float(metrics.get("mean_ap50", 0.0)))
        mean_ap50_95_values.append(float(metrics.get("mean_ap50_95", 0.0)))

    summary = {
        "model": model_name,
        "split": split,
        "num_cases": len(case_metrics),
        "mean_dice": float(np.mean(mean_dice_values)) if mean_dice_values else 0.0,
        "mean_hd95": float(np.mean(mean_hd95_values)) if mean_hd95_values else 0.0,
        "mean_ap50": float(np.mean(mean_ap50_values)) if mean_ap50_values else 0.0,
        "mean_ap50_95": float(np.mean(mean_ap50_95_values)) if mean_ap50_95_values else 0.0,
        "cases": case_metrics,
        "challenge_metrics": evaluate_acdc_challenge_from_prediction_dir(pred_dir),
    }
    save_metrics_report(summary, metrics_dir / f"{split}_metrics.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["cnn", "transformer", "segmamba"])
    parser.add_argument("--split", default="test", choices=["val", "test", "train"])
    args = parser.parse_args()
    print(json.dumps(evaluate_predictions(args.model, args.split), indent=2))


if __name__ == "__main__":
    main()

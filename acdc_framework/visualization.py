from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

from .settings import LV_LABEL, MYO_LABEL, RV_LABEL, model_run_dir


LABEL_COLORS = {
    RV_LABEL: np.array([0.10, 0.70, 0.95], dtype=np.float32),
    MYO_LABEL: np.array([0.95, 0.55, 0.15], dtype=np.float32),
    LV_LABEL: np.array([0.95, 0.20, 0.35], dtype=np.float32),
}


def _normalize_image(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    min_value = float(image.min())
    max_value = float(image.max())
    if max_value - min_value < 1e-8:
        return np.zeros_like(image, dtype=np.float32)
    return (image - min_value) / (max_value - min_value)


def _to_rgb(image_2d: np.ndarray) -> np.ndarray:
    base = _normalize_image(image_2d)
    return np.stack([base, base, base], axis=-1)


def _overlay_mask(image_2d: np.ndarray, mask_2d: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    rgb = _to_rgb(image_2d)
    result = rgb.copy()
    for label, color in LABEL_COLORS.items():
        region = mask_2d == label
        if np.any(region):
            result[region] = (1.0 - alpha) * result[region] + alpha * color
    return np.clip(result, 0.0, 1.0)


def _find_case_prediction(prediction_dir: Path, case_id: Optional[str]) -> Path:
    if case_id is not None:
        pred_path = prediction_dir / f"{case_id}.nii.gz"
        if not pred_path.exists():
            raise FileNotFoundError(f"Prediction file not found: {pred_path}")
        return pred_path

    candidates = sorted(prediction_dir.glob("*.nii.gz"))
    if not candidates:
        raise FileNotFoundError(f"No prediction files found in {prediction_dir}")
    return candidates[0]


def _load_case_bundle(prediction_dir: Path, pred_path: Path):
    resolved_case_id = pred_path.name.replace(".nii.gz", "")
    meta = json.loads((prediction_dir / f"{resolved_case_id}.json").read_text(encoding="utf-8"))
    image = np.asarray(nib.load(meta["image_path"]).get_fdata(), dtype=np.float32)
    gt = np.asarray(nib.load(meta["mask_path"]).get_fdata(), dtype=np.int16)
    pred = np.asarray(nib.load(pred_path).get_fdata(), dtype=np.int16)
    return resolved_case_id, meta, image, gt, pred


def _select_slice(gt: np.ndarray, fallback_depth: int, slice_index: Optional[int]) -> int:
    if slice_index is not None:
        return int(slice_index)
    gt_foreground = np.argwhere(gt > 0)
    if gt_foreground.size > 0:
        return int(np.median(gt_foreground[:, 2]))
    return int(fallback_depth // 2)


def _case_mean_dice(gt: np.ndarray, pred: np.ndarray) -> float:
    scores = []
    for label in (RV_LABEL, MYO_LABEL, LV_LABEL):
        gt_bin = gt == label
        pred_bin = pred == label
        denom = gt_bin.sum() + pred_bin.sum()
        if denom == 0:
            scores.append(1.0)
        else:
            scores.append(float(2.0 * np.logical_and(gt_bin, pred_bin).sum() / denom))
    return float(np.mean(scores))


def create_prediction_figure(
    model_name: str,
    case_id: Optional[str] = None,
    split: str = "test",
    slice_index: Optional[int] = None,
    output_path: Optional[Path] = None,
) -> Dict[str, str]:
    run_dir = model_run_dir(model_name)
    prediction_dir = run_dir / "predictions" / split
    pred_path = _find_case_prediction(prediction_dir, case_id)
    resolved_case_id, meta, image, gt, pred = _load_case_bundle(prediction_dir, pred_path)

    slice_index = _select_slice(gt, image.shape[2], slice_index)

    image_slice = image[:, :, slice_index]
    gt_slice = gt[:, :, slice_index]
    pred_slice = pred[:, :, slice_index]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.ravel()

    axes[0].imshow(image_slice, cmap="gray")
    axes[0].set_title("MRI Slice")

    axes[1].imshow(gt_slice, cmap="viridis", interpolation="nearest", vmin=0, vmax=3)
    axes[1].set_title("Ground Truth")

    axes[2].imshow(pred_slice, cmap="viridis", interpolation="nearest", vmin=0, vmax=3)
    axes[2].set_title("Prediction")

    axes[3].imshow(_overlay_mask(image_slice, gt_slice))
    axes[3].set_title("GT Overlay")

    axes[4].imshow(_overlay_mask(image_slice, pred_slice))
    axes[4].set_title("Prediction Overlay")

    diff = np.zeros_like(pred_slice, dtype=np.int16)
    diff[(gt_slice > 0) & (pred_slice == 0)] = 1
    diff[(gt_slice == 0) & (pred_slice > 0)] = 2
    diff[(gt_slice > 0) & (pred_slice > 0) & (gt_slice != pred_slice)] = 3
    axes[5].imshow(diff, cmap="magma", interpolation="nearest")
    axes[5].set_title("Difference Map")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(f"{model_name.upper()} | {resolved_case_id} | slice {slice_index}", fontsize=14)
    fig.tight_layout()

    if output_path is None:
        output_path = run_dir / "visualizations" / f"{resolved_case_id}_slice{slice_index:02d}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "model": model_name,
        "case_id": resolved_case_id,
        "slice_index": str(slice_index),
        "output_path": str(output_path),
    }
    return summary


def create_case_grid(
    model_name: str,
    split: str = "test",
    case_ids: Optional[List[str]] = None,
    max_cases: int = 6,
    output_path: Optional[Path] = None,
) -> Dict[str, str]:
    run_dir = model_run_dir(model_name)
    prediction_dir = run_dir / "predictions" / split
    prediction_paths = sorted(prediction_dir.glob("*.nii.gz"))
    if case_ids is not None:
        selected_paths = [prediction_dir / f"{case_id}.nii.gz" for case_id in case_ids]
    else:
        selected_paths = prediction_paths[:max_cases]

    if not selected_paths:
        raise FileNotFoundError(f"No prediction files found in {prediction_dir}")

    fig, axes = plt.subplots(len(selected_paths), 3, figsize=(12, 4 * len(selected_paths)))
    axes = np.atleast_2d(axes)

    rendered_case_ids: List[str] = []
    for row_idx, pred_path in enumerate(selected_paths):
        case_id, _meta, image, gt, pred = _load_case_bundle(prediction_dir, pred_path)
        slice_index = _select_slice(gt, image.shape[2], None)

        image_slice = image[:, :, slice_index]
        gt_slice = gt[:, :, slice_index]
        pred_slice = pred[:, :, slice_index]

        axes[row_idx, 0].imshow(image_slice, cmap="gray")
        axes[row_idx, 0].set_title(f"{case_id} | MRI")
        axes[row_idx, 1].imshow(_overlay_mask(image_slice, gt_slice))
        axes[row_idx, 1].set_title("GT Overlay")
        axes[row_idx, 2].imshow(_overlay_mask(image_slice, pred_slice))
        axes[row_idx, 2].set_title("Prediction Overlay")

        for col in range(3):
            axes[row_idx, col].axis("off")
        rendered_case_ids.append(case_id)

    fig.tight_layout()
    if output_path is None:
        output_path = run_dir / "visualizations" / f"{split}_case_grid.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    return {
        "model": model_name,
        "split": split,
        "output_path": str(output_path),
        "case_ids": rendered_case_ids,
    }


def create_best_worst_report(
    model_name: str,
    split: str = "test",
    top_k: int = 3,
    output_dir: Optional[Path] = None,
) -> Dict[str, object]:
    run_dir = model_run_dir(model_name)
    prediction_dir = run_dir / "predictions" / split
    prediction_paths = sorted(prediction_dir.glob("*.nii.gz"))
    if not prediction_paths:
        raise FileNotFoundError(f"No prediction files found in {prediction_dir}")

    case_scores = []
    for pred_path in prediction_paths:
        case_id, _meta, _image, gt, pred = _load_case_bundle(prediction_dir, pred_path)
        case_scores.append({"case_id": case_id, "mean_dice": _case_mean_dice(gt, pred)})

    case_scores = sorted(case_scores, key=lambda item: item["mean_dice"], reverse=True)
    best_cases = case_scores[:top_k]
    worst_cases = case_scores[-top_k:] if len(case_scores) >= top_k else case_scores

    if output_dir is None:
        output_dir = run_dir / "visualizations" / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    best_grid = create_case_grid(
        model_name=model_name,
        split=split,
        case_ids=[item["case_id"] for item in best_cases],
        output_path=output_dir / "best_cases.png",
    )
    worst_grid = create_case_grid(
        model_name=model_name,
        split=split,
        case_ids=[item["case_id"] for item in worst_cases],
        output_path=output_dir / "worst_cases.png",
    )

    summary = {
        "model": model_name,
        "split": split,
        "top_k": top_k,
        "best_cases": best_cases,
        "worst_cases": worst_cases,
        "best_grid": best_grid["output_path"],
        "worst_grid": worst_grid["output_path"],
    }
    (output_dir / "best_worst_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["cnn", "transformer", "segmamba"])
    parser.add_argument("--case-id", default=None)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--slice-index", type=int, default=None)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--mode", choices=["single", "grid", "bestworst"], default="single")
    parser.add_argument("--max-cases", type=int, default=6)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    if args.mode == "single":
        result = create_prediction_figure(
            model_name=args.model,
            case_id=args.case_id,
            split=args.split,
            slice_index=args.slice_index,
            output_path=args.output_path,
        )
    elif args.mode == "grid":
        result = create_case_grid(
            model_name=args.model,
            split=args.split,
            max_cases=args.max_cases,
            output_path=args.output_path,
        )
    else:
        result = create_best_worst_report(
            model_name=args.model,
            split=args.split,
            top_k=args.top_k,
            output_dir=args.output_path,
        )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

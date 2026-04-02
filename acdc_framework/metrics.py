from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt

from .settings import FOREGROUND_LABELS


def dice_score(pred: np.ndarray, target: np.ndarray) -> float:
    pred = pred.astype(bool)
    target = target.astype(bool)
    denom = pred.sum() + target.sum()
    if denom == 0:
        return 1.0
    return float(2.0 * np.logical_and(pred, target).sum() / denom)


def iou_score(pred: np.ndarray, target: np.ndarray) -> float:
    pred = pred.astype(bool)
    target = target.astype(bool)
    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)


def _surface_distances(mask_a: np.ndarray, mask_b: np.ndarray, spacing=(1.0, 1.0, 1.0)) -> np.ndarray:
    if not np.any(mask_a) and not np.any(mask_b):
        return np.asarray([0.0], dtype=np.float32)
    if not np.any(mask_a) or not np.any(mask_b):
        return np.asarray([np.inf], dtype=np.float32)

    border_a = np.logical_xor(mask_a, binary_erosion(mask_a))
    border_b = np.logical_xor(mask_b, binary_erosion(mask_b))
    dt_a = distance_transform_edt(~border_a, sampling=spacing)
    dt_b = distance_transform_edt(~border_b, sampling=spacing)
    return np.concatenate([dt_b[border_a], dt_a[border_b]]).astype(np.float32)


def hd95_score(pred: np.ndarray, target: np.ndarray, spacing=(1.0, 1.0, 1.0)) -> float:
    distances = _surface_distances(pred.astype(bool), target.astype(bool), spacing=spacing)
    if np.isinf(distances).any():
        return float("inf")
    return float(np.percentile(distances, 95))


def evaluate_multiclass(pred: np.ndarray, target: np.ndarray) -> Dict[str, object]:
    class_metrics: Dict[str, Dict[str, float]] = {}
    dice_values: List[float] = []
    hd95_values: List[float] = []
    ap50_values: List[float] = []
    ap50_95_values: List[float] = []

    for cls in FOREGROUND_LABELS:
        pred_cls = pred == cls
        target_cls = target == cls
        cls_dice = dice_score(pred_cls, target_cls)
        cls_hd95 = hd95_score(pred_cls, target_cls)
        
        cls_iou = iou_score(pred_cls, target_cls)
        cls_ap50 = 1.0 if cls_iou >= 0.50 else 0.0
        cls_ap50_95 = float(np.mean([1.0 if cls_iou >= t else 0.0 for t in np.arange(0.50, 1.00, 0.05)]))
        
        class_metrics[str(cls)] = {"dice": cls_dice, "hd95": cls_hd95, "ap50": cls_ap50, "ap50_95": cls_ap50_95}
        dice_values.append(cls_dice)
        hd95_values.append(cls_hd95)
        ap50_values.append(cls_ap50)
        ap50_95_values.append(cls_ap50_95)

    return {
        "per_class": class_metrics,
        "mean_dice": float(np.mean(dice_values)),
        "mean_hd95": float(np.mean(hd95_values)),
        "mean_ap50": float(np.mean(ap50_values)),
        "mean_ap50_95": float(np.mean(ap50_95_values)),
    }


def save_metrics_report(metrics: Dict[str, object], path: str | Path) -> None:
    Path(path).write_text(json.dumps(metrics, indent=2), encoding="utf-8")

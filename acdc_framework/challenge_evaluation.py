from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt

from .settings import FOREGROUND_LABELS, MYO_LABEL, RV_LABEL, LV_LABEL


STRUCTURES = {
    "LV": LV_LABEL,
    "RV": RV_LABEL,
    "MYO": MYO_LABEL,
}
MYO_DENSITY_G_PER_ML = 1.05


@dataclass
class CaseEvaluation:
    case_id: str
    patient_id: str
    phase: str
    pred: np.ndarray
    gt: np.ndarray
    spacing_xyz: Tuple[float, float, float]


def _dice(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0
    return float(2.0 * np.logical_and(pred, gt).sum() / denom)


def _surface_distances(mask_a: np.ndarray, mask_b: np.ndarray, spacing_xyz: Tuple[float, float, float]) -> np.ndarray:
    if not np.any(mask_a) and not np.any(mask_b):
        return np.asarray([0.0], dtype=np.float32)
    if not np.any(mask_a) or not np.any(mask_b):
        return np.asarray([np.inf], dtype=np.float32)

    sampling = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])
    border_a = np.logical_xor(mask_a, binary_erosion(mask_a))
    border_b = np.logical_xor(mask_b, binary_erosion(mask_b))
    dt_a = distance_transform_edt(~border_a, sampling=sampling)
    dt_b = distance_transform_edt(~border_b, sampling=sampling)
    return np.concatenate([dt_b[border_a], dt_a[border_b]]).astype(np.float32)


def _hausdorff(pred: np.ndarray, gt: np.ndarray, spacing_xyz: Tuple[float, float, float]) -> float:
    distances = _surface_distances(pred.astype(bool), gt.astype(bool), spacing_xyz)
    if np.isinf(distances).any():
        return float("inf")
    return float(np.max(distances))


def _voxel_volume_ml(spacing_xyz: Tuple[float, float, float]) -> float:
    return float(np.prod(spacing_xyz) / 1000.0)


def _volume_ml(mask: np.ndarray, label: int, spacing_xyz: Tuple[float, float, float]) -> float:
    return float(np.sum(mask == label) * _voxel_volume_ml(spacing_xyz))


def _corrcoef(pred_values: List[float], gt_values: List[float]) -> float:
    if len(pred_values) < 2:
        return float("nan")
    pred_arr = np.asarray(pred_values, dtype=np.float64)
    gt_arr = np.asarray(gt_values, dtype=np.float64)
    if np.allclose(pred_arr, pred_arr[0]) or np.allclose(gt_arr, gt_arr[0]):
        return float("nan")
    return float(np.corrcoef(pred_arr, gt_arr)[0, 1])


def _bias(pred_values: List[float], gt_values: List[float]) -> float:
    return float(np.mean(np.asarray(pred_values) - np.asarray(gt_values)))


def _loa(pred_values: List[float], gt_values: List[float]) -> float:
    diffs = np.asarray(pred_values) - np.asarray(gt_values)
    if len(diffs) <= 1:
        return 0.0
    return float(1.96 * np.std(diffs, ddof=1))


def _clinical_summary(pred_values: List[float], gt_values: List[float]) -> Dict[str, float]:
    return {
        "correlation": _corrcoef(pred_values, gt_values),
        "bias": _bias(pred_values, gt_values),
        "loa": _loa(pred_values, gt_values),
    }


def _load_case_evaluations(pred_dir: Path) -> List[CaseEvaluation]:
    evaluations: List[CaseEvaluation] = []
    for pred_path in sorted(pred_dir.glob("*.nii.gz")):
        case_id = pred_path.name.replace(".nii.gz", "")
        meta = json.loads((pred_dir / f"{case_id}.json").read_text(encoding="utf-8"))
        pred = np.asarray(nib.load(pred_path).get_fdata(), dtype=np.int16)
        gt_nii = nib.load(meta["mask_path"])
        gt = np.asarray(gt_nii.get_fdata(), dtype=np.int16)
        spacing_xyz = tuple(float(x) for x in gt_nii.header.get_zooms()[:3])
        evaluations.append(
            CaseEvaluation(
                case_id=case_id,
                patient_id=str(meta["patient_id"]),
                phase=str(meta["phase"]),
                pred=pred,
                gt=gt,
                spacing_xyz=spacing_xyz,
            )
        )
    return evaluations


def _geometric_metrics(cases: List[CaseEvaluation]) -> Dict[str, Dict[str, Dict[str, float]]]:
    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for structure_name, label in STRUCTURES.items():
        summary[structure_name] = {}
        for phase in ("ED", "ES"):
            selected = [case for case in cases if case.phase == phase]
            dice_values = [_dice(case.pred == label, case.gt == label) for case in selected]
            hd_values = [_hausdorff(case.pred == label, case.gt == label, case.spacing_xyz) for case in selected]
            summary[structure_name][phase] = {
                "dice_mean": float(np.mean(dice_values)) if dice_values else 0.0,
                "hausdorff_mm_mean": float(np.mean(hd_values)) if hd_values else 0.0,
            }
    return summary


def _clinical_metrics(cases: List[CaseEvaluation]) -> Dict[str, object]:
    by_patient: Dict[str, Dict[str, CaseEvaluation]] = {}
    for case in cases:
        by_patient.setdefault(case.patient_id, {})[case.phase] = case

    lv_edv_pred: List[float] = []
    lv_edv_gt: List[float] = []
    lv_esv_pred: List[float] = []
    lv_esv_gt: List[float] = []
    lv_ef_pred: List[float] = []
    lv_ef_gt: List[float] = []

    rv_edv_pred: List[float] = []
    rv_edv_gt: List[float] = []
    rv_esv_pred: List[float] = []
    rv_esv_gt: List[float] = []
    rv_ef_pred: List[float] = []
    rv_ef_gt: List[float] = []

    myo_mass_ed_pred: List[float] = []
    myo_mass_ed_gt: List[float] = []
    myo_esv_pred: List[float] = []
    myo_esv_gt: List[float] = []

    per_patient: Dict[str, Dict[str, float]] = {}
    for patient_id, phases in by_patient.items():
        if "ED" not in phases or "ES" not in phases:
            continue
        ed = phases["ED"]
        es = phases["ES"]

        lv_edv_p = _volume_ml(ed.pred, LV_LABEL, ed.spacing_xyz)
        lv_edv_g = _volume_ml(ed.gt, LV_LABEL, ed.spacing_xyz)
        lv_esv_p = _volume_ml(es.pred, LV_LABEL, es.spacing_xyz)
        lv_esv_g = _volume_ml(es.gt, LV_LABEL, es.spacing_xyz)
        lv_ef_p = 100.0 * (lv_edv_p - lv_esv_p) / max(lv_edv_p, 1e-8)
        lv_ef_g = 100.0 * (lv_edv_g - lv_esv_g) / max(lv_edv_g, 1e-8)

        rv_edv_p = _volume_ml(ed.pred, RV_LABEL, ed.spacing_xyz)
        rv_edv_g = _volume_ml(ed.gt, RV_LABEL, ed.spacing_xyz)
        rv_esv_p = _volume_ml(es.pred, RV_LABEL, es.spacing_xyz)
        rv_esv_g = _volume_ml(es.gt, RV_LABEL, es.spacing_xyz)
        rv_ef_p = 100.0 * (rv_edv_p - rv_esv_p) / max(rv_edv_p, 1e-8)
        rv_ef_g = 100.0 * (rv_edv_g - rv_esv_g) / max(rv_edv_g, 1e-8)

        myo_mass_ed_p = _volume_ml(ed.pred, MYO_LABEL, ed.spacing_xyz) * MYO_DENSITY_G_PER_ML
        myo_mass_ed_g = _volume_ml(ed.gt, MYO_LABEL, ed.spacing_xyz) * MYO_DENSITY_G_PER_ML
        myo_esv_p = _volume_ml(es.pred, MYO_LABEL, es.spacing_xyz)
        myo_esv_g = _volume_ml(es.gt, MYO_LABEL, es.spacing_xyz)

        lv_edv_pred.append(lv_edv_p)
        lv_edv_gt.append(lv_edv_g)
        lv_esv_pred.append(lv_esv_p)
        lv_esv_gt.append(lv_esv_g)
        lv_ef_pred.append(lv_ef_p)
        lv_ef_gt.append(lv_ef_g)

        rv_edv_pred.append(rv_edv_p)
        rv_edv_gt.append(rv_edv_g)
        rv_esv_pred.append(rv_esv_p)
        rv_esv_gt.append(rv_esv_g)
        rv_ef_pred.append(rv_ef_p)
        rv_ef_gt.append(rv_ef_g)

        myo_mass_ed_pred.append(myo_mass_ed_p)
        myo_mass_ed_gt.append(myo_mass_ed_g)
        myo_esv_pred.append(myo_esv_p)
        myo_esv_gt.append(myo_esv_g)

        per_patient[patient_id] = {
            "lv_edv_ml_pred": lv_edv_p,
            "lv_edv_ml_gt": lv_edv_g,
            "lv_esv_ml_pred": lv_esv_p,
            "lv_esv_ml_gt": lv_esv_g,
            "lv_ef_pred": lv_ef_p,
            "lv_ef_gt": lv_ef_g,
            "rv_edv_ml_pred": rv_edv_p,
            "rv_edv_ml_gt": rv_edv_g,
            "rv_esv_ml_pred": rv_esv_p,
            "rv_esv_ml_gt": rv_esv_g,
            "rv_ef_pred": rv_ef_p,
            "rv_ef_gt": rv_ef_g,
            "myo_mass_ed_g_pred": myo_mass_ed_p,
            "myo_mass_ed_g_gt": myo_mass_ed_g,
            "myo_esv_ml_pred": myo_esv_p,
            "myo_esv_ml_gt": myo_esv_g,
        }

    return {
        "LV": {
            "EDV": _clinical_summary(lv_edv_pred, lv_edv_gt),
            "ESV": _clinical_summary(lv_esv_pred, lv_esv_gt),
            "EF": _clinical_summary(lv_ef_pred, lv_ef_gt),
        },
        "RV": {
            "EDV": _clinical_summary(rv_edv_pred, rv_edv_gt),
            "ESV": _clinical_summary(rv_esv_pred, rv_esv_gt),
            "EF": _clinical_summary(rv_ef_pred, rv_ef_gt),
        },
        "MYO": {
            "Mass_ED": _clinical_summary(myo_mass_ed_pred, myo_mass_ed_gt),
            "ESV": _clinical_summary(myo_esv_pred, myo_esv_gt),
        },
        "per_patient": per_patient,
    }


def evaluate_acdc_challenge_from_prediction_dir(pred_dir: str | Path) -> Dict[str, object]:
    pred_dir = Path(pred_dir)
    cases = _load_case_evaluations(pred_dir)
    return {
        "num_cases": len(cases),
        "geometric": _geometric_metrics(cases),
        "clinical": _clinical_metrics(cases),
    }

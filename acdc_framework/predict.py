from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import nibabel as nib
import numpy as np
import torch
from scipy import ndimage
from tqdm import tqdm

from .dataset import PreprocessedACDCDataset, create_loader
from .models import build_model
from .settings import model_run_dir


def _unwrap_meta_value(value):
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.item()
        return [_unwrap_meta_value(v) for v in value.tolist()]
    if isinstance(value, np.ndarray):
        return [_unwrap_meta_value(v) for v in value.tolist()]
    if isinstance(value, tuple):
        return [_unwrap_meta_value(v) for v in value]
    if isinstance(value, list):
        if len(value) == 1:
            return _unwrap_meta_value(value[0])
        return [_unwrap_meta_value(v) for v in value]
    return value


def _normalize_shape(shape_value) -> List[int]:
    value = _unwrap_meta_value(shape_value)
    while isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    return [int(v) for v in value]


def _normalize_affine(affine_value) -> np.ndarray:
    value = _unwrap_meta_value(affine_value)
    while isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        if len(value[0]) == 4 and all(isinstance(row, list) for row in value[0]):
            value = value[0]
            break
        value = value[0]
    return np.asarray(value, dtype=np.float32)


def _resize_to_original(pred_dhw: np.ndarray, original_shape_hwd: List[int]) -> np.ndarray:
    pred_hwd = np.transpose(pred_dhw, (1, 2, 0))
    original_shape_hwd = _normalize_shape(original_shape_hwd)
    zoom = [o / i for o, i in zip(original_shape_hwd, pred_hwd.shape)]
    return ndimage.zoom(pred_hwd, zoom=zoom, order=0).astype(np.uint8)


def predict_model(model_name: str, split: str = "test", checkpoint_name: str = "best.pt", preprocessed_root: Path | None = None) -> Dict[str, object]:
    run_dir = model_run_dir(model_name)
    pred_dir = run_dir / "predictions" / split
    pred_dir.mkdir(parents=True, exist_ok=True)

    ds = PreprocessedACDCDataset(split, root_dir=preprocessed_root) if preprocessed_root else PreprocessedACDCDataset(split)
    loader = create_loader(ds, batch_size=1, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_name).to(device)
    state_dict = torch.load(run_dir / "checkpoints" / checkpoint_name, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    saved_cases: List[str] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"predict-{model_name}-{split}"):
            image = batch["image"].to(device)
            meta = {key: _unwrap_meta_value(value) for key, value in batch["meta"].items()}
            logits = model(image)
            pred = logits.argmax(dim=1)[0].cpu().numpy()

            case_id = str(meta["case_id"])
            affine = _normalize_affine(meta["affine"])
            original_shape = _normalize_shape(meta["original_shape_hwd"])
            restored = _resize_to_original(pred, original_shape)

            nib.save(nib.Nifti1Image(restored.astype(np.uint8), affine), pred_dir / f"{case_id}.nii.gz")
            (pred_dir / f"{case_id}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            saved_cases.append(case_id)

    summary = {"model": model_name, "split": split, "saved_cases": saved_cases, "prediction_dir": str(pred_dir)}
    (run_dir / f"predict_{split}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["cnn", "transformer", "segmamba"])
    parser.add_argument("--split", default="test", choices=["val", "test", "train"])
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--preprocessed-root", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(predict_model(args.model, args.split, args.checkpoint, args.preprocessed_root), indent=2))


if __name__ == "__main__":
    main()

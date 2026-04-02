from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Sequence

import nibabel as nib
import numpy as np
import torch
from scipy import ndimage

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from acdc_framework.models import build_model
from acdc_framework.settings import DEFAULT_TARGET_SHAPE

DEMO_ROOT = Path(__file__).resolve().parent
DEMO_WEIGHTS_DIR = DEMO_ROOT / "weights"
DEMO_DATA_DIR = DEMO_ROOT / "data"
DEMO_OUTPUT_DIR = DEMO_ROOT / "output"


def _resolve_demo_path(path: Path, base_dir: Path) -> Path:
    path = Path(path)
    if path.exists():
        return path.resolve()

    candidate = (base_dir / path).resolve()
    if candidate.exists():
        return candidate

    matches = list(base_dir.rglob(path.name))
    if len(matches) == 1:
        return matches[0].resolve()
    if len(matches) > 1:
        raise FileNotFoundError(
            f"Found multiple matches for '{path.name}' under '{base_dir}': "
            + ", ".join(str(match) for match in matches)
        )
    raise FileNotFoundError(f"Could not find '{path}' under '{base_dir}'.")


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


def _prepare_image(image_path: Path, target_shape: Sequence[int], normalization: str) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int]]:
    image_nii = nib.load(str(image_path))
    image = np.asarray(image_nii.get_fdata(), dtype=np.float32)
    original_shape_hwd = tuple(int(v) for v in image.shape)
    image_dhw = np.transpose(image, (2, 0, 1))
    resized_image = _resize_volume(image_dhw, target_shape, order=1)
    normalized_image = _normalize(resized_image, normalization)
    return normalized_image[None], image_nii.affine, original_shape_hwd


def _restore_prediction(pred_dhw: np.ndarray, original_shape_hwd: Sequence[int]) -> np.ndarray:
    pred_hwd = np.transpose(pred_dhw, (1, 2, 0))
    zoom = [o / i for o, i in zip(original_shape_hwd, pred_hwd.shape)]
    restored = ndimage.zoom(pred_hwd, zoom=zoom, order=0)
    return restored.astype(np.uint8)


def run_demo(
    model_name: str,
    image_path: Path,
    weights_path: Path,
    output_path: Path | None = None,
    target_shape: Sequence[int] = DEFAULT_TARGET_SHAPE,
    normalization: str = "zscore",
) -> Dict[str, object]:
    DEMO_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    DEMO_DATA_DIR.mkdir(parents=True, exist_ok=True)
    DEMO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_path = _resolve_demo_path(Path(image_path), DEMO_DATA_DIR)
    weights_path = _resolve_demo_path(Path(weights_path), DEMO_WEIGHTS_DIR)
    if output_path is None:
        output_path = DEMO_OUTPUT_DIR / f"{image_path.stem}_pred.nii.gz"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    input_tensor, affine, original_shape_hwd = _prepare_image(
        image_path=image_path,
        target_shape=target_shape,
        normalization=normalization,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_name).to(device)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        image = torch.from_numpy(input_tensor).unsqueeze(0).to(device)
        logits = model(image)
        pred = logits.argmax(dim=1)[0].cpu().numpy()

    restored = _restore_prediction(pred, original_shape_hwd)
    nib.save(nib.Nifti1Image(restored, affine), output_path)

    summary = {
        "model": model_name,
        "image_path": str(image_path),
        "weights_path": str(weights_path),
        "output_path": str(output_path),
        "normalization": normalization,
        "target_shape_dhw": [int(v) for v in target_shape],
        "original_shape_hwd": [int(v) for v in original_shape_hwd],
    }
    output_path.with_suffix("").with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    DEMO_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    DEMO_DATA_DIR.mkdir(parents=True, exist_ok=True)
    DEMO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["cnn", "transformer", "segmamba"])
    parser.add_argument("--image", type=Path, required=True, help=f"Input NIfTI image, e.g. {DEMO_DATA_DIR}\\case.nii.gz")
    parser.add_argument("--weights", type=Path, required=True, help=f"Path to a .pt weight file, e.g. {DEMO_WEIGHTS_DIR}\\best.pt")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--target-shape", type=int, nargs=3, default=DEFAULT_TARGET_SHAPE)
    parser.add_argument("--normalization", type=str, default="zscore", choices=["zscore", "minmax", "none"])
    args = parser.parse_args()

    print(
        json.dumps(
            run_demo(
                model_name=args.model,
                image_path=args.image,
                weights_path=args.weights,
                output_path=args.output,
                target_shape=tuple(args.target_shape),
                normalization=args.normalization,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

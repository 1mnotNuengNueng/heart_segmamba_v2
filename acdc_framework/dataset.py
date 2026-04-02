from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

import monai.transforms as mt
from .settings import SHARED_PREPROCESSED_DIR


class PreprocessedACDCDataset(Dataset):
    def __init__(
        self,
        split: str,
        root_dir: Path = SHARED_PREPROCESSED_DIR,
        augment: bool = False,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.split = split
        self.files: List[Path] = sorted((self.root_dir / split).glob("*.npz"))
        self.augment = augment

        self.transform = None
        if self.augment:
            self.transform = mt.Compose([
                mt.RandAffined(
                    keys=["image", "label"],
                    prob=0.2,
                    rotate_range=(np.pi/6, np.pi/6, np.pi/6),
                    scale_range=(0.3, 0.3, 0.3),
                    mode=("bilinear", "nearest"),
                    padding_mode=("border", "border"),
                ),
                mt.RandGaussianNoised(keys=["image"], prob=0.1, mean=0.0, std=0.1),
                mt.RandGaussianSmoothd(keys=["image"], prob=0.1, sigma_x=(0.5, 1.5), sigma_y=(0.5, 1.5), sigma_z=(0.5, 1.5)),
                mt.RandScaleIntensityd(keys=["image"], factors=0.3, prob=0.15),
                mt.RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.15),
                mt.RandAdjustContrastd(keys=["image"], gamma=(0.7, 1.5), prob=0.15),
                mt.RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),
                mt.RandFlipd(keys=["image", "label"], spatial_axis=1, prob=0.5),
                mt.RandFlipd(keys=["image", "label"], spatial_axis=2, prob=0.5),
            ])

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Dict[str, object]:
        npz_path = self.files[index]
        payload = np.load(npz_path)
        meta = json.loads(npz_path.with_suffix(".json").read_text(encoding="utf-8"))
        image_np = payload["image"].astype(np.float32)
        label_np = payload["label"].astype(np.float32)

        data = {"image": image_np, "label": label_np[None]}

        if self.augment and self.transform is not None:
            data = self.transform(data)

        image = torch.as_tensor(data["image"]).float()
        label = torch.as_tensor(data["label"]).squeeze(0).long()
        return {
            "image": image,
            "label": label,
            "meta": meta,
        }


def create_loader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

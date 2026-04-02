from __future__ import annotations

import sys
from pathlib import Path

import torch.nn as nn
from monai.networks.nets import UNETR, UNet

from .settings import DEFAULT_TARGET_SHAPE, NUM_CLASSES, REPO_ROOT


def build_cnn_model() -> nn.Module:
    return UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=NUM_CLASSES,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )


def build_transformer_model(img_size=DEFAULT_TARGET_SHAPE) -> nn.Module:
    return UNETR(
        in_channels=1,
        out_channels=NUM_CLASSES,
        img_size=img_size,
        feature_size=16,
        hidden_size=384,
        mlp_dim=1536,
        num_heads=6,
        proj_type="conv",
        norm_name="instance",
        res_block=True,
    )


def build_segmamba_model() -> nn.Module:
    segmamba_root = REPO_ROOT / "SegMamba"
    if str(segmamba_root) not in sys.path:
        sys.path.insert(0, str(segmamba_root))
    from models_segmamba.segmambav2 import SegMamba  # type: ignore

    return SegMamba(1, NUM_CLASSES)


def build_model(model_name: str) -> nn.Module:
    model_name = model_name.lower()
    if model_name == "cnn":
        return build_cnn_model()
    if model_name == "transformer":
        return build_transformer_model()
    if model_name == "segmamba":
        return build_segmamba_model()
    raise ValueError(f"Unsupported model name: {model_name}")

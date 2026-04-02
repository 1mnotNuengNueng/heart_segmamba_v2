from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict

import torch

from .models import build_model
from .settings import DEFAULT_TARGET_SHAPE, model_run_dir

try:
    from thop import profile
except ImportError:  # pragma: no cover
    profile = None


def _count_parameters(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def _measure_latency(
    model: torch.nn.Module,
    dummy_input: torch.Tensor,
    warmup: int = 10,
    runs: int = 30,
) -> Dict[str, float]:
    model.eval()
    device = dummy_input.device

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        start = time.perf_counter()
        for _ in range(runs):
            _ = model(dummy_input)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        end = time.perf_counter()

    mean_seconds = (end - start) / runs
    return {
        "latency_seconds_mean": float(mean_seconds),
        "latency_milliseconds_mean": float(mean_seconds * 1000.0),
    }


def _measure_flops(model: torch.nn.Module, dummy_input: torch.Tensor) -> Dict[str, float | None]:
    if profile is None:
        return {"flops": None, "macs": None}

    macs, _params = profile(model, inputs=(dummy_input,), verbose=False)
    return {
        "macs": float(macs),
        "flops": float(2.0 * macs),
    }


def profile_model(
    model_name: str,
    use_cuda: bool = True,
    batch_size: int = 1,
    input_shape_dhw = DEFAULT_TARGET_SHAPE,
) -> Dict[str, object]:
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    model = build_model(model_name).to(device)
    dummy_input = torch.randn(batch_size, 1, *input_shape_dhw, device=device)

    params = _count_parameters(model)
    complexity = _measure_flops(model, dummy_input)
    latency = _measure_latency(model, dummy_input)

    summary = {
        "model": model_name,
        "device": str(device),
        "batch_size": int(batch_size),
        "input_shape_dhw": [int(v) for v in input_shape_dhw],
        "parameters": params,
        **complexity,
        **latency,
    }

    run_dir = model_run_dir(model_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "profile_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["cnn", "transformer", "segmamba"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--input-shape", type=int, nargs=3, default=DEFAULT_TARGET_SHAPE)
    args = parser.parse_args()
    print(
        json.dumps(
            profile_model(
                model_name=args.model,
                use_cuda=not args.cpu,
                batch_size=args.batch_size,
                input_shape_dhw=tuple(args.input_shape),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

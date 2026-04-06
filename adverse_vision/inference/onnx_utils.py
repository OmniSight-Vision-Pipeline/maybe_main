from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from adverse_vision.models.autoencoder import UNetAutoencoder
from adverse_vision.utils.timing import latency_summary


def _load_model(weights_path: str | Path, device: str = "cpu") -> UNetAutoencoder:
    model = UNetAutoencoder().to(device).eval()
    checkpoint = torch.load(Path(weights_path), map_location=device)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    return model


def export_to_onnx(
    weights_path: str | Path,
    output_path: str | Path,
    img_size: tuple[int, int] = (640, 384),
    opset: int = 17,
) -> str:
    model = _load_model(weights_path, device="cpu")
    dummy = torch.randn(1, 3, img_size[1], img_size[0], dtype=torch.float32)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        output_path.as_posix(),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )
    return output_path.as_posix()


def benchmark_onnx_cpu(
    onnx_path: str | Path,
    num_iters: int = 50,
    img_size: tuple[int, int] = (640, 384),
) -> dict[str, float]:
    try:
        import onnxruntime as ort
    except Exception as exc:
        raise RuntimeError("onnxruntime is required for ONNX benchmarking.") from exc

    session = ort.InferenceSession(Path(onnx_path).as_posix(), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    sample = np.random.rand(1, 3, img_size[1], img_size[0]).astype(np.float32)
    latencies = []
    for _ in range(num_iters):
        start = time.perf_counter()
        _ = session.run(None, {input_name: sample})
        latencies.append((time.perf_counter() - start) * 1000.0)
    return latency_summary(latencies)

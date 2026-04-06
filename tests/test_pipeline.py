from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from adverse_vision.data.corruption import CorruptionConfig
from adverse_vision.inference.pipeline import RestorationPipeline, restore_frame, run_pipeline
from adverse_vision.models.autoencoder import UNetAutoencoder


def test_restore_and_pipeline_interfaces(tmp_path: Path) -> None:
    model = UNetAutoencoder()
    ckpt_path = tmp_path / "ckpt.pt"
    torch.save({"model_state_dict": model.state_dict()}, ckpt_path)

    pipe = RestorationPipeline(weights_path=ckpt_path, enable_detector=False, img_size=(96, 64))
    frame = np.random.randint(0, 255, (128, 192, 3), dtype=np.uint8)

    restored = restore_frame(frame, pipeline=pipe)
    assert restored.shape == frame.shape
    assert restored.dtype == np.uint8

    result = run_pipeline(frame, pipeline=pipe, corruption_config=CorruptionConfig(rain_level=2, seed=4))
    assert "restored_frame" in result
    assert "detections" in result
    assert "timings_ms" in result
    assert "corruption_metadata" in result
    assert result["restored_frame"].shape == frame.shape

from __future__ import annotations

import torch

from adverse_vision.models.autoencoder import UNetAutoencoder


def test_autoencoder_output_shape_and_range() -> None:
    model = UNetAutoencoder()
    x = torch.rand(2, 3, 128, 192)
    with torch.no_grad():
        y = model(x)
    assert y.shape == x.shape
    assert float(y.min()) >= 0.0
    assert float(y.max()) <= 1.0

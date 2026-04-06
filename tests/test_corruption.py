from __future__ import annotations

import numpy as np

from adverse_vision.data.corruption import CorruptionConfig, apply_corruption


def test_corruption_is_deterministic_for_same_seed() -> None:
    image = np.full((96, 128, 3), 140, dtype=np.uint8)
    config = CorruptionConfig(rain_level=3, fog_level=2, dark_level=1, combined_mode="none", seed=123)
    out1, meta1 = apply_corruption(image, config)
    out2, meta2 = apply_corruption(image, config)
    assert np.array_equal(out1, out2)
    assert meta1 == meta2


def test_higher_fog_severity_increases_average_distortion() -> None:
    image = np.full((96, 128, 3), 180, dtype=np.uint8)
    low, _ = apply_corruption(image, CorruptionConfig(fog_level=1, seed=7))
    high, _ = apply_corruption(image, CorruptionConfig(fog_level=5, seed=7))
    low_diff = np.mean(np.abs(low.astype(np.float32) - image.astype(np.float32)))
    high_diff = np.mean(np.abs(high.astype(np.float32) - image.astype(np.float32)))
    assert high_diff > low_diff


def test_higher_dark_severity_increases_average_distortion() -> None:
    image = np.full((96, 128, 3), 180, dtype=np.uint8)
    low, _ = apply_corruption(image, CorruptionConfig(dark_level=1, seed=19))
    high, _ = apply_corruption(image, CorruptionConfig(dark_level=5, seed=19))
    low_diff = np.mean(np.abs(low.astype(np.float32) - image.astype(np.float32)))
    high_diff = np.mean(np.abs(high.astype(np.float32) - image.astype(np.float32)))
    assert high_diff > low_diff

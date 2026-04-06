from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import cv2
import numpy as np

CombinedMode = Literal["none", "rain_fog", "fog_dark"]


@dataclass(frozen=True)
class CorruptionConfig:
    rain_level: int = 0
    fog_level: int = 0
    dark_level: int = 0
    combined_mode: CombinedMode = "none"
    seed: int = 42

    def validated(self) -> "CorruptionConfig":
        for name in ("rain_level", "fog_level", "dark_level"):
            value = getattr(self, name)
            if not 0 <= value <= 5:
                raise ValueError(f"{name} must be in [0, 5], got {value}")
        if self.combined_mode not in {"none", "rain_fog", "fog_dark"}:
            raise ValueError(f"Unsupported combined_mode: {self.combined_mode}")
        return self


def build_deterministic_config(index: int, seed: int = 42) -> CorruptionConfig:
    rng = np.random.default_rng(seed + index * 9973)
    mode = rng.choice(["rain", "fog", "dark", "rain_fog", "fog_dark"])
    rain = fog = dark = 0
    combined_mode: CombinedMode = "none"

    if mode == "rain":
        rain = int(rng.integers(1, 6))
    elif mode == "fog":
        fog = int(rng.integers(1, 6))
    elif mode == "dark":
        dark = int(rng.integers(1, 6))
    elif mode == "rain_fog":
        rain = int(rng.integers(1, 6))
        fog = int(rng.integers(1, 6))
        combined_mode = "rain_fog"
    else:
        fog = int(rng.integers(1, 6))
        dark = int(rng.integers(1, 6))
        combined_mode = "fog_dark"

    return CorruptionConfig(
        rain_level=rain,
        fog_level=fog,
        dark_level=dark,
        combined_mode=combined_mode,
        seed=seed + index * 9973,
    )


def _apply_rain(image: np.ndarray, level: int, rng: np.random.Generator) -> np.ndarray:
    if level <= 0:
        return image
    h, w = image.shape[:2]
    overlay = np.zeros((h, w), dtype=np.float32)

    density = int((0.001 + 0.0015 * level) * h * w)
    streak_length = int(8 + level * 4)
    thickness = 1 if level < 4 else 2
    angle = rng.uniform(-25, 25)
    dx = int(streak_length * np.sin(np.deg2rad(angle)))
    dy = int(streak_length * np.cos(np.deg2rad(angle)))

    for _ in range(max(1, density)):
        x = int(rng.integers(0, w))
        y = int(rng.integers(0, h))
        x2 = np.clip(x + dx, 0, w - 1)
        y2 = np.clip(y + dy, 0, h - 1)
        cv2.line(overlay, (x, y), (int(x2), int(y2)), color=255, thickness=thickness)

    ksize = int(3 + 2 * level)
    overlay = cv2.GaussianBlur(overlay, (ksize | 1, ksize | 1), sigmaX=0)
    overlay_rgb = np.repeat((overlay / 255.0)[..., None], 3, axis=2)
    base = image.astype(np.float32) / 255.0
    strength = 0.1 + 0.08 * level
    rainy = np.clip(base * (1.0 - 0.05 * level) + overlay_rgb * strength, 0.0, 1.0)
    return (rainy * 255.0).astype(np.uint8)


def _apply_fog(image: np.ndarray, level: int, rng: np.random.Generator) -> np.ndarray:
    if level <= 0:
        return image
    h, w = image.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx = w / 2.0 + float(rng.uniform(-0.2, 0.2) * w)
    cy = h / 2.0 + float(rng.uniform(-0.2, 0.2) * h)
    radial_dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    radial_norm = radial_dist / (np.sqrt(cx * cx + cy * cy) + 1e-6)
    depth = np.clip(0.3 + 0.7 * radial_norm, 0.0, 1.0)
    beta = 0.4 + 0.25 * level
    transmittance = np.exp(-beta * depth)[..., None]
    atmospheric_light = (0.85 + 0.03 * level) * np.ones((1, 1, 3), dtype=np.float32)
    img = image.astype(np.float32) / 255.0
    fogged = img * transmittance + atmospheric_light * (1.0 - transmittance)
    fogged = cv2.GaussianBlur(fogged, (0, 0), sigmaX=0.8 + level * 0.5)
    return np.clip(fogged * 255.0, 0, 255).astype(np.uint8)


def _apply_dark(image: np.ndarray, level: int, rng: np.random.Generator) -> np.ndarray:
    if level <= 0:
        return image
    img = image.astype(np.float32) / 255.0
    gamma = 1.1 + 0.45 * level
    dark = np.power(np.clip(img, 1e-6, 1.0), gamma)

    shot_scale = 35.0 + 10.0 * level
    poisson = rng.poisson(np.clip(dark * shot_scale, 0.0, shot_scale)) / shot_scale
    gaussian_std = 0.005 + 0.006 * level
    gaussian = rng.normal(loc=0.0, scale=gaussian_std, size=dark.shape)
    noisy = np.clip(poisson + gaussian, 0.0, 1.0)

    return np.clip(noisy * 255.0, 0, 255).astype(np.uint8)


def apply_corruption(image: np.ndarray, config: CorruptionConfig) -> tuple[np.ndarray, dict]:
    config = config.validated()
    rng = np.random.default_rng(config.seed)
    output = image.copy()

    if config.combined_mode == "rain_fog":
        output = _apply_rain(output, config.rain_level, rng)
        output = _apply_fog(output, config.fog_level, rng)
    elif config.combined_mode == "fog_dark":
        output = _apply_fog(output, config.fog_level, rng)
        output = _apply_dark(output, config.dark_level, rng)
    else:
        output = _apply_rain(output, config.rain_level, rng)
        output = _apply_fog(output, config.fog_level, rng)
        output = _apply_dark(output, config.dark_level, rng)

    metadata = asdict(config)
    metadata["corruption_tags"] = [
        name for name, lvl in [("rain", config.rain_level), ("fog", config.fog_level), ("dark", config.dark_level)] if lvl > 0
    ]
    return output, metadata

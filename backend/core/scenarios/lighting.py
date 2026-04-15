import cv2
import numpy as np
from backend.core.scenarios.base import BaseScenario, register_scenario


@register_scenario
class LowLightScenario(BaseScenario):
    name = "low_light"
    description = "Simulate low-light / underexposed conditions."

    def apply(self, image: np.ndarray) -> np.ndarray:
        gamma = self.params.get("gamma", 2.5)   # >1 darkens (power law)
        table = np.array([
            (i / 255.0) ** gamma * 255
            for i in range(256)
        ]).astype(np.uint8)
        return cv2.LUT(image, table)


@register_scenario
class OverexposureScenario(BaseScenario):
    name = "overexposure"
    description = "Simulate overexposed / high-brightness conditions."

    def apply(self, image: np.ndarray) -> np.ndarray:
        factor = self.params.get("factor", 1.8)
        out = np.clip(image.astype(np.float32) * factor, 0, 255)
        return out.astype(np.uint8)


@register_scenario
class ShadowScenario(BaseScenario):
    name = "shadow"
    description = "Add a random shadow region to simulate partial lighting obstruction."

    def apply(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        darkness = self.params.get("darkness", 0.4)

        x1, x2 = np.random.randint(0, w, 2)
        shadow_mask = np.zeros((h, w), dtype=np.float32)
        for row in range(h):
            col_boundary = int(x1 + (x2 - x1) * row / h)
            shadow_mask[row, :col_boundary] = 1

        out = image.astype(np.float32)
        out[shadow_mask == 1] *= darkness
        return np.clip(out, 0, 255).astype(np.uint8)


@register_scenario
class FogScenario(BaseScenario):
    name = "fog"
    description = "Simulate foggy / hazy weather conditions."

    def apply(self, image: np.ndarray) -> np.ndarray:
        intensity = self.params.get("intensity", 0.4)
        fog_color = np.ones_like(image, dtype=np.float32) * 220
        out = (
            image.astype(np.float32) * (1 - intensity) + fog_color * intensity
        )
        return np.clip(out, 0, 255).astype(np.uint8)

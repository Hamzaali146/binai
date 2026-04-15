import cv2
import numpy as np
from backend.core.scenarios.base import BaseScenario, register_scenario


@register_scenario
class GaussianNoiseScenario(BaseScenario):
    name = "gaussian_noise"
    description = "Add Gaussian random noise to simulate sensor noise."

    def apply(self, image: np.ndarray) -> np.ndarray:
        mean = self.params.get("mean", 0)
        std = self.params.get("std", 25)
        noise = np.random.normal(mean, std, image.shape).astype(np.float32)
        noisy = np.clip(image.astype(np.float32) + noise, 0, 255)
        return noisy.astype(np.uint8)


@register_scenario
class SaltPepperNoiseScenario(BaseScenario):
    name = "salt_pepper_noise"
    description = "Add salt-and-pepper noise to simulate transmission errors."

    def apply(self, image: np.ndarray) -> np.ndarray:
        density = self.params.get("density", 0.05)
        out = image.copy()
        total = image.size // image.shape[2] if len(image.shape) == 3 else image.size
        num_salt = int(density * total * 0.5)
        num_pepper = int(density * total * 0.5)
        h, w = image.shape[:2]
        coords = [np.random.randint(0, i, num_salt) for i in [h, w]]
        out[coords[0], coords[1]] = 255
        coords = [np.random.randint(0, i, num_pepper) for i in [h, w]]
        out[coords[0], coords[1]] = 0
        return out


@register_scenario
class SpeckleNoiseScenario(BaseScenario):
    name = "speckle_noise"
    description = "Add multiplicative speckle noise (common in radar/ultrasound images)."

    def apply(self, image: np.ndarray) -> np.ndarray:
        intensity = self.params.get("intensity", 0.2)
        noise = np.random.randn(*image.shape) * intensity
        noisy = np.clip(image.astype(np.float32) * (1 + noise), 0, 255)
        return noisy.astype(np.uint8)

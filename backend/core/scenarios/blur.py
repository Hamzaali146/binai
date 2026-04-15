import cv2
import numpy as np
from backend.core.scenarios.base import BaseScenario, register_scenario


@register_scenario
class GaussianBlurScenario(BaseScenario):
    name = "gaussian_blur"
    description = "Apply Gaussian blur to simulate camera defocus or motion."

    def apply(self, image: np.ndarray) -> np.ndarray:
        kernel_size = self.params.get("kernel_size", 15)
        sigma = self.params.get("sigma", 0)
        if kernel_size % 2 == 0:
            kernel_size += 1
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)


@register_scenario
class MotionBlurScenario(BaseScenario):
    name = "motion_blur"
    description = "Apply directional motion blur to simulate camera shake."

    def apply(self, image: np.ndarray) -> np.ndarray:
        kernel_size = self.params.get("kernel_size", 15)
        angle = self.params.get("angle", 0)   # degrees
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[kernel_size // 2, :] = 1.0 / kernel_size
        rotation_mat = cv2.getRotationMatrix2D(
            (kernel_size // 2, kernel_size // 2), angle, 1
        )
        kernel = cv2.warpAffine(kernel, rotation_mat, (kernel_size, kernel_size))
        kernel /= kernel.sum() + 1e-8
        return cv2.filter2D(image, -1, kernel)


@register_scenario
class MedianBlurScenario(BaseScenario):
    name = "median_blur"
    description = "Apply median blur to simulate sensor noise smoothing."

    def apply(self, image: np.ndarray) -> np.ndarray:
        ksize = self.params.get("ksize", 5)
        if ksize % 2 == 0:
            ksize += 1
        return cv2.medianBlur(image, ksize)

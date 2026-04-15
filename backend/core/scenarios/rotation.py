import cv2
import numpy as np
from backend.core.scenarios.base import BaseScenario, register_scenario


@register_scenario
class RotationScenario(BaseScenario):
    name = "rotation"
    description = "Rotate image by a fixed angle to test orientation robustness."

    def apply(self, image: np.ndarray) -> np.ndarray:
        angle = self.params.get("angle", 15)
        scale = self.params.get("scale", 1.0)
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
        return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT)


@register_scenario
class FlipScenario(BaseScenario):
    name = "flip"
    description = "Horizontally or vertically flip the image."

    def apply(self, image: np.ndarray) -> np.ndarray:
        flip_code = self.params.get("flip_code", 1)   # 1=horizontal, 0=vertical, -1=both
        return cv2.flip(image, flip_code)


@register_scenario
class PerspectiveScenario(BaseScenario):
    name = "perspective"
    description = "Apply a random perspective warp to simulate camera angle change."

    def apply(self, image: np.ndarray) -> np.ndarray:
        distortion = self.params.get("distortion", 0.05)
        h, w = image.shape[:2]
        d = int(min(h, w) * distortion)
        src = np.float32([
            [0, 0], [w, 0], [w, h], [0, h]
        ])
        dst = np.float32([
            [np.random.randint(0, d), np.random.randint(0, d)],
            [w - np.random.randint(0, d), np.random.randint(0, d)],
            [w - np.random.randint(0, d), h - np.random.randint(0, d)],
            [np.random.randint(0, d), h - np.random.randint(0, d)],
        ])
        M = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(image, M, (w, h), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REFLECT)


@register_scenario
class ZoomScenario(BaseScenario):
    name = "zoom"
    description = "Simulate digital zoom by cropping and upscaling."

    def apply(self, image: np.ndarray) -> np.ndarray:
        zoom_factor = self.params.get("zoom_factor", 1.5)
        h, w = image.shape[:2]
        crop_h = int(h / zoom_factor)
        crop_w = int(w / zoom_factor)
        y1 = (h - crop_h) // 2
        x1 = (w - crop_w) // 2
        cropped = image[y1:y1 + crop_h, x1:x1 + crop_w]
        return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

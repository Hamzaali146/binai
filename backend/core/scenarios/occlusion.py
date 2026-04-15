import cv2
import numpy as np
from backend.core.scenarios.base import BaseScenario, register_scenario


@register_scenario
class RandomOcclusionScenario(BaseScenario):
    name = "random_occlusion"
    description = "Place random black rectangles to simulate partial occlusion."

    def apply(self, image: np.ndarray) -> np.ndarray:
        num_patches = self.params.get("num_patches", 3)
        patch_ratio = self.params.get("patch_ratio", 0.15)
        fill_value = self.params.get("fill_value", 0)

        h, w = image.shape[:2]
        out = image.copy()
        ph = int(h * patch_ratio)
        pw = int(w * patch_ratio)

        for _ in range(num_patches):
            y = np.random.randint(0, max(1, h - ph))
            x = np.random.randint(0, max(1, w - pw))
            out[y:y + ph, x:x + pw] = fill_value
        return out


@register_scenario
class GridOcclusionScenario(BaseScenario):
    name = "grid_occlusion"
    description = "Overlay a regular grid mask to simulate structured occlusion."

    def apply(self, image: np.ndarray) -> np.ndarray:
        num_tiles = self.params.get("num_tiles", 5)
        drop_ratio = self.params.get("drop_ratio", 0.3)

        h, w = image.shape[:2]
        out = image.copy()
        th = h // num_tiles
        tw = w // num_tiles

        for i in range(num_tiles):
            for j in range(num_tiles):
                if np.random.rand() < drop_ratio:
                    y1, y2 = i * th, (i + 1) * th
                    x1, x2 = j * tw, (j + 1) * tw
                    out[y1:y2, x1:x2] = 0
        return out


@register_scenario
class WatermarkOcclusionScenario(BaseScenario):
    name = "watermark_occlusion"
    description = "Overlay a semi-transparent text watermark to simulate branding overlays."

    def apply(self, image: np.ndarray) -> np.ndarray:
        text = self.params.get("text", "TEST")
        alpha = self.params.get("alpha", 0.5)
        font_scale = self.params.get("font_scale", 2.0)

        h, w = image.shape[:2]
        overlay = image.copy()
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 3)
        tx = (w - text_size[0]) // 2
        ty = (h + text_size[1]) // 2
        cv2.putText(overlay, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (128, 128, 128), 3, cv2.LINE_AA)
        return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

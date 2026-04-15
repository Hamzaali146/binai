"""
binai.scenarios
----------------
Robustness scenario transforms — apply individually or pass to EvaluationEngine.

Available scenarios
-------------------
Blur:       gaussian_blur, motion_blur, median_blur
Noise:      gaussian_noise, salt_pepper_noise, speckle_noise
Lighting:   low_light, overexposure, shadow, fog
Occlusion:  random_occlusion, grid_occlusion, watermark_occlusion
Geometry:   rotation, flip, perspective, zoom

Examples
--------
>>> from binai.scenarios import get_scenario, list_scenarios

>>> # Apply a single scenario to an image
>>> transform = get_scenario("gaussian_blur", kernel_size=15)
>>> blurred = transform.apply(image)

>>> # List everything available
>>> for sc in list_scenarios():
...     print(sc["name"], "—", sc["description"])
"""

from backend.core.scenarios import (
    get_scenario,
    list_scenarios,
    SCENARIO_REGISTRY,
    BaseScenario,
)

# Also expose individual scenario classes for subclassing / type hints
from backend.core.scenarios.blur import (
    GaussianBlurScenario,
    MotionBlurScenario,
    MedianBlurScenario,
)
from backend.core.scenarios.noise import (
    GaussianNoiseScenario,
    SaltPepperNoiseScenario,
    SpeckleNoiseScenario,
)
from backend.core.scenarios.lighting import (
    LowLightScenario,
    OverexposureScenario,
    ShadowScenario,
    FogScenario,
)
from backend.core.scenarios.occlusion import (
    RandomOcclusionScenario,
    GridOcclusionScenario,
    WatermarkOcclusionScenario,
)
from backend.core.scenarios.rotation import (
    RotationScenario,
    FlipScenario,
    PerspectiveScenario,
    ZoomScenario,
)

__all__ = [
    "get_scenario",
    "list_scenarios",
    "SCENARIO_REGISTRY",
    "BaseScenario",
    # Blur
    "GaussianBlurScenario",
    "MotionBlurScenario",
    "MedianBlurScenario",
    # Noise
    "GaussianNoiseScenario",
    "SaltPepperNoiseScenario",
    "SpeckleNoiseScenario",
    # Lighting
    "LowLightScenario",
    "OverexposureScenario",
    "ShadowScenario",
    "FogScenario",
    # Occlusion
    "RandomOcclusionScenario",
    "GridOcclusionScenario",
    "WatermarkOcclusionScenario",
    # Geometry
    "RotationScenario",
    "FlipScenario",
    "PerspectiveScenario",
    "ZoomScenario",
]

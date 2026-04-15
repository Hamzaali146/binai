import numpy as np
import pytest
from backend.core.scenarios import get_scenario, list_scenarios, SCENARIO_REGISTRY


@pytest.fixture
def img():
    return np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)


class TestScenarioRegistry:
    def test_all_scenarios_registered(self):
        scenarios = list_scenarios()
        names = [s["name"] for s in scenarios]
        expected = [
            "gaussian_blur", "motion_blur", "median_blur",
            "gaussian_noise", "salt_pepper_noise", "speckle_noise",
            "low_light", "overexposure", "shadow", "fog",
            "random_occlusion", "grid_occlusion", "watermark_occlusion",
            "rotation", "flip", "perspective", "zoom",
        ]
        for name in expected:
            assert name in names, f"Scenario '{name}' not registered"

    def test_unknown_scenario_raises(self):
        with pytest.raises(ValueError, match="Unknown scenario"):
            get_scenario("nonexistent_scenario_xyz")


class TestBlurScenarios:
    def test_gaussian_blur_output_shape(self, img):
        out = get_scenario("gaussian_blur", kernel_size=7).apply(img)
        assert out.shape == img.shape

    def test_motion_blur_output_shape(self, img):
        out = get_scenario("motion_blur", kernel_size=11, angle=45).apply(img)
        assert out.shape == img.shape

    def test_blur_changes_image(self, img):
        out = get_scenario("gaussian_blur", kernel_size=15).apply(img)
        assert not np.array_equal(out, img)


class TestNoiseScenarios:
    def test_gaussian_noise_dtype(self, img):
        out = get_scenario("gaussian_noise", std=40).apply(img)
        assert out.dtype == np.uint8
        assert out.shape == img.shape

    def test_salt_pepper_range(self, img):
        out = get_scenario("salt_pepper_noise", density=0.1).apply(img)
        assert out.min() >= 0 and out.max() <= 255


class TestLightingScenarios:
    def test_low_light_darkens(self):
        bright = np.full((128, 128, 3), 200, dtype=np.uint8)
        out = get_scenario("low_light", gamma=3.0).apply(bright)
        assert float(out.mean()) < float(bright.mean())

    def test_overexposure_brightens(self, img):
        dark = (img * 0.3).astype(np.uint8)
        out = get_scenario("overexposure", factor=2.0).apply(dark)
        assert float(out.mean()) >= float(dark.mean())

    def test_fog_output_shape(self, img):
        out = get_scenario("fog", intensity=0.5).apply(img)
        assert out.shape == img.shape


class TestOcclusionScenarios:
    def test_random_occlusion_has_black_pixels(self, img):
        white = np.ones((128, 128, 3), dtype=np.uint8) * 200
        out = get_scenario("random_occlusion", num_patches=3, fill_value=0).apply(white)
        assert (out == 0).any()

    def test_grid_occlusion_shape(self, img):
        out = get_scenario("grid_occlusion").apply(img)
        assert out.shape == img.shape


class TestRotationScenarios:
    def test_rotation_preserves_shape(self, img):
        out = get_scenario("rotation", angle=30).apply(img)
        assert out.shape == img.shape

    def test_flip_horizontal(self, img):
        out = get_scenario("flip", flip_code=1).apply(img)
        assert out.shape == img.shape
        assert np.array_equal(out, img[:, ::-1])

    def test_zoom_preserves_shape(self, img):
        out = get_scenario("zoom", zoom_factor=2.0).apply(img)
        assert out.shape == img.shape

# binai — Automated CV Testing Framework

Production-grade platform to **test, evaluate, and monitor** computer vision models across multiple real-world scenarios.

## Install

```bash
# Core library only (metrics + scenarios, no server)
pip install binai

# With REST API server
pip install "binai[server]"

# With analytics dashboard
pip install "binai[dashboard]"

# Everything
pip install "binai[all]"
```

## Quick Start

```python
from binai import EvaluationEngine
from binai.scenarios import get_scenario
from binai.metrics import compute_classification_metrics

# 1. Use metrics standalone
metrics = compute_classification_metrics(
    y_true=[0, 1, 2, 0, 1],
    y_pred=[0, 1, 0, 0, 1],
    class_names=["cat", "dog", "bird"],
)
print(metrics["accuracy"])   # 0.8
print(metrics["f1_score"])   # 0.8

# 2. Apply a scenario to any image
import numpy as np
image = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)

blurred   = get_scenario("gaussian_blur", kernel_size=15).apply(image)
noisy     = get_scenario("gaussian_noise", std=30).apply(image)
dark      = get_scenario("low_light", gamma=3.0).apply(image)
occluded  = get_scenario("random_occlusion", num_patches=3).apply(image)

# 3. Run a full evaluation with robustness testing
engine = EvaluationEngine(
    task_type="classification",       # or "detection" / "segmentation"
    dataset_path="/data/my_dataset",  # folder with annotations.json
    predict_fn=my_model.predict,      # your model's inference function
    class_names=["cat", "dog", "bird"],
)

results = engine.run(config={
    "confidence_threshold": 0.5,
    "max_samples": 500,
    "scenarios": [
        {"name": "gaussian_blur",    "params": {"kernel_size": 11}},
        {"name": "low_light",        "params": {"gamma": 3.0}},
        {"name": "random_occlusion", "params": {"num_patches": 2}},
        {"name": "rotation",         "params": {"angle": 20}},
    ],
})

print(results["baseline_metrics"]["accuracy"])   # e.g. 0.91
for sc in results["scenario_results"]:
    print(sc["scenario_name"], "degradation:", sc["degradation_pct"], "%")
```

## CLI

```bash
# Show version + installed components
binai info

# List all 17 scenarios
binai scenarios

# Run evaluation from terminal
binai evaluate \
  --dataset ./data/my_dataset \
  --task classification \
  --scenarios gaussian_blur low_light rotation \
  --report html

# Start REST API  (http://localhost:8000/docs)
binai server

# Start dashboard (http://localhost:8501)
binai dashboard
```

## Supported scenarios (17 total)

| Category  | Scenarios |
|-----------|-----------|
| Blur      | `gaussian_blur`, `motion_blur`, `median_blur` |
| Noise     | `gaussian_noise`, `salt_pepper_noise`, `speckle_noise` |
| Lighting  | `low_light`, `overexposure`, `shadow`, `fog` |
| Occlusion | `random_occlusion`, `grid_occlusion`, `watermark_occlusion` |
| Geometry  | `rotation`, `flip`, `perspective`, `zoom` |

## Metrics

| Task           | Metrics |
|----------------|---------|
| Classification | Accuracy, Precision, Recall, F1, AUC, Top-5, per-class |
| Detection      | mAP@50, mAP@75, mAP@50:95, per-class AP |
| Segmentation   | Mean IoU, Pixel Accuracy, per-class IoU |

## Custom scenario

```python
from binai.scenarios import BaseScenario, SCENARIO_REGISTRY
import numpy as np

class RainScenario(BaseScenario):
    name = "rain"
    description = "Simulate rain streaks."

    def apply(self, image: np.ndarray) -> np.ndarray:
        intensity = self.params.get("intensity", 0.3)
        out = image.copy()
        h, w = out.shape[:2]
        for _ in range(int(h * w * intensity * 0.01)):
            x = np.random.randint(0, w)
            y1 = np.random.randint(0, h - 20)
            out[y1:y1+20, x] = 200
        return out

# Register it so it works everywhere (CLI, API, engine)
SCENARIO_REGISTRY["rain"] = RainScenario

# Use it
transform = RainScenario(intensity=0.5)
result = transform.apply(image)
```

## Dataset format

Create a folder with an `annotations.json` file:

**Classification:**
```json
{
  "classes": ["cat", "dog", "bird"],
  "samples": [
    {"id": "img_001", "image_path": "/data/images/001.jpg", "class_id": 0},
    {"id": "img_002", "image_path": "/data/images/002.jpg", "class_id": 1}
  ]
}
```

**Detection:**
```json
{
  "samples": [
    {
      "id": "img_001",
      "image_path": "/data/images/001.jpg",
      "boxes": [
        {"class_id": 0, "bbox": [10, 20, 100, 150]},
        {"class_id": 1, "bbox": [200, 50, 350, 300]}
      ]
    }
  ]
}
```

## Publish to PyPI

```bash
pip install build twine
python -m build
twine upload dist/*
```

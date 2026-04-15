"""
binai — Automated Testing & Evaluation Framework for Computer Vision Models
============================================================================

Quick start
-----------
>>> from binai import EvaluationEngine
>>> from binai.metrics import compute_classification_metrics
>>> from binai.scenarios import get_scenario, list_scenarios

Evaluate a model in 3 lines:

>>> engine = EvaluationEngine(
...     task_type="classification",
...     dataset_path="/data/my_dataset",
...     predict_fn=my_model.predict,
...     class_names=["cat", "dog"],
... )
>>> results = engine.run(config={"scenarios": [{"name": "gaussian_blur"}]})
>>> print(results["baseline_metrics"])
"""

from binai._version import __version__
from binai.engine import EvaluationEngine
from binai import metrics, scenarios, reporting

__all__ = [
    "__version__",
    "EvaluationEngine",
    "metrics",
    "scenarios",
    "reporting",
]

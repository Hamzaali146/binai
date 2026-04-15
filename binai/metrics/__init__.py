"""
binai.metrics
--------------
Standalone metric functions — use them directly without running a full evaluation.

Examples
--------
>>> from binai.metrics import compute_classification_metrics
>>> metrics = compute_classification_metrics(y_true=[0,1,2], y_pred=[0,1,0])
>>> print(metrics["accuracy"], metrics["f1_score"])

>>> from binai.metrics import compute_detection_metrics
>>> metrics = compute_detection_metrics(gt_boxes_per_image, pred_boxes_per_image)
>>> print(metrics["map_50"])

>>> from binai.metrics import compute_segmentation_metrics
>>> metrics = compute_segmentation_metrics(gt_masks, pred_masks, num_classes=3)
>>> print(metrics["mean_iou"])
"""

from backend.core.metrics.classification import (
    compute_classification_metrics,
    find_classification_failures,
)
from backend.core.metrics.detection import (
    compute_detection_metrics,
    find_detection_failures,
    compute_iou,
)
from backend.core.metrics.segmentation import (
    compute_segmentation_metrics,
    find_segmentation_failures,
)

__all__ = [
    # Classification
    "compute_classification_metrics",
    "find_classification_failures",
    # Detection
    "compute_detection_metrics",
    "find_detection_failures",
    "compute_iou",
    # Segmentation
    "compute_segmentation_metrics",
    "find_segmentation_failures",
]

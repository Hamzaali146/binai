import numpy as np
import pytest
from backend.core.metrics.classification import (
    compute_classification_metrics, find_classification_failures
)
from backend.core.metrics.detection import compute_detection_metrics, compute_iou
from backend.core.metrics.segmentation import compute_segmentation_metrics


class TestClassificationMetrics:
    def test_perfect_accuracy(self):
        y = list(range(10))
        m = compute_classification_metrics(y, y, class_names=["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"])
        assert m["accuracy"] == 1.0
        assert m["f1_score"] == pytest.approx(1.0, abs=1e-4)

    def test_zero_accuracy(self):
        y_true = [0] * 10
        y_pred = [1] * 10
        m = compute_classification_metrics(y_true, y_pred, class_names=["a", "b"])
        assert m["accuracy"] == 0.0

    def test_partial_accuracy(self, sample_classification_data):
        ids, y_true, y_pred, y_prob = sample_classification_data
        m = compute_classification_metrics(y_true, y_pred, y_prob)
        assert 0.0 <= m["accuracy"] <= 1.0
        assert "per_class_metrics" in m
        assert "confusion_matrix" in m

    def test_failure_detection(self):
        ids = ["a", "b", "c"]
        y_true = [0, 1, 2]
        y_pred = [0, 0, 0]
        failures = find_classification_failures(ids, y_true, y_pred)
        assert len(failures) == 2
        assert all(f["error_type"] == "wrong_class" for f in failures)

    def test_no_failures_when_perfect(self):
        ids = ["a", "b"]
        y_true = [0, 1]
        failures = find_classification_failures(ids, y_true, y_true)
        assert len(failures) == 0


class TestDetectionMetrics:
    def test_iou_perfect_overlap(self):
        box = [10, 10, 50, 50]
        assert compute_iou(box, box) == pytest.approx(1.0)

    def test_iou_no_overlap(self):
        assert compute_iou([0, 0, 10, 10], [20, 20, 30, 30]) == pytest.approx(0.0)

    def test_iou_partial_overlap(self):
        iou = compute_iou([0, 0, 20, 20], [10, 10, 30, 30])
        assert 0.0 < iou < 1.0

    def test_map_perfect_predictions(self, sample_detection_data):
        ids, gt_boxes, pred_boxes = sample_detection_data
        m = compute_detection_metrics(gt_boxes, pred_boxes, class_names=["a", "b"])
        assert m["map_50"] > 0.8

    def test_map_zero_predictions(self, sample_detection_data):
        _, gt_boxes, _ = sample_detection_data
        empty_preds = [[] for _ in gt_boxes]
        m = compute_detection_metrics(gt_boxes, empty_preds)
        assert m["map_50"] == pytest.approx(0.0)

    def test_metrics_structure(self, sample_detection_data):
        _, gt_boxes, pred_boxes = sample_detection_data
        m = compute_detection_metrics(gt_boxes, pred_boxes)
        assert "map_50" in m
        assert "map_75" in m
        assert "map_50_95" in m
        assert "per_class_metrics" in m


class TestSegmentationMetrics:
    def test_perfect_masks(self):
        masks = [np.zeros((64, 64), dtype=np.uint8) for _ in range(5)]
        m = compute_segmentation_metrics(masks, masks, num_classes=2)
        assert m["pixel_accuracy"] == pytest.approx(1.0)
        assert m["mean_iou"] >= 0.0

    def test_pixel_accuracy_range(self):
        gt = [np.random.randint(0, 2, (64, 64), dtype=np.uint8) for _ in range(5)]
        pred = [np.random.randint(0, 2, (64, 64), dtype=np.uint8) for _ in range(5)]
        m = compute_segmentation_metrics(gt, pred, num_classes=2)
        assert 0.0 <= m["pixel_accuracy"] <= 1.0
        assert 0.0 <= m["mean_iou"] <= 1.0

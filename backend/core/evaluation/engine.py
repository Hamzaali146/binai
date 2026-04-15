"""
Central evaluation engine — orchestrates dataset loading, scenario application,
metric computation, and failure logging for all task types.
"""
from __future__ import annotations

import uuid
import time
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import cv2

from backend.config import settings
from backend.core.scenarios import get_scenario
from backend.core.metrics.classification import (
    compute_classification_metrics, find_classification_failures
)
from backend.core.metrics.detection import (
    compute_detection_metrics, find_detection_failures
)
from backend.core.metrics.segmentation import (
    compute_segmentation_metrics, find_segmentation_failures
)

logger = logging.getLogger(__name__)


# ── Dataset loader protocol ───────────────────────────────────────────────────
# Callers must inject a `predict_fn` that accepts a list of images (np.ndarray)
# and returns predictions in the appropriate format for the task type:
#   classification → list[dict{"class_id": int, "scores": list[float]}]
#   detection      → list[list[dict{"class_id": int, "bbox": [x1,y1,x2,y2], "score": float}]]
#   segmentation   → list[np.ndarray]  (H×W class-id masks)


def _load_image(path: str) -> Optional[np.ndarray]:
    img = cv2.imread(path)
    if img is None:
        logger.warning(f"Could not load image: {path}")
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _load_classification_dataset(dataset_path: str) -> tuple[list[str], list[str], list[int]]:
    """
    Expects dataset_path to contain an annotations.json:
    {"samples": [{"id": "...", "image_path": "...", "class_id": 0}], "classes": [...]}
    """
    ann_file = Path(dataset_path) / "annotations.json"
    with open(ann_file) as f:
        data = json.load(f)
    ids = [s["id"] for s in data["samples"]]
    paths = [s["image_path"] for s in data["samples"]]
    labels = [s["class_id"] for s in data["samples"]]
    return ids, paths, labels


def _load_detection_dataset(dataset_path: str) -> tuple[list[str], list[str], list[list[dict]]]:
    """
    annotations.json: {"samples": [{"id","image_path","boxes":[{"class_id","bbox":[x1,y1,x2,y2]}]}]}
    """
    ann_file = Path(dataset_path) / "annotations.json"
    with open(ann_file) as f:
        data = json.load(f)
    ids = [s["id"] for s in data["samples"]]
    paths = [s["image_path"] for s in data["samples"]]
    boxes = [s["boxes"] for s in data["samples"]]
    return ids, paths, boxes


def _load_segmentation_dataset(dataset_path: str) -> tuple[list[str], list[str], list[str]]:
    """
    annotations.json: {"samples": [{"id","image_path","mask_path":"..."}]}
    """
    ann_file = Path(dataset_path) / "annotations.json"
    with open(ann_file) as f:
        data = json.load(f)
    ids = [s["id"] for s in data["samples"]]
    paths = [s["image_path"] for s in data["samples"]]
    masks = [s["mask_path"] for s in data["samples"]]
    return ids, paths, masks


# ── Core runner ───────────────────────────────────────────────────────────────

class EvaluationEngine:
    def __init__(self, task_type: str, dataset_path: str, predict_fn: Callable,
                 class_names: Optional[list[str]] = None, num_classes: int = 2):
        self.task_type = task_type
        self.dataset_path = dataset_path
        self.predict_fn = predict_fn
        self.class_names = class_names
        self.num_classes = num_classes

    def _run_classification(
        self, image_ids, image_paths, labels,
        scenario_name: Optional[str] = None,
        scenario_transform=None,
        conf_threshold: float = 0.5,
        max_samples: Optional[int] = None,
    ) -> tuple[dict, list[dict]]:
        if max_samples:
            image_ids = image_ids[:max_samples]
            image_paths = image_paths[:max_samples]
            labels = labels[:max_samples]

        images = []
        valid_ids, valid_labels = [], []
        for img_id, img_path, label in zip(image_ids, image_paths, labels):
            img = _load_image(img_path)
            if img is None:
                continue
            if scenario_transform:
                img = scenario_transform.apply(img)
            images.append(img)
            valid_ids.append(img_id)
            valid_labels.append(label)

        if not images:
            return {}, []

        raw_preds = self.predict_fn(images)
        y_pred = [p["class_id"] for p in raw_preds]
        y_prob = [p.get("scores") for p in raw_preds]
        if any(p is None for p in y_prob):
            y_prob = None

        metrics = compute_classification_metrics(
            valid_labels, y_pred, y_prob, self.class_names
        )
        failures = find_classification_failures(
            valid_ids, valid_labels, y_pred, y_prob, self.class_names
        )
        if scenario_name:
            for f in failures:
                f["scenario"] = scenario_name
        return metrics, failures

    def _run_detection(
        self, image_ids, image_paths, gt_boxes,
        scenario_name=None, scenario_transform=None,
        iou_threshold=0.5, conf_threshold=0.5,
        max_samples=None,
    ) -> tuple[dict, list[dict]]:
        if max_samples:
            image_ids = image_ids[:max_samples]
            image_paths = image_paths[:max_samples]
            gt_boxes = gt_boxes[:max_samples]

        images, valid_ids, valid_gts = [], [], []
        for img_id, img_path, gt in zip(image_ids, image_paths, gt_boxes):
            img = _load_image(img_path)
            if img is None:
                continue
            if scenario_transform:
                img = scenario_transform.apply(img)
            images.append(img)
            valid_ids.append(img_id)
            valid_gts.append(gt)

        if not images:
            return {}, []

        pred_boxes = self.predict_fn(images)
        metrics = compute_detection_metrics(
            valid_gts, pred_boxes, self.class_names, iou_thresholds=[iou_threshold, 0.75]
        )
        failures = find_detection_failures(
            valid_ids, valid_gts, pred_boxes, iou_threshold, conf_threshold
        )
        if scenario_name:
            for f in failures:
                f["scenario"] = scenario_name
        return metrics, failures

    def _run_segmentation(
        self, image_ids, image_paths, mask_paths,
        scenario_name=None, scenario_transform=None,
        iou_threshold=0.5, max_samples=None,
    ) -> tuple[dict, list[dict]]:
        if max_samples:
            image_ids = image_ids[:max_samples]
            image_paths = image_paths[:max_samples]
            mask_paths = mask_paths[:max_samples]

        images, gt_masks, valid_ids = [], [], []
        for img_id, img_path, mask_path in zip(image_ids, image_paths, mask_paths):
            img = _load_image(img_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if img is None or mask is None:
                continue
            if scenario_transform:
                img = scenario_transform.apply(img)
            images.append(img)
            gt_masks.append(mask)
            valid_ids.append(img_id)

        if not images:
            return {}, []

        pred_masks = self.predict_fn(images)
        metrics = compute_segmentation_metrics(
            gt_masks, pred_masks, self.num_classes, self.class_names
        )
        failures = find_segmentation_failures(
            valid_ids, gt_masks, pred_masks, iou_threshold, self.num_classes
        )
        if scenario_name:
            for f in failures:
                f["scenario"] = scenario_name
        return metrics, failures

    def run(
        self,
        config: dict,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> dict[str, Any]:
        t0 = time.time()
        result: dict[str, Any] = {
            "baseline_metrics": {},
            "scenario_results": [],
            "all_failures": [],
            "num_samples": 0,
        }

        conf_threshold = config.get("confidence_threshold", 0.5)
        iou_threshold = config.get("iou_threshold", 0.5)
        max_samples = config.get("max_samples")
        scenarios = config.get("scenarios", [])

        # Load dataset
        if progress_callback:
            progress_callback("Loading dataset", 0.05)

        if self.task_type == "classification":
            ids, paths, labels = _load_classification_dataset(self.dataset_path)
            runner = self._run_classification
            runner_kwargs = dict(conf_threshold=conf_threshold, max_samples=max_samples)
            runner_args = (ids, paths, labels)
        elif self.task_type == "detection":
            ids, paths, gt_boxes = _load_detection_dataset(self.dataset_path)
            runner = self._run_detection
            runner_kwargs = dict(iou_threshold=iou_threshold, conf_threshold=conf_threshold,
                                 max_samples=max_samples)
            runner_args = (ids, paths, gt_boxes)
        elif self.task_type == "segmentation":
            ids, paths, mask_paths = _load_segmentation_dataset(self.dataset_path)
            runner = self._run_segmentation
            runner_kwargs = dict(iou_threshold=iou_threshold, max_samples=max_samples)
            runner_args = (ids, paths, mask_paths)
        else:
            raise ValueError(f"Unknown task_type: {self.task_type}")

        # Baseline evaluation
        if progress_callback:
            progress_callback("Running baseline evaluation", 0.1)
        baseline_metrics, baseline_failures = runner(*runner_args, **runner_kwargs)
        result["baseline_metrics"] = baseline_metrics
        result["all_failures"].extend(baseline_failures)
        result["num_samples"] = max_samples or len(ids)

        # Scenario evaluations
        total_scenarios = len([s for s in scenarios if s.get("enabled", True)])
        for s_idx, sc_cfg in enumerate(scenarios):
            if not sc_cfg.get("enabled", True):
                continue
            sc_name = sc_cfg["name"]
            sc_params = sc_cfg.get("params", {})
            pct = 0.2 + 0.7 * (s_idx / max(1, total_scenarios))
            if progress_callback:
                progress_callback(f"Running scenario: {sc_name}", pct)

            try:
                transform = get_scenario(sc_name, **sc_params)
                sc_metrics, sc_failures = runner(
                    *runner_args,
                    scenario_name=sc_name,
                    scenario_transform=transform,
                    **runner_kwargs,
                )
                # Compute degradation vs baseline
                degradation = _compute_degradation(baseline_metrics, sc_metrics, self.task_type)
                result["scenario_results"].append({
                    "scenario_name": sc_name,
                    "params": sc_params,
                    "metrics": sc_metrics,
                    "num_failures": len(sc_failures),
                    "degradation_pct": degradation,
                })
                result["all_failures"].extend(sc_failures)
            except Exception as e:
                logger.error(f"Scenario {sc_name} failed: {e}")
                result["scenario_results"].append({
                    "scenario_name": sc_name,
                    "params": sc_params,
                    "metrics": {},
                    "num_failures": 0,
                    "degradation_pct": None,
                    "error": str(e),
                })

        result["duration_seconds"] = round(time.time() - t0, 2)
        result["num_failures"] = len(result["all_failures"])
        if progress_callback:
            progress_callback("Completed", 1.0)
        return result


def _compute_degradation(baseline: dict, scenario: dict, task_type: str) -> Optional[float]:
    """Return % degradation of primary metric vs baseline."""
    key_map = {
        "classification": "accuracy",
        "detection": "map_50",
        "segmentation": "mean_iou",
    }
    key = key_map.get(task_type)
    if not key or key not in baseline or key not in scenario:
        return None
    base_val = baseline[key]
    sc_val = scenario[key]
    if base_val == 0:
        return None
    return round((base_val - sc_val) / base_val * 100, 2)

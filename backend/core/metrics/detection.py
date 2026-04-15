import numpy as np
from typing import Optional


def compute_iou(box_a: list[float], box_b: list[float]) -> float:
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])

    inter = max(0, xb - xa) * max(0, yb - ya)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """Compute AP using 11-point interpolation."""
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        prec = precisions[recalls >= t]
        ap += (np.max(prec) if prec.size > 0 else 0.0)
    return ap / 11.0


def compute_precision_recall_curve(
    gt_boxes_per_image: list[list[dict]],   # [{class_id, bbox}]
    pred_boxes_per_image: list[list[dict]], # [{class_id, bbox, score}]
    class_id: int,
    iou_threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return recalls, precisions, num_gt for a single class."""
    all_preds = []
    num_gt = 0

    for img_idx, (gts, preds) in enumerate(zip(gt_boxes_per_image, pred_boxes_per_image)):
        class_gts = [g for g in gts if g["class_id"] == class_id]
        class_preds = [p for p in preds if p["class_id"] == class_id]
        num_gt += len(class_gts)
        matched = [False] * len(class_gts)

        for pred in sorted(class_preds, key=lambda x: -x["score"]):
            best_iou = 0.0
            best_j = -1
            for j, gt in enumerate(class_gts):
                iou = compute_iou(pred["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_j = j

            if best_iou >= iou_threshold and best_j >= 0 and not matched[best_j]:
                matched[best_j] = True
                all_preds.append((pred["score"], img_idx, True))
            else:
                all_preds.append((pred["score"], img_idx, False))

    if not all_preds:
        return np.array([0.0]), np.array([0.0]), num_gt

    all_preds.sort(key=lambda x: -x[0])
    tp = np.cumsum([1 if p[2] else 0 for p in all_preds])
    fp = np.cumsum([0 if p[2] else 1 for p in all_preds])

    recalls = tp / (num_gt + 1e-10)
    precisions = tp / (tp + fp + 1e-10)
    return recalls, precisions, num_gt


def compute_detection_metrics(
    gt_boxes_per_image: list[list[dict]],
    pred_boxes_per_image: list[list[dict]],
    class_names: Optional[list[str]] = None,
    iou_thresholds: Optional[list[float]] = None,
) -> dict:
    if iou_thresholds is None:
        iou_thresholds = [0.5, 0.75]

    all_class_ids = set()
    for gts in gt_boxes_per_image:
        for g in gts:
            all_class_ids.add(g["class_id"])

    per_class = {}
    ap_at_50 = []
    ap_at_75 = []
    ap_coco = []   # mean over [0.5:0.05:0.95]

    coco_thresholds = np.arange(0.5, 1.0, 0.05).tolist()

    for class_id in sorted(all_class_ids):
        cls_name = class_names[class_id] if class_names else str(class_id)
        recalls_50, precs_50, num_gt = compute_precision_recall_curve(
            gt_boxes_per_image, pred_boxes_per_image, class_id, iou_threshold=0.5
        )
        recalls_75, precs_75, _ = compute_precision_recall_curve(
            gt_boxes_per_image, pred_boxes_per_image, class_id, iou_threshold=0.75
        )

        ap50 = compute_ap(recalls_50, precs_50)
        ap75 = compute_ap(recalls_75, precs_75)
        ap_at_50.append(ap50)
        ap_at_75.append(ap75)

        coco_aps = []
        for thr in coco_thresholds:
            r, p, _ = compute_precision_recall_curve(
                gt_boxes_per_image, pred_boxes_per_image, class_id, iou_threshold=thr
            )
            coco_aps.append(compute_ap(r, p))
        cls_coco_ap = float(np.mean(coco_aps))
        ap_coco.append(cls_coco_ap)

        per_class[cls_name] = {
            "ap_50": round(ap50, 4),
            "ap_75": round(ap75, 4),
            "ap_50_95": round(cls_coco_ap, 4),
            "num_gt": num_gt,
        }

    metrics = {
        "map_50": round(float(np.mean(ap_at_50)), 4) if ap_at_50 else 0.0,
        "map_75": round(float(np.mean(ap_at_75)), 4) if ap_at_75 else 0.0,
        "map_50_95": round(float(np.mean(ap_coco)), 4) if ap_coco else 0.0,
        "per_class_metrics": per_class,
        "num_classes": len(all_class_ids),
    }
    return metrics


def find_detection_failures(
    image_ids: list[str],
    gt_boxes_per_image: list[list[dict]],
    pred_boxes_per_image: list[list[dict]],
    iou_threshold: float = 0.5,
    conf_threshold: float = 0.5,
) -> list[dict]:
    failures = []
    for img_id, gts, preds in zip(image_ids, gt_boxes_per_image, pred_boxes_per_image):
        high_conf_preds = [p for p in preds if p["score"] >= conf_threshold]
        matched_gt = [False] * len(gts)

        for pred in sorted(high_conf_preds, key=lambda x: -x["score"]):
            best_iou = 0.0
            best_j = -1
            for j, gt in enumerate(gts):
                if gt["class_id"] == pred["class_id"]:
                    iou = compute_iou(pred["bbox"], gt["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best_j = j

            if best_iou >= iou_threshold and best_j >= 0 and not matched_gt[best_j]:
                matched_gt[best_j] = True
            else:
                error_type = "false_positive" if best_iou < iou_threshold else "wrong_class"
                failures.append({
                    "image_id": img_id,
                    "ground_truth": gts,
                    "prediction": pred,
                    "iou_score": best_iou,
                    "confidence": pred["score"],
                    "error_type": error_type,
                })

        for j, (gt, matched) in enumerate(zip(gts, matched_gt)):
            if not matched:
                failures.append({
                    "image_id": img_id,
                    "ground_truth": gt,
                    "prediction": None,
                    "iou_score": 0.0,
                    "confidence": None,
                    "error_type": "false_negative",
                })
    return failures

import numpy as np
from typing import Optional


def compute_iou_per_class(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    class_id: int,
) -> float:
    pred_cls = pred_mask == class_id
    gt_cls = gt_mask == class_id
    intersection = np.logical_and(pred_cls, gt_cls).sum()
    union = np.logical_or(pred_cls, gt_cls).sum()
    return float(intersection / union) if union > 0 else float(gt_cls.sum() == 0)


def compute_segmentation_metrics(
    gt_masks: list[np.ndarray],
    pred_masks: list[np.ndarray],
    num_classes: int,
    class_names: Optional[list[str]] = None,
    ignore_index: int = 255,
) -> dict:
    per_class_iou = {i: [] for i in range(num_classes)}
    pixel_correct = 0
    pixel_total = 0

    for gt, pred in zip(gt_masks, pred_masks):
        valid = gt != ignore_index
        pixel_correct += int(((pred == gt) & valid).sum())
        pixel_total += int(valid.sum())

        for cls_id in range(num_classes):
            cls_mask = (gt == cls_id) & valid
            if cls_mask.sum() > 0:
                per_class_iou[cls_id].append(compute_iou_per_class(pred, gt, cls_id))

    pixel_accuracy = pixel_correct / pixel_total if pixel_total > 0 else 0.0

    iou_values = []
    per_class = {}
    for cls_id in range(num_classes):
        cls_name = class_names[cls_id] if class_names else str(cls_id)
        if per_class_iou[cls_id]:
            mean_iou = float(np.mean(per_class_iou[cls_id]))
        else:
            mean_iou = 0.0
        iou_values.append(mean_iou)
        per_class[cls_name] = {
            "iou": round(mean_iou, 4),
            "num_images": len(per_class_iou[cls_id]),
        }

    return {
        "pixel_accuracy": round(pixel_accuracy, 4),
        "mean_iou": round(float(np.mean(iou_values)), 4) if iou_values else 0.0,
        "per_class_metrics": per_class,
        "num_classes": num_classes,
    }


def find_segmentation_failures(
    image_ids: list[str],
    gt_masks: list[np.ndarray],
    pred_masks: list[np.ndarray],
    iou_threshold: float = 0.5,
    num_classes: int = 2,
) -> list[dict]:
    failures = []
    for img_id, gt, pred in zip(image_ids, gt_masks, pred_masks):
        mean_iou = np.mean([
            compute_iou_per_class(pred, gt, cls_id)
            for cls_id in range(num_classes)
        ])
        if mean_iou < iou_threshold:
            pixel_acc = float((pred == gt).mean())
            failures.append({
                "image_id": img_id,
                "ground_truth": {"shape": list(gt.shape)},
                "prediction": {"shape": list(pred.shape)},
                "iou_score": round(float(mean_iou), 4),
                "confidence": round(pixel_acc, 4),
                "error_type": "low_iou",
            })
    return failures

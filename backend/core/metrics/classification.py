import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    roc_auc_score, top_k_accuracy_score
)
from typing import Optional


def compute_classification_metrics(
    y_true: list[int],
    y_pred: list[int],
    y_prob: Optional[list[list[float]]] = None,
    class_names: Optional[list[str]] = None,
    average: str = "weighted",
) -> dict:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
    }

    # Top-5 accuracy (if prob scores provided and more than 5 classes)
    if y_prob is not None:
        y_prob_arr = np.array(y_prob)
        n_classes = y_prob_arr.shape[1]
        if n_classes >= 5:
            metrics["top5_accuracy"] = float(
                top_k_accuracy_score(y_true, y_prob_arr, k=5, labels=np.arange(n_classes))
            )
        # AUC for binary tasks
        if n_classes == 2:
            try:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob_arr[:, 1]))
            except Exception:
                pass

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm.tolist()

    # Per-class metrics
    labels = list(range(int(y_true.max()) + 1))
    names = class_names if class_names else [str(i) for i in labels]
    report = classification_report(
        y_true, y_pred, labels=labels, target_names=names,
        output_dict=True, zero_division=0
    )
    per_class = {}
    for cls_name in names:
        if cls_name in report:
            per_class[cls_name] = {
                "precision": round(report[cls_name]["precision"], 4),
                "recall": round(report[cls_name]["recall"], 4),
                "f1_score": round(report[cls_name]["f1-score"], 4),
                "support": int(report[cls_name]["support"]),
            }
    metrics["per_class_metrics"] = per_class

    return metrics


def find_classification_failures(
    image_ids: list[str],
    y_true: list[int],
    y_pred: list[int],
    y_prob: Optional[list[list[float]]] = None,
    class_names: Optional[list[str]] = None,
) -> list[dict]:
    failures = []
    for i, (img_id, gt, pred) in enumerate(zip(image_ids, y_true, y_pred)):
        if gt != pred:
            conf = float(max(y_prob[i])) if y_prob else None
            gt_name = class_names[gt] if class_names else str(gt)
            pred_name = class_names[pred] if class_names else str(pred)
            failures.append({
                "image_id": img_id,
                "ground_truth": {"class_id": gt, "class_name": gt_name},
                "prediction": {"class_id": pred, "class_name": pred_name},
                "confidence": conf,
                "error_type": "wrong_class",
            })
    return failures

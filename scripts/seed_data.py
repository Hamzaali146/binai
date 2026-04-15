"""
Seed script — generates synthetic test dataset and registers a dummy model + dataset.
Run: python scripts/seed_data.py
"""
import json
import os
import sys
import uuid
from pathlib import Path

import cv2
import numpy as np
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

API_BASE = "http://localhost:8000/api/v1"
DATASET_DIR = ROOT / "data" / "datasets" / "synthetic_classification"

NUM_CLASSES = 3
SAMPLES_PER_CLASS = 20
CLASS_NAMES = ["circle", "square", "triangle"]
IMAGE_SIZE = (128, 128)


def create_synthetic_image(class_id: int, idx: int) -> np.ndarray:
    img = np.ones((*IMAGE_SIZE, 3), dtype=np.uint8) * 200
    color = [(255, 80, 80), (80, 255, 80), (80, 80, 255)][class_id]
    noise = np.random.randint(-20, 20, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    cx, cy = IMAGE_SIZE[0] // 2, IMAGE_SIZE[1] // 2
    r = 35 + idx % 10

    if class_id == 0:  # circle
        cv2.circle(img, (cx, cy), r, color, -1)
    elif class_id == 1:  # square
        half = r
        cv2.rectangle(img, (cx - half, cy - half), (cx + half, cy + half), color, -1)
    else:  # triangle
        pts = np.array([[cx, cy - r], [cx - r, cy + r], [cx + r, cy + r]], dtype=np.int32)
        cv2.fillPoly(img, [pts], color)
    return img


def build_dataset():
    print(f"Creating synthetic classification dataset at {DATASET_DIR}")
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    (DATASET_DIR / "images").mkdir(exist_ok=True)

    samples = []
    for cls_id in range(NUM_CLASSES):
        for i in range(SAMPLES_PER_CLASS):
            img = create_synthetic_image(cls_id, i)
            img_id = f"{CLASS_NAMES[cls_id]}_{i:04d}"
            img_path = DATASET_DIR / "images" / f"{img_id}.jpg"
            cv2.imwrite(str(img_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            samples.append({
                "id": img_id,
                "image_path": str(img_path),
                "class_id": cls_id,
                "class_name": CLASS_NAMES[cls_id],
            })

    annotations = {"samples": samples, "classes": CLASS_NAMES, "num_classes": NUM_CLASSES}
    ann_path = DATASET_DIR / "annotations.json"
    ann_path.write_text(json.dumps(annotations, indent=2))
    print(f"  Created {len(samples)} images and {ann_path}")
    return str(DATASET_DIR)


def register_model():
    payload = {
        "name": "ShapeClassifier",
        "version": "1.0.0",
        "task_type": "classification",
        "framework": "pytorch",
        "description": "Synthetic shape classifier for testing",
        "config": {"backbone": "resnet18", "num_classes": NUM_CLASSES},
    }
    r = requests.post(f"{API_BASE}/models", json=payload)
    r.raise_for_status()
    model = r.json()
    print(f"  Registered model: {model['name']} v{model['version']} (id={model['id'][:8]})")
    return model


def register_dataset(path: str):
    payload = {
        "name": "SyntheticShapes",
        "task_type": "classification",
        "path": path,
        "description": "Synthetic shapes: circles, squares, triangles",
        "num_samples": NUM_CLASSES * SAMPLES_PER_CLASS,
        "num_classes": NUM_CLASSES,
        "classes": CLASS_NAMES,
        "extra_metadata": {"image_size": list(IMAGE_SIZE)},
    }
    r = requests.post(f"{API_BASE}/datasets", json=payload)
    r.raise_for_status()
    ds = r.json()
    print(f"  Registered dataset: {ds['name']} (id={ds['id'][:8]})")
    return ds


def run_evaluation(model_id: str, dataset_id: str):
    payload = {
        "name": "Baseline — ShapeClassifier v1",
        "model_id": model_id,
        "dataset_id": dataset_id,
        "config": {
            "confidence_threshold": 0.5,
            "iou_threshold": 0.5,
            "max_samples": 30,
            "scenarios": [
                {"name": "gaussian_blur", "enabled": True, "params": {"kernel_size": 11}},
                {"name": "gaussian_noise", "enabled": True, "params": {"std": 30}},
                {"name": "low_light", "enabled": True, "params": {"gamma": 3.0}},
                {"name": "random_occlusion", "enabled": True, "params": {"num_patches": 2}},
                {"name": "rotation", "enabled": True, "params": {"angle": 20}},
            ],
        },
    }
    r = requests.post(f"{API_BASE}/evaluations", json=payload)
    r.raise_for_status()
    ev = r.json()
    print(f"  Started evaluation: {ev['name']} (id={ev['id'][:8]}) — status: {ev['status']}")
    return ev


if __name__ == "__main__":
    print("\n=== CV Testing Framework — Seed Data ===\n")
    dataset_path = build_dataset()
    print()
    try:
        model = register_model()
        dataset = register_dataset(dataset_path)
        print()
        ev = run_evaluation(model["id"], dataset["id"])
        print("\nSeed complete! Open http://localhost:8501 to view the dashboard.\n")
    except requests.exceptions.ConnectionError:
        print("ERROR: API server not running. Start it first:\n  python -m backend.main\n")
        sys.exit(1)

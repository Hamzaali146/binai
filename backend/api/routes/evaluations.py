"""
Evaluation lifecycle endpoints.
POST /evaluations/           → create & kick off async evaluation
GET  /evaluations/           → list
GET  /evaluations/{id}       → detail
GET  /evaluations/{id}/failures → failure cases
POST /evaluations/compare    → side-by-side model comparison
"""
import uuid
import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from backend.db.database import get_db
from backend.db import models as db_models, crud
from backend.schemas.schemas import (
    EvaluationCreate, EvaluationResponse, FailureCaseResponse,
    MetricsSummary, ScenarioResult, ModelComparisonRequest, ModelComparisonResponse,
)
from backend.core.evaluation.engine import EvaluationEngine
from backend.core.reporting.report_generator import save_report

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/evaluations", tags=["Evaluations"])


# ── Dummy predict_fn factory (replace with real model loader in production) ───

def _make_dummy_predict_fn(task_type: str, num_classes: int = 3):
    """
    Placeholder predictor — returns random predictions.
    In production swap this for your actual model inference.
    """
    import numpy as np

    def predict(images):
        results = []
        for img in images:
            if task_type == "classification":
                scores = np.random.dirichlet(np.ones(num_classes)).tolist()
                results.append({"class_id": int(np.argmax(scores)), "scores": scores})
            elif task_type == "detection":
                h, w = img.shape[:2]
                boxes = []
                for _ in range(np.random.randint(0, 4)):
                    x1 = int(np.random.uniform(0, w * 0.7))
                    y1 = int(np.random.uniform(0, h * 0.7))
                    x2 = int(x1 + np.random.uniform(10, w * 0.3))
                    y2 = int(y1 + np.random.uniform(10, h * 0.3))
                    boxes.append({
                        "class_id": int(np.random.randint(0, num_classes)),
                        "bbox": [x1, y1, min(x2, w), min(y2, h)],
                        "score": float(np.random.uniform(0.3, 1.0)),
                    })
                results.append(boxes)
            elif task_type == "segmentation":
                h, w = img.shape[:2]
                results.append(np.random.randint(0, num_classes, (h, w)).astype(np.uint8))
        return results
    return predict


# ── Background evaluation task ────────────────────────────────────────────────

def _run_evaluation_task(eval_id: str):
    from backend.db.database import SessionLocal
    db = SessionLocal()
    try:
        ev = crud.get_evaluation(db, eval_id)
        if not ev:
            return

        crud.update_evaluation(db, eval_id, {
            "status": db_models.EvaluationStatus.RUNNING,
            "started_at": datetime.utcnow(),
        })

        dataset = crud.get_dataset(db, ev.dataset_id)
        model = crud.get_model(db, ev.model_id)
        config = ev.config or {}
        num_classes = dataset.num_classes if dataset else 3

        predict_fn = _make_dummy_predict_fn(ev.task_type.value, num_classes)
        engine = EvaluationEngine(
            task_type=ev.task_type.value,
            dataset_path=dataset.path if dataset else "/tmp",
            predict_fn=predict_fn,
            class_names=dataset.classes if dataset else None,
            num_classes=num_classes,
        )

        try:
            result = engine.run(config)
        except Exception as exc:
            logger.exception(f"Engine run failed for eval {eval_id}")
            crud.update_evaluation(db, eval_id, {
                "status": db_models.EvaluationStatus.FAILED,
                "error_message": str(exc),
                "completed_at": datetime.utcnow(),
            })
            return

        # Persist failure cases (capped)
        failures = result.get("all_failures", [])[:100]
        failure_objs = [
            db_models.FailureCase(
                id=str(uuid.uuid4()),
                evaluation_id=eval_id,
                image_id=f.get("image_id"),
                ground_truth=f.get("ground_truth"),
                prediction=f.get("prediction"),
                confidence=f.get("confidence"),
                iou_score=f.get("iou_score"),
                error_type=f.get("error_type"),
                scenario=f.get("scenario"),
                extra_metadata={},
                created_at=datetime.utcnow(),
            )
            for f in failures
        ]
        crud.bulk_create_failures(db, failure_objs)

        # Persist scenario runs
        for sc in result.get("scenario_results", []):
            sc_run = db_models.ScenarioRun(
                id=str(uuid.uuid4()),
                evaluation_id=eval_id,
                scenario_name=sc["scenario_name"],
                scenario_params=sc.get("params", {}),
                metrics=sc.get("metrics", {}),
                num_failures=sc.get("num_failures", 0),
                degradation_pct=sc.get("degradation_pct"),
                created_at=datetime.utcnow(),
            )
            crud.create_scenario_run(db, sc_run)

        crud.update_evaluation(db, eval_id, {
            "status": db_models.EvaluationStatus.COMPLETED,
            "metrics": result.get("baseline_metrics", {}),
            "scenario_results": result.get("scenario_results", []),
            "num_samples": result.get("num_samples", 0),
            "num_failures": result.get("num_failures", 0),
            "duration_seconds": result.get("duration_seconds"),
            "completed_at": datetime.utcnow(),
        })

        # Generate JSON report
        ev_dict = {
            "id": eval_id,
            "name": ev.name,
            "metrics": result.get("baseline_metrics", {}),
            "scenario_results": result.get("scenario_results", []),
            "num_failures": result.get("num_failures", 0),
            "failure_cases": [
                {"image_id": f.get("image_id"), "error_type": f.get("error_type"),
                 "ground_truth": f.get("ground_truth"), "prediction": f.get("prediction"),
                 "confidence": f.get("confidence"), "scenario": f.get("scenario")}
                for f in failures
            ],
        }
        save_report(
            ev_dict,
            fmt="json",
            model_info=f"{model.name} v{model.version}" if model else "Unknown",
            dataset_info=dataset.name if dataset else "Unknown",
        )

    finally:
        db.close()


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/", response_model=list[EvaluationResponse])
def list_evaluations(
    model_id: Optional[str] = None,
    dataset_id: Optional[str] = None,
    skip: int = 0,
    limit: int = 50,
    db: Session = Depends(get_db),
):
    rows = crud.list_evaluations(db, model_id=model_id, dataset_id=dataset_id,
                                  skip=skip, limit=limit)
    return [_db_to_response(r) for r in rows]


@router.post("/", response_model=EvaluationResponse, status_code=status.HTTP_202_ACCEPTED)
def create_evaluation(
    payload: EvaluationCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    if not crud.get_model(db, payload.model_id):
        raise HTTPException(status_code=404, detail="Model not found")
    if not crud.get_dataset(db, payload.dataset_id):
        raise HTTPException(status_code=404, detail="Dataset not found")

    model = crud.get_model(db, payload.model_id)
    ev = db_models.EvaluationRun(
        id=str(uuid.uuid4()),
        name=payload.name,
        model_id=payload.model_id,
        dataset_id=payload.dataset_id,
        task_type=model.task_type,
        status=db_models.EvaluationStatus.PENDING,
        config=payload.config.model_dump(),
        metrics={},
        scenario_results=[],
        num_samples=0,
        num_failures=0,
        created_at=datetime.utcnow(),
    )
    crud.create_evaluation(db, ev)
    background_tasks.add_task(_run_evaluation_task, ev.id)
    return _db_to_response(ev)


@router.get("/{eval_id}", response_model=EvaluationResponse)
def get_evaluation(eval_id: str, db: Session = Depends(get_db)):
    obj = crud.get_evaluation(db, eval_id)
    if not obj:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    return _db_to_response(obj)


@router.get("/{eval_id}/failures", response_model=list[FailureCaseResponse])
def list_failures(
    eval_id: str,
    error_type: Optional[str] = None,
    skip: int = 0,
    limit: int = 50,
    db: Session = Depends(get_db),
):
    if not crud.get_evaluation(db, eval_id):
        raise HTTPException(status_code=404, detail="Evaluation not found")
    return crud.list_failures(db, eval_id, error_type=error_type, skip=skip, limit=limit)


@router.post("/compare", response_model=ModelComparisonResponse)
def compare_evaluations(payload: ModelComparisonRequest, db: Session = Depends(get_db)):
    evals = []
    for eid in payload.evaluation_ids:
        ev = crud.get_evaluation(db, eid)
        if not ev:
            raise HTTPException(status_code=404, detail=f"Evaluation {eid} not found")
        model = crud.get_model(db, ev.model_id)
        evals.append({
            "id": ev.id,
            "name": ev.name,
            "model_name": model.name if model else "Unknown",
            "model_version": model.version if model else "?",
            "status": ev.status.value if ev.status else "unknown",
            "metrics": ev.metrics or {},
            "num_samples": ev.num_samples,
            "num_failures": ev.num_failures,
        })

    # Compute deltas between first and each subsequent run
    metric_deltas = {}
    if len(evals) >= 2:
        base = evals[0]["metrics"]
        for ev_data in evals[1:]:
            for m in payload.metrics:
                base_val = base.get(m)
                comp_val = ev_data["metrics"].get(m)
                if base_val is not None and comp_val is not None:
                    key = f"{ev_data['name']} vs {evals[0]['name']}"
                    if key not in metric_deltas:
                        metric_deltas[key] = {}
                    metric_deltas[key][m] = round(comp_val - base_val, 4)

    best = max(evals, key=lambda e: e["metrics"].get(payload.metrics[0], -1)) if evals else None
    return ModelComparisonResponse(
        evaluations=evals,
        metric_deltas=metric_deltas,
        best_model=best["model_name"] if best else None,
    )


# ── Helper ────────────────────────────────────────────────────────────────────

def _db_to_response(ev: db_models.EvaluationRun) -> EvaluationResponse:
    sr_list = []
    for sc in (ev.scenario_results or []):
        sc_metrics_raw = sc.get("metrics", {})
        sc_metrics = MetricsSummary(
            accuracy=sc_metrics_raw.get("accuracy"),
            precision=sc_metrics_raw.get("precision"),
            recall=sc_metrics_raw.get("recall"),
            f1_score=sc_metrics_raw.get("f1_score"),
            map_50=sc_metrics_raw.get("map_50"),
            map_75=sc_metrics_raw.get("map_75"),
            map_50_95=sc_metrics_raw.get("map_50_95"),
            mean_iou=sc_metrics_raw.get("mean_iou"),
            pixel_accuracy=sc_metrics_raw.get("pixel_accuracy"),
            per_class_metrics=sc_metrics_raw.get("per_class_metrics"),
        )
        sr_list.append(ScenarioResult(
            scenario_name=sc.get("scenario_name", ""),
            params=sc.get("params", {}),
            metrics=sc_metrics,
            num_samples=sc.get("num_samples", 0),
            num_failures=sc.get("num_failures", 0),
            degradation_pct=sc.get("degradation_pct"),
        ))

    m = ev.metrics or {}
    metrics = MetricsSummary(
        accuracy=m.get("accuracy"),
        precision=m.get("precision"),
        recall=m.get("recall"),
        f1_score=m.get("f1_score"),
        map_50=m.get("map_50"),
        map_75=m.get("map_75"),
        map_50_95=m.get("map_50_95"),
        mean_iou=m.get("mean_iou"),
        pixel_accuracy=m.get("pixel_accuracy"),
        per_class_metrics=m.get("per_class_metrics"),
    ) if m else None

    return EvaluationResponse(
        id=ev.id,
        name=ev.name,
        model_id=ev.model_id,
        dataset_id=ev.dataset_id,
        task_type=ev.task_type,
        status=ev.status,
        config=ev.config or {},
        metrics=metrics,
        scenario_results=sr_list,
        num_samples=ev.num_samples,
        num_failures=ev.num_failures,
        duration_seconds=ev.duration_seconds,
        error_message=ev.error_message,
        started_at=ev.started_at,
        completed_at=ev.completed_at,
        created_at=ev.created_at,
    )

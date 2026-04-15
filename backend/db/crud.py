from sqlalchemy.orm import Session
from sqlalchemy import desc
from typing import Optional
from backend.db import models


# ── Model Registry ──────────────────────────────────────────────────────────

def get_model(db: Session, model_id: str) -> Optional[models.ModelRegistry]:
    return db.query(models.ModelRegistry).filter(models.ModelRegistry.id == model_id).first()


def list_models(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.ModelRegistry).offset(skip).limit(limit).all()


def create_model(db: Session, model: models.ModelRegistry) -> models.ModelRegistry:
    db.add(model)
    db.commit()
    db.refresh(model)
    return model


def delete_model(db: Session, model_id: str) -> bool:
    obj = get_model(db, model_id)
    if not obj:
        return False
    db.delete(obj)
    db.commit()
    return True


# ── Datasets ─────────────────────────────────────────────────────────────────

def get_dataset(db: Session, dataset_id: str) -> Optional[models.Dataset]:
    return db.query(models.Dataset).filter(models.Dataset.id == dataset_id).first()


def list_datasets(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.Dataset).offset(skip).limit(limit).all()


def create_dataset(db: Session, dataset: models.Dataset) -> models.Dataset:
    db.add(dataset)
    db.commit()
    db.refresh(dataset)
    return dataset


def delete_dataset(db: Session, dataset_id: str) -> bool:
    obj = get_dataset(db, dataset_id)
    if not obj:
        return False
    db.delete(obj)
    db.commit()
    return True


# ── Evaluation Runs ──────────────────────────────────────────────────────────

def get_evaluation(db: Session, eval_id: str) -> Optional[models.EvaluationRun]:
    return (
        db.query(models.EvaluationRun)
        .filter(models.EvaluationRun.id == eval_id)
        .first()
    )


def list_evaluations(
    db: Session,
    model_id: Optional[str] = None,
    dataset_id: Optional[str] = None,
    skip: int = 0,
    limit: int = 100,
):
    q = db.query(models.EvaluationRun)
    if model_id:
        q = q.filter(models.EvaluationRun.model_id == model_id)
    if dataset_id:
        q = q.filter(models.EvaluationRun.dataset_id == dataset_id)
    return q.order_by(desc(models.EvaluationRun.created_at)).offset(skip).limit(limit).all()


def create_evaluation(db: Session, evaluation: models.EvaluationRun) -> models.EvaluationRun:
    db.add(evaluation)
    db.commit()
    db.refresh(evaluation)
    return evaluation


def update_evaluation(db: Session, eval_id: str, updates: dict) -> Optional[models.EvaluationRun]:
    obj = get_evaluation(db, eval_id)
    if not obj:
        return None
    for k, v in updates.items():
        setattr(obj, k, v)
    db.commit()
    db.refresh(obj)
    return obj


# ── Failure Cases ─────────────────────────────────────────────────────────────

def create_failure_case(db: Session, failure: models.FailureCase) -> models.FailureCase:
    db.add(failure)
    db.commit()
    db.refresh(failure)
    return failure


def list_failures(
    db: Session,
    eval_id: str,
    error_type: Optional[str] = None,
    skip: int = 0,
    limit: int = 50,
):
    q = db.query(models.FailureCase).filter(models.FailureCase.evaluation_id == eval_id)
    if error_type:
        q = q.filter(models.FailureCase.error_type == error_type)
    return q.order_by(desc(models.FailureCase.created_at)).offset(skip).limit(limit).all()


def bulk_create_failures(db: Session, failures: list[models.FailureCase]):
    db.bulk_save_objects(failures)
    db.commit()


# ── Scenario Runs ─────────────────────────────────────────────────────────────

def create_scenario_run(db: Session, run: models.ScenarioRun) -> models.ScenarioRun:
    db.add(run)
    db.commit()
    db.refresh(run)
    return run


def list_scenario_runs(db: Session, eval_id: str):
    return (
        db.query(models.ScenarioRun)
        .filter(models.ScenarioRun.evaluation_id == eval_id)
        .all()
    )

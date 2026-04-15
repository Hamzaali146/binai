import uuid
from datetime import datetime
from pathlib import Path
import json
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status
from sqlalchemy.orm import Session

from backend.db.database import get_db
from backend.db import models as db_models, crud
from backend.schemas.schemas import DatasetCreate, DatasetResponse
from backend.config import settings

router = APIRouter(prefix="/datasets", tags=["Datasets"])


@router.get("/", response_model=list[DatasetResponse])
def list_datasets(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    return crud.list_datasets(db, skip=skip, limit=limit)


@router.post("/", response_model=DatasetResponse, status_code=status.HTTP_201_CREATED)
def register_dataset(payload: DatasetCreate, db: Session = Depends(get_db)):
    dataset_path = Path(payload.path)
    if not dataset_path.exists():
        raise HTTPException(
            status_code=400,
            detail=f"Dataset path does not exist: {payload.path}"
        )
    ds = db_models.Dataset(
        id=str(uuid.uuid4()),
        name=payload.name,
        task_type=payload.task_type,
        description=payload.description,
        path=str(dataset_path.resolve()),
        num_samples=payload.num_samples,
        num_classes=payload.num_classes,
        classes=payload.classes,
        extra_metadata=payload.extra_metadata,
        created_at=datetime.utcnow(),
    )
    return crud.create_dataset(db, ds)


@router.get("/{dataset_id}", response_model=DatasetResponse)
def get_dataset(dataset_id: str, db: Session = Depends(get_db)):
    obj = crud.get_dataset(db, dataset_id)
    if not obj:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return obj


@router.delete("/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_dataset(dataset_id: str, db: Session = Depends(get_db)):
    if not crud.delete_dataset(db, dataset_id):
        raise HTTPException(status_code=404, detail="Dataset not found")


@router.post("/validate/{dataset_id}")
def validate_dataset(dataset_id: str, db: Session = Depends(get_db)):
    """Validate that dataset annotations.json is well-formed."""
    obj = crud.get_dataset(db, dataset_id)
    if not obj:
        raise HTTPException(status_code=404, detail="Dataset not found")

    ann_file = Path(obj.path) / "annotations.json"
    if not ann_file.exists():
        return {"valid": False, "error": "annotations.json not found"}
    try:
        with open(ann_file) as f:
            data = json.load(f)
        samples = data.get("samples", [])
        return {
            "valid": True,
            "num_samples": len(samples),
            "classes": data.get("classes", []),
        }
    except Exception as e:
        return {"valid": False, "error": str(e)}

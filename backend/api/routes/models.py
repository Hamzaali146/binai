import uuid
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.db.database import get_db
from backend.db import models as db_models, crud
from backend.schemas.schemas import ModelCreate, ModelResponse

router = APIRouter(prefix="/models", tags=["Model Registry"])


@router.get("/", response_model=list[ModelResponse])
def list_models(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    return crud.list_models(db, skip=skip, limit=limit)


@router.post("/", response_model=ModelResponse, status_code=status.HTTP_201_CREATED)
def register_model(payload: ModelCreate, db: Session = Depends(get_db)):
    model = db_models.ModelRegistry(
        id=str(uuid.uuid4()),
        name=payload.name,
        version=payload.version,
        task_type=payload.task_type,
        description=payload.description,
        framework=payload.framework,
        model_path=payload.model_path,
        config=payload.config,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    return crud.create_model(db, model)


@router.get("/{model_id}", response_model=ModelResponse)
def get_model(model_id: str, db: Session = Depends(get_db)):
    obj = crud.get_model(db, model_id)
    if not obj:
        raise HTTPException(status_code=404, detail="Model not found")
    return obj


@router.delete("/{model_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_model(model_id: str, db: Session = Depends(get_db)):
    if not crud.delete_model(db, model_id):
        raise HTTPException(status_code=404, detail="Model not found")

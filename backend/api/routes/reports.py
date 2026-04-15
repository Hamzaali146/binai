from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from sqlalchemy.orm import Session

from backend.db.database import get_db
from backend.db import crud
from backend.schemas.schemas import ReportRequest
from backend.core.reporting.report_generator import generate_html_report, save_report

router = APIRouter(prefix="/reports", tags=["Reports"])


@router.post("/generate")
def generate_report(payload: ReportRequest, db: Session = Depends(get_db)):
    ev = crud.get_evaluation(db, payload.evaluation_id)
    if not ev:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    if ev.status.value != "completed":
        raise HTTPException(status_code=400, detail="Evaluation is not yet completed")

    model = crud.get_model(db, ev.model_id)
    dataset = crud.get_dataset(db, ev.dataset_id)
    failures = crud.list_failures(db, ev.id, limit=100)

    ev_dict = {
        "id": ev.id,
        "name": ev.name,
        "metrics": ev.metrics or {},
        "scenario_results": ev.scenario_results or [],
        "num_failures": ev.num_failures,
        "failure_cases": [
            {
                "image_id": f.image_id,
                "error_type": f.error_type,
                "ground_truth": f.ground_truth,
                "prediction": f.prediction,
                "confidence": f.confidence,
                "iou_score": f.iou_score,
                "scenario": f.scenario,
            }
            for f in failures
        ],
    }

    out_path = save_report(
        ev_dict,
        fmt=payload.format,
        model_info=f"{model.name} v{model.version}" if model else "Unknown",
        dataset_info=dataset.name if dataset else "Unknown",
    )

    return {"report_path": str(out_path), "format": payload.format}


@router.get("/html/{eval_id}", response_class=HTMLResponse)
def get_html_report(eval_id: str, db: Session = Depends(get_db)):
    ev = crud.get_evaluation(db, eval_id)
    if not ev:
        raise HTTPException(status_code=404, detail="Evaluation not found")

    model = crud.get_model(db, ev.model_id)
    dataset = crud.get_dataset(db, ev.dataset_id)
    failures = crud.list_failures(db, eval_id, limit=100)

    ev_dict = {
        "id": ev.id,
        "name": ev.name,
        "metrics": ev.metrics or {},
        "scenario_results": ev.scenario_results or [],
        "num_failures": ev.num_failures,
        "failure_cases": [
            {"image_id": f.image_id, "error_type": f.error_type,
             "ground_truth": f.ground_truth, "prediction": f.prediction,
             "confidence": f.confidence, "scenario": f.scenario}
            for f in failures
        ],
    }
    html = generate_html_report(
        ev_dict,
        model_info=f"{model.name} v{model.version}" if model else "Unknown",
        dataset_info=dataset.name if dataset else "Unknown",
    )
    return HTMLResponse(content=html)


@router.get("/download/{eval_id}")
def download_report(eval_id: str, fmt: str = "json", db: Session = Depends(get_db)):
    ev = crud.get_evaluation(db, eval_id)
    if not ev:
        raise HTTPException(status_code=404, detail="Evaluation not found")

    model = crud.get_model(db, ev.model_id)
    dataset = crud.get_dataset(db, ev.dataset_id)
    failures = crud.list_failures(db, eval_id, limit=100)

    ev_dict = {
        "id": ev.id,
        "name": ev.name,
        "metrics": ev.metrics or {},
        "scenario_results": ev.scenario_results or [],
        "num_failures": ev.num_failures,
        "failure_cases": [
            {"image_id": f.image_id, "error_type": f.error_type,
             "ground_truth": f.ground_truth, "prediction": f.prediction,
             "confidence": f.confidence, "scenario": f.scenario}
            for f in failures
        ],
    }
    out_path = save_report(
        ev_dict, fmt=fmt,
        model_info=f"{model.name} v{model.version}" if model else "Unknown",
        dataset_info=dataset.name if dataset else "Unknown",
    )
    media_type = "text/html" if fmt == "html" else "application/json"
    return FileResponse(str(out_path), media_type=media_type, filename=out_path.name)

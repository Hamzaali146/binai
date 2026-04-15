from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
import numpy as np
import cv2
import io
from backend.core.scenarios import list_scenarios, get_scenario, SCENARIO_REGISTRY

router = APIRouter(prefix="/scenarios", tags=["Scenarios"])


@router.get("/")
def get_available_scenarios():
    """List all available robustness scenarios with their descriptions."""
    return list_scenarios()


@router.post("/preview")
async def preview_scenario(
    scenario_name: str,
    image: UploadFile = File(...),
    params: str = "{}",
):
    """
    Apply a scenario to an uploaded image and return the transformed image.
    Useful for visually validating scenario effects.
    """
    import json
    if scenario_name not in SCENARIO_REGISTRY:
        raise HTTPException(status_code=400, detail=f"Unknown scenario: {scenario_name}")

    try:
        sc_params = json.loads(params)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in params")

    contents = await image.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    transform = get_scenario(scenario_name, **sc_params)
    out = transform.apply(img_rgb)
    out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

    _, encoded = cv2.imencode(".jpg", out_bgr)
    return StreamingResponse(io.BytesIO(encoded.tobytes()), media_type="image/jpeg")


@router.get("/{scenario_name}/params")
def get_scenario_params(scenario_name: str):
    """Return the default parameters and description for a named scenario."""
    if scenario_name not in SCENARIO_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Scenario '{scenario_name}' not found")
    cls = SCENARIO_REGISTRY[scenario_name]
    return {
        "name": cls.name,
        "description": cls.description,
    }

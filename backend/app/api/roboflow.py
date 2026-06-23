from fastapi import APIRouter, File, Form, UploadFile
from ..core.dataset_registry import DATASETS
from ..services.roboflow_service import roboflow_service

router = APIRouter(prefix="/api", tags=["roboflow"])


@router.get("/roboflow/status")
def roboflow_status():
    return roboflow_service.status()


@router.get("/datasets")
def datasets():
    return {"datasets": DATASETS}


@router.post("/roboflow/test-inference")
async def test_inference(image: UploadFile = File(...), model_type: str = Form("vehicle")):
    if model_type not in {"vehicle", "plate", "behavior"}:
        return {"error": "model_type must be vehicle, plate, or behavior"}
    content = await image.read()
    result = await roboflow_service.infer(content, model_type)
    if result is not None:
        return {"provider": "RoboflowHostedProvider", "model_type": model_type, "predictions": result.get("predictions", [])}
    return {
        "provider": "MockAIProvider", "model_type": model_type, "simulated": True,
        "message": "Roboflow is not configured, returning a simulated prediction.",
        "predictions": [{"class": "TOGG" if model_type == "vehicle" else model_type, "confidence": .91, "x": 320, "y": 180, "width": 260, "height": 160}],
    }

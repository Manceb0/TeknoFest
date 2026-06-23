from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api import auth, incidents, qod, roboflow
from .core.config import settings
from .core.metrics import metrics
from .db import db
from .ws import session

app = FastAPI(title=settings.app_name, version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(qod.router)
app.include_router(incidents.router)
app.include_router(roboflow.router)
app.include_router(session.router)


@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "service": settings.app_name,
        "active_ai_provider": "LocalYOLOProvider",
        "inference_real": True,
        "device": settings.resolved_device,
        "model": settings.yolo_model_path,
    }


@app.get("/api/diagnostics")
def diagnostics():
    return metrics.snapshot(db.count(), "LocalYOLOProvider", settings.roboflow_configured)

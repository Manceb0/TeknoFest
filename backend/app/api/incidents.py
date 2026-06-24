from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from ..db import db

router = APIRouter(prefix="/api/incidents", tags=["incidents"])
EVIDENCE_DIR = Path("evidence")


@router.get("")
def list_incidents():
    return {"incidents": db.incidents()}


@router.get("/{incident_id}")
def get_incident(incident_id: str):
    incident = db.incident(incident_id)
    if not incident:
        raise HTTPException(404, "Incident not found")
    return incident


@router.get("/{incident_id}/similar")
def similar_incidents(incident_id: str, k: int = 5):
    if not db.incident(incident_id):
        raise HTTPException(404, "Incident not found")
    return {"incident_id": incident_id, "similar": db.similar(incident_id, k)}


@router.get("/{incident_id}/snapshot")
def incident_snapshot(incident_id: str):
    """Serve the best-evidence photo captured for this incident."""
    path = EVIDENCE_DIR / f"{incident_id}.jpg"
    if not path.exists():
        raise HTTPException(404, "Snapshot not found")
    return FileResponse(path, media_type="image/jpeg")

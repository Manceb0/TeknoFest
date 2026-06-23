from fastapi import APIRouter, HTTPException
from ..db import db

router = APIRouter(prefix="/api/incidents", tags=["incidents"])


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

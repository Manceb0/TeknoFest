from uuid import uuid4
from ..db import db


class IncidentStore:
    def save(self, detection: dict, embedding: list[float] | None = None,
             snapshot_path: str | None = None) -> str:
        incident_id = uuid4().hex
        db.add_incident({
            "incident_id": incident_id,
            "detection": detection,
            "snapshot_path": snapshot_path,
            "embedding": embedding,
        })
        return incident_id


incident_store = IncidentStore()

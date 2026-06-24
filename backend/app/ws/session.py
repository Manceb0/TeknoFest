import asyncio
import base64
from pathlib import Path

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from ..core.metrics import metrics
from ..services.ai_provider import SessionState
from ..services.incident_store import incident_store
from ..services.local_yolo_provider import LocalYOLOProvider

router = APIRouter()
provider = LocalYOLOProvider()

EVIDENCE_DIR = Path("evidence")
EVIDENCE_DIR.mkdir(exist_ok=True)


def _jpeg_bytes(payload: dict) -> bytes:
    encoded = payload.get("image", "")
    if "," in encoded:
        encoded = encoded.split(",", 1)[1]
    return base64.b64decode(encoded)


def _is_notable(detection: dict) -> bool:
    """A frame worth keeping as evidence: QoD active, risky behavior, or high risk."""
    return (
        detection["qod"]["state"] == "active"
        or detection["behavior"]["label"] in ("smoking_detected", "phone_detected")
        or detection["risk"]["score"] >= 70
    )


@router.websocket("/ws/session/{session_id}")
async def session_socket(websocket: WebSocket, session_id: str):
    await websocket.accept()
    metrics.active_sessions += 1
    state = SessionState(session_id=session_id)
    queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=2)

    async def receive_frames():
        while True:
            payload = await websocket.receive_json()
            if payload.get("type") != "frame" or not payload.get("image"):
                continue
            if queue.full():
                try:
                    queue.get_nowait()
                    queue.task_done()
                    metrics.dropped_frames += 1
                except asyncio.QueueEmpty:
                    pass
            await queue.put(payload)

    async def process_frames():
        while True:
            payload = await queue.get()
            detection = await provider.process_frame(payload, state)

            # Track the best evidence frame: closest vehicle (largest area) among
            # notable frames, so the saved photo is the clearest moment.
            area = detection["detections"][0]["bbox_area_ratio"] if detection["detections"] else 0.0
            if _is_notable(detection) and area > state.best_evidence_area:
                state.best_evidence_area = area
                state.best_evidence_jpeg = _jpeg_bytes(payload)
                # if an incident snapshot already exists, upgrade it to this closer frame
                if state.snapshot_incident_id:
                    (EVIDENCE_DIR / f"{state.snapshot_incident_id}.jpg").write_bytes(state.best_evidence_jpeg)

            should_store = detection["qod"]["triggered"] or detection["risk"]["score"] >= 70
            if should_store and (not state.incident_recorded or detection["qod"]["triggered"]):
                embedding = await asyncio.to_thread(provider.embed_from_payload, payload)
                snapshot_jpeg = state.best_evidence_jpeg or _jpeg_bytes(payload)
                incident_id = incident_store.save(detection, embedding, snapshot_path="pending")
                (EVIDENCE_DIR / f"{incident_id}.jpg").write_bytes(snapshot_jpeg)
                # persist the real relative path now that we know the id
                from ..db import db
                with db.lock:
                    db.conn.execute("UPDATE incidents SET snapshot_path=? WHERE incident_id=?",
                                    [f"evidence/{incident_id}.jpg", incident_id])
                detection["incident_id"] = incident_id
                state.snapshot_incident_id = incident_id
                state.incident_recorded = True

            await websocket.send_json(detection)
            queue.task_done()

    tasks = [asyncio.create_task(receive_frames()), asyncio.create_task(process_frames())]
    try:
        await asyncio.gather(*tasks)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        for task in tasks:
            task.cancel()
        metrics.active_sessions = max(0, metrics.active_sessions - 1)

import asyncio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from ..core.metrics import metrics
from ..services.ai_provider import SessionState
from ..services.incident_store import incident_store
from ..services.local_yolo_provider import LocalYOLOProvider

router = APIRouter()
provider = LocalYOLOProvider()


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
            should_store = detection["qod"]["triggered"] or detection["risk"]["score"] >= 70
            if should_store and (not state.incident_recorded or detection["qod"]["triggered"]):
                embedding = await asyncio.to_thread(provider.embed_from_payload, payload)
                detection["incident_id"] = incident_store.save(detection, embedding)
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

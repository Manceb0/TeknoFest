from fastapi import APIRouter
from pydantic import BaseModel
from ..services.qod_service import qod_service

router = APIRouter(prefix="/api/qod", tags=["qod"])


class QoDRequest(BaseModel):
    session_id: str
    reason: str = "Manual demo trigger"
    duration_seconds: int = 10


@router.post("/request")
async def request_qod(body: QoDRequest):
    return await qod_service.request(body.session_id, body.reason, body.duration_seconds)


@router.post("/release")
async def release_qod(body: QoDRequest):
    return await qod_service.release(body.session_id)

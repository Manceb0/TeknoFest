from fastapi import APIRouter
from pydantic import BaseModel
from ..services.number_verification import number_verification

router = APIRouter(prefix="/api/auth", tags=["auth"])


class PhoneRequest(BaseModel):
    phone: str


@router.post("/number-verification")
async def verify_number(body: PhoneRequest):
    return await number_verification.verify(body.phone)

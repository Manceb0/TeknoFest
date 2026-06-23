from abc import ABC, abstractmethod
from datetime import datetime, timezone
from uuid import uuid4
from ..db import db


class NumberVerificationService(ABC):
    @abstractmethod
    async def verify(self, phone: str) -> dict: ...


class MockNumberVerificationService(NumberVerificationService):
    async def verify(self, phone: str) -> dict:
        session_id = uuid4().hex
        now = datetime.now(timezone.utc).isoformat()
        db.create_session(session_id, phone, now)
        return {
            "verified": True,
            "provider": "MockTurkcellNumberVerification",
            "session_id": session_id,
            "message": "Silent verification simulated successfully",
        }


number_verification = MockNumberVerificationService()

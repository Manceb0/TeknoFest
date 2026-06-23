from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone


class QoDService(ABC):
    @abstractmethod
    async def request(self, session_id: str, reason: str, duration_seconds: int) -> dict: ...

    @abstractmethod
    async def release(self, session_id: str) -> dict: ...


class MockQoDService(QoDService):
    def __init__(self):
        self.sessions: dict[str, datetime] = {}

    async def request(self, session_id, reason, duration_seconds=10):
        expires = datetime.now(timezone.utc) + timedelta(seconds=duration_seconds)
        self.sessions[session_id] = expires
        return {
            "qod_active": True, "state": "active", "reason": reason,
            "quality_profile": "1080p", "bandwidth_target_mbps": 4,
            "expires_at": expires.isoformat(),
        }

    async def release(self, session_id):
        self.sessions.pop(session_id, None)
        return {"qod_active": False, "state": "baseline", "quality_profile": "480p", "bandwidth_target_mbps": 1}

    def active(self, session_id):
        expires = self.sessions.get(session_id)
        if expires and expires > datetime.now(timezone.utc):
            return True
        self.sessions.pop(session_id, None)
        return False


qod_service = MockQoDService()

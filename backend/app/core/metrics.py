from collections import deque
from dataclasses import dataclass, field


@dataclass
class Metrics:
    active_sessions: int = 0
    dropped_frames: int = 0
    qod_trigger_count: int = 0
    latencies: deque[float] = field(default_factory=lambda: deque(maxlen=500))

    def snapshot(self, incident_count: int, provider: str, roboflow: bool) -> dict:
        values = sorted(self.latencies)
        avg = sum(values) / len(values) if values else 0
        p95 = values[min(len(values) - 1, int(len(values) * .95))] if values else 0
        return {
            "active_sessions": self.active_sessions,
            "current_queue_size": 0,
            "avg_latency_ms": round(avg, 1),
            "p95_latency_ms": round(p95, 1),
            "dropped_frames": self.dropped_frames,
            "qod_trigger_count": self.qod_trigger_count,
            "incident_count": incident_count,
            "active_ai_provider": provider,
            "roboflow_configured": roboflow,
            "backend_status": "healthy",
        }


metrics = Metrics()

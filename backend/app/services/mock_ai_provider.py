import math
import random
from datetime import datetime, timezone
from .ai_provider import AIProvider, SessionState
from .qod_service import qod_service
from ..core.metrics import metrics


class MockAIProvider(AIProvider):
    name = "MockAIProvider"

    async def process_frame(self, payload: dict, state: SessionState) -> dict:
        state.frame_id += 1
        progress = float(payload.get("progress", (state.frame_id % 180) / 180))
        width = int(payload.get("width", 1280))
        height = int(payload.get("height", 720))
        wave = (math.sin(progress * math.pi * 2) + 1) / 2
        ratio = min(.28, .035 + progress * .22)
        if state.manual_trigger:
            ratio = max(ratio, .19)
            progress = max(progress, .7)
            state.manual_trigger = False
        box_w = int(width * math.sqrt(ratio * 1.65))
        box_h = int(height * math.sqrt(ratio / 1.65))
        x = int((width - box_w) / 2 + (wave - .5) * width * .08)
        y = max(12, int(height * .62 - box_h / 2))
        qod_was_active = qod_service.active(state.session_id)
        triggered = ratio > .15 and not qod_was_active
        if triggered:
            await qod_service.request(state.session_id, "TOGG bbox > 15% frame", 10)
            metrics.qod_trigger_count += 1
        active = qod_service.active(state.session_id)
        behavior = "phone_use" if progress > .52 else ("smoking" if progress > .35 else "attentive")
        behavior_conf = min(.96, .73 + (.12 if active else 0) + random.uniform(-.025, .025))
        plate_conf = min(.97, .79 + (.11 if active else 0) + random.uniform(-.02, .02))
        speeding = progress > .42
        signals = {
            "swerving": round(wave * 30 if progress > .25 else 0),
            "smoking": 25 if behavior == "smoking" else 0,
            "speeding": 25 if speeding else 0,
            "phone_use": 20 if behavior == "phone_use" else 0,
        }
        raw_risk = sum(signals.values())
        state.risk_window = (state.risk_window + [raw_risk])[-6:]
        risk = round(sum(state.risk_window) / len(state.risk_window))
        latency = round(random.uniform(48, 84) if active else random.uniform(72, 126), 1)
        metrics.latencies.append(latency)
        return {
            "session_id": state.session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "frame_id": state.frame_id,
            "mode": "qod" if active else "baseline",
            "stream_quality": "1080p" if active else "480p",
            "detections": [{
                "track_id": "TOGG-001", "label": "TOGG",
                "bbox": {"x": x, "y": y, "w": box_w, "h": box_h},
                "bbox_area_ratio": round(ratio, 3), "confidence": round(.9 + random.uniform(-.025, .025), 2),
            }],
            "plate": {
                "text": "34 TOG 2026" if ratio > .07 else "READING…",
                "confidence": round(plate_conf, 2) if ratio > .07 else 0,
                "roi": {"x": x + int(box_w * .32), "y": y + int(box_h * .72), "w": int(box_w * .36), "h": int(box_h * .14)},
            },
            "behavior": {"label": behavior, "confidence": round(behavior_conf, 2)},
            "speed": {"posted_limit": 30, "estimated_flag": "over_limit" if speeding else "under_limit", "confidence": .74},
            "risk": {"score": risk, "signals": signals},
            "qod": {
                "triggered": triggered, "reason": "TOGG bbox > 15% frame" if active else None,
                "state": "active" if active else "baseline",
                "quality_profile": "1080p" if active else "480p",
                "bandwidth_target_mbps": 4 if active else 1,
            },
            "latency_ms": latency, "model_provider": self.name,
        }

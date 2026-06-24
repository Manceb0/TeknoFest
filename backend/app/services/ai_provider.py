from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SessionState:
    session_id: str
    frame_id: int = 0
    manual_trigger: bool = False
    incident_recorded: bool = False
    risk_window: list[float] = field(default_factory=list)
    previous_area_ratio: float = 0
    centers: list[tuple[float, float]] = field(default_factory=list)
    cached_plate: str = "SIN LECTURA"
    cached_plate_confidence: float = 0
    last_ocr_frame: int = 0
    cached_plate_roi: dict = field(default_factory=lambda: {"x": 0, "y": 0, "w": 0, "h": 0})
    # Per-session DeepSORT tracker (lazily created by the provider).
    tracker: Any = None
    track_id: str | None = None
    # Multi-frame plate voting: maps a normalized plate string to the best
    # confidence seen for it, so a stable reading wins over one-off noise.
    plate_votes: dict[str, float] = field(default_factory=dict)
    # Posted speed limit, updated when a speed sign is read via OCR.
    posted_limit: int = 0
    last_sign_frame: int = 0
    # Behavior classifier cache (label + confidence) so the heavy classifier
    # need not run on every frame.
    behavior_label: str = "not_observable"
    behavior_confidence: float = 0.0
    last_behavior_frame: int = 0
    behavior_bbox: dict | None = None
    # Occupant detection cache (driver + passengers inside the vehicle crop).
    occupant_count: int = 0
    occupant_confidence: float = 0.0
    last_occupant_frame: int = 0
    occupant_boxes: list = field(default_factory=list)
    # Best-evidence snapshot: keep the closest (largest vehicle area) frame on which
    # something notable was detected, to save as the incident photo.
    best_evidence_area: float = 0.0
    best_evidence_jpeg: bytes | None = None
    snapshot_incident_id: str | None = None


class AIProvider(ABC):
    name = "AIProvider"

    @abstractmethod
    async def process_frame(self, payload: dict, state: SessionState) -> dict: ...

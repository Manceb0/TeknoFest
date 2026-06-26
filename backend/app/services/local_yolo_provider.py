import asyncio
import base64
import re
import time
from datetime import datetime, timedelta, timezone

import cv2
import numpy as np
from ultralytics import YOLO

from .ai_provider import AIProvider, SessionState
from .qod_service import qod_service
from ..core.config import settings
from ..core.metrics import metrics

_PLATE_RE = re.compile(r"(\d{2})([A-Z]{1,3})(\d{2,4})")
_SPEED_VALUES = {20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120}


class LocalYOLOProvider(AIProvider):
    name = "LocalYOLOProvider"
    vehicle_classes = {2, 3, 5, 7}

    def __init__(self):
        self.device = settings.resolved_device
        self.imgsz = settings.yolo_imgsz
        self.model = YOLO(settings.yolo_model_path)
        # Ultralytics shares one predictor per model; calling .embed() switches
        # that predictor into embedding mode and corrupts later .predict() calls.
        # A dedicated instance keeps detection and embedding fully independent.
        self.embed_model = YOLO(settings.embed_model_path)
        # fast-plate-ocr (ONNX, plate-specific, reads "34TC8532" correctly vs EasyOCR).
        # EasyOCR kept only as fallback for speed signs.
        try:
            from fast_plate_ocr import LicensePlateRecognizer
            self._plate_recognizer = LicensePlateRecognizer("global-plates-mobile-vit-v2-model")
        except Exception:
            self._plate_recognizer = None
        import easyocr
        self._ocr_reader = easyocr.Reader(["en"], gpu=settings.use_gpu, verbose=False)
        # Driver-behavior classifier source: local .pt > Roboflow hosted > none.
        self.behavior_model = (
            YOLO(settings.behavior_model_path) if settings.behavior_model_path else None
        )
        if self.behavior_model:
            self.behavior_mode = "local"
        elif settings.roboflow_configured and settings.roboflow_project_behavior:
            self.behavior_mode = "roboflow"
        else:
            self.behavior_mode = "none"
        # Dedicated plate detector (Koushim/yolov8-license-plate-detection from HuggingFace).
        # Gives a tight plate bbox → fast-plate-ocr reads the correct text.
        self.plate_model = (
            YOLO(settings.plate_model_path) if settings.plate_model_path else None
        )

    async def process_frame(self, payload: dict, state: SessionState) -> dict:
        return await asyncio.to_thread(self._process_sync, payload, state)

    def _decode(self, payload: dict) -> np.ndarray:
        encoded = payload.get("image", "")
        if "," in encoded:
            encoded = encoded.split(",", 1)[1]
        data = np.frombuffer(base64.b64decode(encoded), dtype=np.uint8)
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Invalid JPEG frame")
        return frame

    # ------------------------------------------------------------------ tracking
    def _ensure_tracker(self, state: SessionState):
        if not settings.enable_deepsort:
            return None
        if state.tracker is None:
            from deep_sort_realtime.deepsort_tracker import DeepSort
            state.tracker = DeepSort(
                max_age=15, n_init=2, embedder="mobilenet", embedder_gpu=settings.use_gpu
            )
        return state.tracker

    def _track_vehicle(self, state, vehicles, frame):
        """Return (bbox_xyxy, track_id) for the primary vehicle using DeepSORT.

        Falls back to the largest raw detection with a static id when DeepSORT is
        disabled or has not yet confirmed a track.
        """
        largest = max(
            vehicles,
            key=lambda v: (v["xyxy"][2] - v["xyxy"][0]) * (v["xyxy"][3] - v["xyxy"][1]),
            default=None,
        )
        tracker = self._ensure_tracker(state)
        if tracker is None or not vehicles:
            if largest is None:
                return None, None
            state.track_id = state.track_id or "VEHICLE-001"
            return largest["xyxy"], state.track_id

        ds_input = []
        for v in vehicles:
            x1, y1, x2, y2 = v["xyxy"]
            ds_input.append(([x1, y1, x2 - x1, y2 - y1], v["confidence"], str(v["class_id"])))
        tracks = tracker.update_tracks(ds_input, frame=frame)
        confirmed = [t for t in tracks if t.is_confirmed()]
        if not confirmed:
            if largest is None:
                return None, None
            return largest["xyxy"], state.track_id

        def area(t):
            l, tp, r, b = t.to_ltrb()
            return (r - l) * (b - tp)

        primary = max(confirmed, key=area)
        l, tp, r, b = [int(v) for v in primary.to_ltrb()]
        state.track_id = f"VEHICLE-{int(primary.track_id):03d}"
        return (l, tp, r, b), state.track_id

    # ----------------------------------------------------------------- inference
    def _process_sync(self, payload: dict, state: SessionState) -> dict:
        started = time.perf_counter()
        frame = self._decode(payload)
        height, width = frame.shape[:2]
        state.frame_id += 1
        result = self.model.predict(
            frame, imgsz=self.imgsz, conf=settings.yolo_confidence, device=self.device,
            classes=sorted(self.vehicle_classes | {0, 67}), verbose=False,
        )[0]

        vehicles, persons, phones = [], [], []
        for box in result.boxes:
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())
            x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
            item = {"class_id": class_id, "confidence": confidence, "xyxy": (x1, y1, x2, y2)}
            if class_id in self.vehicle_classes:
                vehicles.append(item)
            elif class_id == 0:
                persons.append(item)
            elif class_id == 67:
                phones.append(item)

        vehicle_box, track_id = self._track_vehicle(state, vehicles, frame)
        detections = []
        ratio = growth = 0.0
        driver_visible = False
        smoking = False
        phone_visible = False
        occupants = 0
        occ_conf = 0.0

        if vehicle_box:
            x1, y1, x2, y2 = vehicle_box
            box_w, box_h = x2 - x1, y2 - y1
            ratio = (box_w * box_h) / (width * height)
            growth = ratio - state.previous_area_ratio
            state.previous_area_ratio = ratio
            center = ((x1 + x2) / 2 / width, (y1 + y2) / 2 / height)
            state.centers = (state.centers + [center])[-12:]
            # label the track from the most-overlapping raw detection
            label = "VEHICLE"
            best_iou = 0.0
            for v in vehicles:
                iou = self._iou(v["xyxy"], vehicle_box)
                if iou > best_iou:
                    best_iou, label = iou, self.model.names[v["class_id"]].upper()
            detections.append({
                "track_id": track_id,
                "label": label,
                "bbox": {"x": x1, "y": y1, "w": box_w, "h": box_h},
                "bbox_area_ratio": round(ratio, 4),
                "confidence": round(max((v["confidence"] for v in vehicles), default=0.0), 3),
            })

            occupants, occ_conf = self._count_occupants(frame, vehicle_box, state)
            driver_visible = occupants >= 1
            phone_visible = bool(phones)

            # Behavior runs on the full vehicle crop (driver is visible through the
            # side window at this surveillance angle), matching how the competition
            # frames were annotated.
            behavior = self._classify_behavior(frame, (x1, y1, x2, y2), state)
            smoking = behavior["smoking"]
            phone_visible = phone_visible or behavior["phone"]

            if (state.last_ocr_frame == 0 or state.frame_id - state.last_ocr_frame >= 8) and ratio > .09:
                plate_text, plate_conf, plate_roi = self._read_plate(frame, vehicle_box)
                state.last_ocr_frame = state.frame_id
                self._vote_plate(state, plate_text, plate_conf, plate_roi)

        posted_limit = self._read_speed_sign(frame, state)

        qod_was_active = qod_service.active(state.session_id)
        triggered = ratio > .15 and not qod_was_active
        if triggered:
            qod_service.sessions[state.session_id] = datetime.now(timezone.utc) + timedelta(seconds=10)
            metrics.qod_trigger_count += 1
        active = qod_service.active(state.session_id)

        swerving = self._swerving_score(state.centers)
        speeding = growth > .008
        signals = {
            "swerving": round(swerving * 30),
            "smoking": 25 if smoking else 0,
            "speeding": 25 if speeding else 0,
            "phone_use": 20 if phone_visible else 0,
        }
        raw_risk = sum(signals.values())
        state.risk_window = (state.risk_window + [raw_risk])[-6:]
        risk = round(sum(state.risk_window) / len(state.risk_window))
        latency = round((time.perf_counter() - started) * 1000, 1)
        metrics.latencies.append(latency)

        if smoking:
            behavior_label = "smoking_detected"
        elif phone_visible:
            behavior_label = "phone_detected"
        elif driver_visible:
            behavior_label = "driver_visible"
        else:
            behavior_label = "not_observable"
        behavior_conf = max(
            state.behavior_confidence,
            max([item["confidence"] for item in phones], default=0.0),
        )
        behavior_evidence = {
            "local": "YOLOv8s-cls windshield classifier",
            "roboflow": "Roboflow hosted behavior model (windshield ROI)",
            "none": "COCO person/phone detection; specialized behavior weights not loaded",
        }[self.behavior_mode]

        return {
            "session_id": state.session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "frame_id": state.frame_id,
            "frame_width": width,
            "frame_height": height,
            "mode": "qod" if active else "baseline",
            "stream_quality": "1080p" if active else "480p",
            "detections": detections,
            "plate": {
                "text": state.cached_plate,
                "confidence": round(state.cached_plate_confidence, 3),
                "roi": state.cached_plate_roi,
            },
            "behavior": {
                "label": behavior_label,
                "confidence": round(float(behavior_conf), 3),
                "evidence": behavior_evidence,
                "bbox": state.behavior_bbox if behavior_label in ("phone_detected", "smoking_detected") else None,
            },
            "occupants": {
                "count": occupants,
                "driver": 1 if occupants >= 1 else 0,
                "passengers": max(0, occupants - 1),
                "confidence": round(float(occ_conf), 3),
                "boxes": state.occupant_boxes,
            },
            "speed": {
                "posted_limit": posted_limit,
                "estimated_flag": "approaching_fast" if speeding else "stable",
                "confidence": round(min(1, abs(growth) * 45), 3),
                "bbox_growth_rate": round(growth, 5),
            },
            "risk": {"score": risk, "signals": signals},
            "qod": {
                "triggered": triggered,
                "reason": "Real vehicle bbox > 15% frame" if active else None,
                "state": "active" if active else "baseline",
                "quality_profile": "1080p" if active else "480p",
                "bandwidth_target_mbps": 4 if active else 1,
            },
            "latency_ms": latency,
            "model_provider": self.name,
            "inference_real": True,
        }

    def embed(self, frame: np.ndarray) -> list[float]:
        """Return the YOLOv8 feature embedding for a frame (for VSS storage)."""
        vector = self.embed_model.embed(frame, imgsz=self.imgsz, device=self.device, verbose=False)[0]
        return [float(x) for x in vector.tolist()]

    def embed_from_payload(self, payload: dict) -> list[float]:
        return self.embed(self._decode(payload))

    # ----------------------------------------------------------------- occupants
    def _count_occupants(self, frame, vehicle_box, state):
        """Count people (driver + passengers) inside the vehicle crop.

        Runs person detection on the upscaled vehicle region — at this
        surveillance distance occupants are too small to detect in the full
        frame, but become detectable once the cabin is cropped and enlarged.
        Throttled and cached so it does not run every frame.
        """
        if state.last_occupant_frame != 0 and state.frame_id - state.last_occupant_frame < 6:
            return state.occupant_count, state.occupant_confidence
        ox, oy = max(0, vehicle_box[0]), max(0, vehicle_box[1])
        crop = frame[oy:vehicle_box[3], ox:vehicle_box[2]]
        if crop.size == 0:
            return state.occupant_count, state.occupant_confidence
        state.last_occupant_frame = state.frame_id
        scale = max(1.0, 480.0 / max(crop.shape[1], 1))
        if scale > 1.0:
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        res = self.model.predict(crop, imgsz=640, conf=0.40, classes=[0],
                                 device=self.device, verbose=False)[0]
        ch, cw = crop.shape[:2]
        confs, boxes = [], []
        for box in res.boxes:
            bx1, by1, bx2, by2 = box.xyxy[0].tolist()
            if ((bx2 - bx1) * (by2 - by1)) / (cw * ch) >= 0.02:  # drop tiny noise
                confs.append(float(box.conf))
                boxes.append({"x": int(ox + bx1 / scale), "y": int(oy + by1 / scale),
                              "w": int((bx2 - bx1) / scale), "h": int((by2 - by1) / scale)})
        state.occupant_count = len(confs)
        state.occupant_confidence = max(confs) if confs else 0.0
        state.occupant_boxes = boxes
        return state.occupant_count, state.occupant_confidence

    # ------------------------------------------------------------------ behavior
    def _classify_behavior(self, frame, region, state) -> dict:
        """Classify the driver region (local .pt or Roboflow hosted) and cache it."""
        result = {"smoking": False, "phone": False}
        if self.behavior_mode == "none":
            return result
        if state.frame_id - state.last_behavior_frame < 5:
            return {
                "smoking": state.behavior_label == "smoking",
                "phone": state.behavior_label == "phone",
            }
        ox, oy = max(0, region[0]), max(0, region[1])
        crop = frame[oy:region[3], ox:region[2]]
        if crop.size == 0:
            return result
        # upscale the cabin crop so the seated driver is large enough to classify
        scale = max(1.0, 480.0 / max(crop.shape[1], 1))
        if scale > 1.0:
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        state.last_behavior_frame = state.frame_id
        if self.behavior_mode == "local":
            name, conf, box = self._classify_local(crop)
        else:
            name, conf, box = self._classify_roboflow(crop)
        if name is None:
            return result
        state.behavior_confidence = conf
        # map the detection box (crop coords) back to frame coords for the overlay
        if box is not None:
            bx1, by1, bx2, by2 = box
            state.behavior_bbox = {"x": int(ox + bx1 / scale), "y": int(oy + by1 / scale),
                                   "w": int((bx2 - bx1) / scale), "h": int((by2 - by1) / scale)}
        if any(k in name for k in ("smok", "cigar")):
            state.behavior_label, result["smoking"] = "smoking", True
        elif "phone" in name or "cell" in name or "call" in name:
            state.behavior_label, result["phone"] = "phone", True
        else:
            state.behavior_label, state.behavior_bbox = "safe", None
        return result

    def _classify_local(self, crop):
        prediction = self.behavior_model.predict(crop, imgsz=512, conf=0.20, device=self.device, verbose=False)[0]
        probs = getattr(prediction, "probs", None)
        if probs is not None:  # classification model (no box)
            return prediction.names[int(probs.top1)].lower(), float(probs.top1conf), None
        boxes = getattr(prediction, "boxes", None)
        if boxes is not None and len(boxes):  # detection model
            best = max(boxes, key=lambda b: float(b.conf))
            return prediction.names[int(best.cls)].lower(), float(best.conf), [int(v) for v in best.xyxy[0].tolist()]
        return None, 0.0, None

    def _classify_roboflow(self, crop):
        from .roboflow_service import roboflow_service
        ok, buffer = cv2.imencode(".jpg", crop)
        if not ok:
            return None, 0.0, None
        try:
            response = roboflow_service.infer_sync(buffer.tobytes(), "behavior")
        except Exception:
            return None, 0.0, None
        name, conf = self._parse_roboflow_behavior(response)
        return name, conf, None

    @staticmethod
    def _parse_roboflow_behavior(response):
        """Extract the top (class, confidence) from a Roboflow detect/classify reply."""
        if not response:
            return None, 0.0
        best_name, best_conf = None, 0.0
        preds = response.get("predictions")
        items = []
        if isinstance(preds, list):
            items = preds
        elif isinstance(preds, dict):
            # classification endpoints return {"predictions": {class: {confidence}}}
            items = [{"class": k, "confidence": v.get("confidence", 0)} for k, v in preds.items()]
        for item in items:
            conf = float(item.get("confidence", 0))
            if conf > best_conf:
                best_conf, best_name = conf, str(item.get("class", "")).lower()
        if best_name is None and response.get("top"):
            best_name = str(response["top"]).lower()
            best_conf = float(response.get("confidence", 0))
        return best_name, best_conf

    # --------------------------------------------------------------- speed signs
    def _read_speed_sign(self, frame, state) -> int:
        """Best-effort OCR of a posted speed-limit sign; cached between reads."""
        if state.frame_id - state.last_sign_frame >= 30:
            state.last_sign_frame = state.frame_id
            small = cv2.resize(frame, (640, int(640 * frame.shape[0] / frame.shape[1])))
            try:
                readings = self._ocr_reader.readtext(
                    small, detail=1, allowlist="0123456789"
                )
            except Exception:
                readings = []
            for _, text, conf in readings:
                digits = re.sub(r"\D", "", text)
                if digits and int(digits) in _SPEED_VALUES and conf > .4:
                    state.posted_limit = int(digits)
                    break
        return state.posted_limit or settings.default_speed_limit

    # --------------------------------------------------------------------- plate
    def _vote_plate(self, state, text, conf, roi):
        """Accumulate plate readings across frames so a stable reading wins."""
        if not text:
            return
        weight = conf * (1.6 if _PLATE_RE.fullmatch(text.replace(" ", "")) else 1.0)
        score, best_conf, best_roi = state.plate_votes.get(text, (0.0, 0.0, roi))
        score += weight
        if conf >= best_conf:
            best_conf, best_roi = conf, roi
        state.plate_votes[text] = (score, best_conf, best_roi)
        winner = max(state.plate_votes, key=lambda t: state.plate_votes[t][0])
        _, win_conf, win_roi = state.plate_votes[winner]
        state.cached_plate = winner
        state.cached_plate_confidence = win_conf
        state.cached_plate_roi = win_roi

    @staticmethod
    def _iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _center_inside(box, region):
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        rx1, ry1, rx2, ry2 = region
        return rx1 <= cx <= rx2 and ry1 <= cy <= ry2

    @staticmethod
    def _swerving_score(centers):
        if len(centers) < 5:
            return 0
        xs = [point[0] for point in centers]
        direction_changes = sum(1 for a, b, c in zip(xs, xs[1:], xs[2:]) if (b - a) * (c - b) < 0)
        lateral_range = max(xs) - min(xs)
        return min(1.0, lateral_range * 5 + direction_changes * .12)

    def _ocr_crop(self, crop, x_offset, y_offset):
        """Run CLAHE + Otsu + EasyOCR on a plate crop; return (text, conf, roi_dict)."""
        crop = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        enhanced = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4)).apply(gray)
        soft = cv2.GaussianBlur(enhanced, (0, 0), 3)
        enhanced = cv2.addWeighted(enhanced, 2, soft, -1, 0)
        readings = []
        for variant in (enhanced, gray):
            results = self._ocr_reader.readtext(variant, detail=1, paragraph=True,
                                                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ")
            for result in results:
                text = re.sub(r"\s+", " ", re.sub(r"[^A-Z0-9 ]", "", result[1].upper())).strip()
                confidence = float(result[2]) if len(result) > 2 else .5
                compact = text.replace(" ", "")
                match = _PLATE_RE.fullmatch(compact)
                if match:
                    normalized = f"{match.group(1)} {match.group(2)} {match.group(3)}"
                    readings.append((1 + confidence, confidence, normalized))
                elif len(compact) >= 5:
                    readings.append((confidence, confidence, text))
        if readings:
            _, confidence, text = max(readings)
            return text, confidence, {"x": x_offset, "y": y_offset,
                                      "w": crop.shape[1] // 4, "h": crop.shape[0] // 4}
        return "", 0, {"x": 0, "y": 0, "w": 0, "h": 0}

    def _fast_plate_ocr(self, crop, x_offset, y_offset):
        """Use fast-plate-ocr (ONNX, plate-specific) on a BGR crop."""
        if self._plate_recognizer is None:
            return None
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        pred = self._plate_recognizer.run(gray)[0]
        text = pred.plate.strip() if hasattr(pred, "plate") else str(pred).strip()
        text = re.sub(r"[^A-Z0-9]", "", text.upper())
        if len(text) < 4:
            return None
        m = _PLATE_RE.fullmatch(text)
        if m:
            normalized = f"{m.group(1)} {m.group(2)} {m.group(3)}"
            return normalized, 0.95, {"x": x_offset, "y": y_offset,
                                      "w": crop.shape[1], "h": crop.shape[0]}
        return text, 0.7, {"x": x_offset, "y": y_offset,
                           "w": crop.shape[1], "h": crop.shape[0]}

    def _read_plate(self, frame, vehicle_box):
        x1, y1, x2, y2 = vehicle_box
        # --- Path A: dedicated detector → fast-plate-ocr (correctly reads "34 TC 8532") ---
        if self.plate_model is not None:
            r = self.plate_model.predict(frame, imgsz=640, conf=0.20,
                                         device=self.device, verbose=False)[0]
            if r.boxes:
                best = max(r.boxes, key=lambda b: float(b.conf))
                px1, py1, px2, py2 = [int(v) for v in best.xyxy[0].tolist()]
                pad = 4
                cx1 = max(0, px1 - pad); cy1 = max(0, py1 - pad)
                cx2 = min(frame.shape[1], px2 + pad); cy2 = min(frame.shape[0], py2 + pad)
                crop = frame[cy1:cy2, cx1:cx2]
                if crop.size > 0:
                    result = self._fast_plate_ocr(crop, cx1, cy1)
                    if result:
                        return result
                    return self._ocr_crop(crop, cx1, cy1)
        # --- Path B: contour-based heuristic on lower half of vehicle box ---
        box_h = y2 - y1
        lower_y = y1 + int(box_h * .48)
        roi = frame[lower_y:y2, x1:x2]
        if roi.size == 0:
            return "", 0, {"x": 0, "y": 0, "w": 0, "h": 0}
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        _, threshold = cv2.threshold(blurred, 130, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for contour in contours:
            px, py, pw, ph = cv2.boundingRect(contour)
            aspect = pw / max(ph, 1)
            if 2.0 < aspect < 7.5 and pw > 35 and ph > 8 and pw * ph > 250:
                candidates.append((pw * ph, px, py, pw, ph))
        for _, px, py, pw, ph in sorted(candidates, reverse=True)[:4]:
            pad = 4
            crop = roi[max(0, py-pad):min(roi.shape[0], py+ph+pad),
                       max(0, px-pad):min(roi.shape[1], px+pw+pad)]
            text, conf, roi_dict = self._ocr_crop(crop, x1 + px, lower_y + py)
            if text:
                return text, conf, roi_dict
        return "", 0, {"x": 0, "y": 0, "w": 0, "h": 0}

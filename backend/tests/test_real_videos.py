"""End-to-end smoke test against the three supplied TEKNOFEST clips."""
from pathlib import Path

import cv2
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]
VIDEOS = sorted((ROOT / "frontend" / "public" / "demo-videos").glob("*.mp4"))


def main():
    assert len(VIDEOS) == 3, f"Expected 3 demo videos, found {len(VIDEOS)}"
    model = YOLO(str(ROOT / "backend" / "yolov8n.pt"))
    for path in VIDEOS:
        cap = cv2.VideoCapture(str(path))
        assert cap.isOpened(), f"Cannot open {path.name}"
        areas = []
        confidences = []
        for second in (1, 2, 3, 4, 5):
            cap.set(cv2.CAP_PROP_POS_MSEC, second * 1000)
            ok, frame = cap.read()
            if not ok:
                continue
            result = model.predict(frame, imgsz=512, conf=.18, classes=[2, 3, 5, 7], verbose=False)[0]
            if not result.boxes:
                continue
            box = max(result.boxes, key=lambda item: float((item.xyxy[0][2] - item.xyxy[0][0]) * (item.xyxy[0][3] - item.xyxy[0][1])))
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            areas.append((x2 - x1) * (y2 - y1) / (frame.shape[1] * frame.shape[0]))
            confidences.append(float(box.conf))
        cap.release()
        assert len(areas) >= 3, f"{path.name}: vehicle was not detected consistently"
        assert max(confidences) >= .65, f"{path.name}: weak confidence {max(confidences):.2f}"
        assert max(areas) > areas[0] * 1.8, f"{path.name}: approach was not measured"
        print(
            f"PASS {path.name}: {len(areas)} detections, "
            f"max confidence={max(confidences):.3f}, "
            f"area={areas[0]:.3f}->{max(areas):.3f}"
        )


if __name__ == "__main__":
    main()

"""Detached behavior-model training run (survives session restarts).

Trains YOLOv8s on the cited Roboflow 'distracted-driving' dataset (which includes
Safe Driving negatives + phone/texting/distraction classes), with extra
brightness/blur augmentation to push toward the night surveillance domain.

Launched via Start-Process so it runs independently of the agent session.
Progress: backend/runs/behavior/results.csv  ·  weights: runs/behavior/weights/best.pt
"""
import os
from ultralytics import YOLO

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(os.path.dirname(BACKEND), "tmp", "distracted_driving", "data.yaml")

if __name__ == "__main__":
    model = YOLO(os.path.join(BACKEND, "yolov8s.pt"))
    model.train(
        data=DATA,
        epochs=30, imgsz=512, batch=16, device=0,
        project=os.path.join(BACKEND, "runs"), name="behavior", exist_ok=True,
        hsv_v=0.6, hsv_s=0.5, degrees=5, translate=0.1, scale=0.5, fliplr=0.5,
        mosaic=1.0, patience=12, plots=True, verbose=True,
    )
    print("TRAIN_DONE:", os.path.join(BACKEND, "runs", "behavior", "weights", "best.pt"))

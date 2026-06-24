"""Detached training of the FOCUSED behavior model: phone vs safe.

The full 13-class distracted-driving model diluted the phone classes (and the
original test split had only 3 phone boxes). This trains a focused 2-class
detector (phone = texting + talking-on-phone, safe = safe driving) on a
re-stratified 70/15/15 split where every split contains phone examples.

Progress: runs/behavior_focused/results.csv  ·  weights: .../weights/best.pt
"""
import os
from ultralytics import YOLO

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(os.path.dirname(BACKEND), "tmp", "behavior_ds", "data.yaml")

if __name__ == "__main__":
    model = YOLO(os.path.join(BACKEND, "yolov8s.pt"))
    model.train(
        data=DATA, epochs=40, imgsz=512, batch=16, device=0,
        project=os.path.join(BACKEND, "runs"), name="behavior_focused", exist_ok=True,
        hsv_v=0.6, hsv_s=0.5, degrees=5, translate=0.1, scale=0.5, fliplr=0.5,
        mosaic=1.0, patience=15, plots=True, verbose=True,
    )
    print("FOCUSED_TRAIN_DONE:", os.path.join(BACKEND, "runs", "behavior_focused", "weights", "best.pt"))

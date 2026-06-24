"""Detached fine-tune on the COMBINED dataset (public phone/safe + competition
cigarette/phone), classes ['phone','cigarette','safe'], with heavy night
augmentation for domain adaptation. Weights: runs/behavior_combined/weights/best.pt
"""
import os
from ultralytics import YOLO

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(os.path.dirname(BACKEND), "tmp", "behavior_combined", "data.yaml")

if __name__ == "__main__":
    model = YOLO(os.path.join(BACKEND, "yolov8s.pt"))
    model.train(
        data=DATA, epochs=40, imgsz=512, batch=16, device=0, workers=0,
        project=os.path.join(BACKEND, "runs"), name="behavior_combined", exist_ok=True,
        hsv_v=0.7, hsv_s=0.5, degrees=6, translate=0.12, scale=0.6, fliplr=0.5,
        mosaic=1.0, mixup=0.1, patience=15, plots=True, verbose=True,
    )
    print("COMBINED_TRAIN_DONE:", os.path.join(BACKEND, "runs", "behavior_combined", "weights", "best.pt"))

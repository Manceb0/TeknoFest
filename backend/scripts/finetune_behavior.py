"""Fine-tune the behavior base model on YOUR annotated competition frames.

The base model (runs/behavior/weights/best.pt) scores mAP50≈0.99 on the public
distracted-driving dataset but does NOT transfer to the night/surveillance clips.
This fine-tunes it on the domain frames so it works on real footage.

Prerequisite: in Roboflow project 'afrikaans/quismotion-driver-behavior', add
`phone` / `cigarette` boxes (occupants are already pre-labeled), then Generate a
Version. Pass that version number here.

    python scripts/finetune_behavior.py --version <N> --api-key <KEY>

Result weights: runs/behavior_ft/weights/best.pt  ->  set BEHAVIOR_MODEL_PATH to it.
"""
import argparse
import os

from ultralytics import YOLO

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-key", default=os.environ.get("ROBOFLOW_API_KEY"))
    ap.add_argument("--workspace", default="afrikaans")
    ap.add_argument("--project", default="quismotion-driver-behavior")
    ap.add_argument("--version", type=int, required=True, help="Roboflow version after annotating")
    ap.add_argument("--base", default=os.path.join(BACKEND, "runs", "behavior", "weights", "best.pt"))
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    if not args.api_key:
        raise SystemExit("ERROR: pass --api-key or set ROBOFLOW_API_KEY.")

    from roboflow import Roboflow
    proj = Roboflow(api_key=args.api_key).workspace(args.workspace).project(args.project)
    ds = proj.version(args.version).download(
        "yolov8", location=os.path.join(os.path.dirname(BACKEND), "tmp", "cabin_rf"), overwrite=True)
    print("dataset:", ds.location)

    base = args.base if os.path.exists(args.base) else "yolov8s.pt"
    print("fine-tuning from:", base)
    model = YOLO(base)
    model.train(
        data=os.path.join(ds.location, "data.yaml"),
        epochs=args.epochs, imgsz=args.imgsz, batch=16, device=args.device,
        project=os.path.join(BACKEND, "runs"), name="behavior_ft", exist_ok=True,
        # strong augmentation: few domain images, push generalization
        hsv_v=0.7, hsv_s=0.5, degrees=8, translate=0.12, scale=0.6,
        fliplr=0.5, mosaic=1.0, mixup=0.1, patience=25, plots=True,
    )
    out = os.path.join(BACKEND, "runs", "behavior_ft", "weights", "best.pt")
    print("FT_DONE ->", out)
    print("Set BEHAVIOR_MODEL_PATH to this path in backend/.env, then restart the backend.")


if __name__ == "__main__":
    main()

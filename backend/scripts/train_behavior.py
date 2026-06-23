"""Download a cited Roboflow dataset and train the driver-behavior classifier.

Implements the PDR behavior stage: a fine-tuned YOLOv8s-cls model for smoking /
phone use, trained on the cited Roboflow Universe datasets.

Requires a (free) Roboflow API key for the download step only — Roboflow gates
dataset access behind a key even for public Universe projects.

Example (cited "Distracted Driving" dataset, ipylot-project):
    python scripts/train_behavior.py \
        --api-key YOUR_KEY \
        --workspace ipylot-project \
        --project distracted-driving-v2wk5 \
        --version 1 \
        --epochs 30 --device cuda:0

The resulting weights land in runs/classify/<name>/weights/best.pt. Point the
backend at them with BEHAVIOR_MODEL_PATH in backend/.env.
"""
import argparse
import os

from roboflow import Roboflow
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8s-cls driver-behavior model.")
    parser.add_argument("--api-key", default=os.environ.get("ROBOFLOW_API_KEY"))
    parser.add_argument("--workspace", default="ipylot-project")
    parser.add_argument("--project", default="distracted-driving-v2wk5")
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--task", choices=["classify", "detect"], default="classify")
    parser.add_argument("--export-format", default="folder",
                        help='Roboflow export: "folder" for classify, "yolov8" for detect')
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--device", default="cpu", help='"cpu" or "cuda:0"')
    parser.add_argument("--name", default="behavior")
    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit("ERROR: pass --api-key or set ROBOFLOW_API_KEY.")

    rf = Roboflow(api_key=args.api_key)
    project = rf.workspace(args.workspace).project(args.project)
    dataset = project.version(args.version).download(args.export_format)
    print(f"Dataset downloaded to: {dataset.location}")

    base = "yolov8s-cls.pt" if args.task == "classify" else "yolov8s.pt"
    data = dataset.location if args.task == "classify" else f"{dataset.location}/data.yaml"
    model = YOLO(base)
    model.train(
        data=data, epochs=args.epochs, imgsz=args.imgsz,
        device=args.device, name=args.name,
    )
    print("\nDone. Set BEHAVIOR_MODEL_PATH to the produced weights/best.pt")


if __name__ == "__main__":
    main()

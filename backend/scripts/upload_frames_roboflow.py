"""Extract driver-visible frames from the demo videos and upload them to a
Roboflow project for manual annotation (smoking / phone use).

This is the PDR's domain-adaptation step: build the "competition footage" dataset
from real surveillance-angle frames, since public datasets do not transfer to this
night/side-angle domain.

Usage:
    ROBOFLOW_API_KEY=xxx python scripts/upload_frames_roboflow.py
    # extract only, do not touch the Roboflow account:
    python scripts/upload_frames_roboflow.py --no-upload
"""
import argparse
import glob
import os

import cv2
from ultralytics import YOLO


def extract(videos_dir, out_dir, stride, min_area):
    model = YOLO("yolov8s.pt")
    os.makedirs(out_dir, exist_ok=True)
    saved = []
    for video in sorted(glob.glob(os.path.join(videos_dir, "*.mp4"))):
        name = os.path.splitext(os.path.basename(video))[0]
        cap = cv2.VideoCapture(video)
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                h, w = frame.shape[:2]
                result = model.predict(frame, imgsz=512, conf=0.25, classes=[2, 3, 5, 7], verbose=False)[0]
                if len(result.boxes):
                    box = max(result.boxes, key=lambda b: (b.xyxy[0][2] - b.xyxy[0][0]) * (b.xyxy[0][3] - b.xyxy[0][1]))
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    area = ((x2 - x1) * (y2 - y1)) / (w * h)
                    if area >= min_area:
                        path = os.path.join(out_dir, f"{name}_{idx:04d}.jpg")
                        cv2.imwrite(path, frame)
                        saved.append((path, name))
            idx += 1
        cap.release()
    return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-key", default=os.environ.get("ROBOFLOW_API_KEY"))
    ap.add_argument("--workspace", default="afrikaans")
    ap.add_argument("--project", default="quismotion-driver-behavior")
    ap.add_argument("--videos", default="../frontend/public/demo-videos")
    ap.add_argument("--out", default="../tmp/dataset_frames")
    ap.add_argument("--stride", type=int, default=10, help="Keep every Nth frame")
    ap.add_argument("--min-area", type=float, default=0.04, help="Min vehicle bbox area ratio")
    ap.add_argument("--no-upload", action="store_true")
    args = ap.parse_args()

    saved = extract(args.videos, args.out, args.stride, args.min_area)
    print(f"EXTRACTED {len(saved)} frames to {args.out}")
    if args.no_upload:
        return
    if not args.api_key:
        raise SystemExit("ERROR: pass --api-key or set ROBOFLOW_API_KEY.")

    from roboflow import Roboflow
    workspace = Roboflow(api_key=args.api_key).workspace(args.workspace)
    try:
        project = workspace.create_project(args.project, "object-detection", "MIT", "behavior")
        print(f"CREATED project '{args.project}'")
    except Exception as exc:
        print("create_project note:", repr(exc)[:160])
        project = workspace.project(args.project)

    uploaded = 0
    for i, (path, name) in enumerate(saved):
        try:
            project.single_upload(image_path=path, batch_name="quismotion-demo",
                                  tag_names=[name], split="train")
            uploaded += 1
        except Exception as exc:
            print("upload err", os.path.basename(path), repr(exc)[:120])
        if (i + 1) % 20 == 0:
            print(f"  uploaded {i + 1}/{len(saved)}")
    print(f"UPLOADED {uploaded}/{len(saved)} -> {args.workspace}/{args.project}")


if __name__ == "__main__":
    main()

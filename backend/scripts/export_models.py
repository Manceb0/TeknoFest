"""Export QuisMotion YOLO models to ONNX with optional FP16 / INT8 quantization.

This implements the PDR optimization layer: "Models are exported to ONNX for
runtime efficiency" and "Quantization experiments (FP16/INT8 where feasible)".

Examples
--------
# FP32 ONNX (works on CPU and GPU)
python scripts/export_models.py --model yolov8s.pt

# FP16 ONNX (GPU recommended, e.g. RTX 4060)
python scripts/export_models.py --model yolov8x.pt --half --device cuda:0

# INT8 ONNX (needs a calibration dataset yaml)
python scripts/export_models.py --model yolov8s.pt --int8 --data coco128.yaml

After exporting, point the backend at the result:
    YOLO_MODEL_PATH=yolov8s.onnx
The embedding model (EMBED_MODEL_PATH) should stay a .pt file.
"""
import argparse
import sys
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Export YOLO weights to ONNX.")
    parser.add_argument("--model", default="yolov8s.pt", help="Source .pt weights")
    parser.add_argument("--imgsz", type=int, default=512, help="Inference image size")
    parser.add_argument("--device", default="cpu", help='"cpu", "cuda:0", ...')
    parser.add_argument("--half", action="store_true", help="FP16 export (GPU)")
    parser.add_argument("--int8", action="store_true", help="INT8 export (needs --data)")
    parser.add_argument("--data", default=None, help="Calibration dataset yaml for INT8")
    parser.add_argument("--dynamic", action="store_true", help="Dynamic input shapes")
    args = parser.parse_args()

    if args.half and args.device == "cpu":
        print("WARNING: FP16 is only meaningful on GPU; export will likely fall back to FP32.")
    if args.int8 and not args.data:
        print("ERROR: INT8 export requires --data <calibration.yaml>.", file=sys.stderr)
        return 1

    src = Path(args.model)
    if not src.exists():
        print(f"Loading '{args.model}' (Ultralytics will fetch it if it is a known name).")
    model = YOLO(args.model)

    out = model.export(
        format="onnx",
        imgsz=args.imgsz,
        half=args.half,
        int8=args.int8,
        data=args.data,
        dynamic=args.dynamic,
        simplify=True,
        device=args.device,
    )
    precision = "INT8" if args.int8 else ("FP16" if args.half else "FP32")
    print(f"\nExported {args.model} -> {out}  ({precision}, imgsz={args.imgsz})")
    print("Set YOLO_MODEL_PATH to this file in backend/.env to use it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

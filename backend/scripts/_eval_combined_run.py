"""Detached/standalone test-set eval of the COMBINED behavior model
(phone/cigarette/safe). Writes a full summary (overall P/R/F1/mAP + per-class + FPS)
to runs/combined_test/eval.json, matching what notebook 07 consumes."""
import os, json, time, glob
import cv2
import torch
from ultralytics import YOLO

B = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(B, "runs", "combined_test")
os.makedirs(OUT, exist_ok=True)
DEV = 0 if torch.cuda.is_available() else "cpu"

m = YOLO(os.path.join(B, "runs", "behavior_combined", "weights", "best.pt"))
r = m.val(split="test", project=os.path.join(B, "runs"), name="combined_test",
          exist_ok=True, plots=True, verbose=False, device=DEV, workers=0).box

f1 = 2 * r.mp * r.mr / (r.mp + r.mr + 1e-9)
per_class = []
for i, ci in enumerate(r.ap_class_index):
    cf1 = 2 * r.p[i] * r.r[i] / (r.p[i] + r.r[i] + 1e-9)
    per_class.append({"class": m.names[int(ci)], "P": round(float(r.p[i]), 3),
                      "R": round(float(r.r[i]), 3), "F1": round(float(cf1), 3),
                      "mAP50": round(float(r.ap50[i]), 3)})

# FPS on test images
imgs = [cv2.imread(f) for f in glob.glob(os.path.join(B, "..", "tmp", "behavior_combined", "test", "images", "*.jpg"))[:120]]
if imgs:
    m.predict(imgs[0], imgsz=512, device=DEV, verbose=False)
    t = time.perf_counter()
    for im in imgs:
        m.predict(im, imgsz=512, device=DEV, verbose=False)
    fps = round(len(imgs) / (time.perf_counter() - t), 1)
else:
    fps = 0.0

summary = {
    "device": "GPU" if DEV == 0 else "CPU",
    "overall": {"precision": round(float(r.mp), 3), "recall": round(float(r.mr), 3),
                "F1": round(float(f1), 3), "mAP50": round(float(r.map50), 3),
                "mAP50_95": round(float(r.map), 3)},
    "per_class": per_class,
    "fps_gpu_512": fps,
}
json.dump(summary, open(os.path.join(OUT, "eval.json"), "w"), indent=2)
print("EVAL_DONE", json.dumps(summary))

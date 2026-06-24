"""Detached evaluation of the behavior model on the TEST split (+ FPS).

Writes a JSON summary so the result survives session restarts.
Output: runs/behavior_test/eval_summary.json  (+ Ultralytics plots in runs/behavior_test/)
"""
import os, json, time, glob
import cv2
from ultralytics import YOLO

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BACKEND, "runs", "behavior_test")

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    m = YOLO(os.path.join(BACKEND, "runs", "behavior", "weights", "best.pt"))
    res = m.val(split="test", project=os.path.join(BACKEND, "runs"),
                name="behavior_test", exist_ok=True, plots=True, verbose=False, device=0)
    b = res.box
    f1 = 2 * b.mp * b.mr / (b.mp + b.mr + 1e-9)
    names = m.names
    per_class = []
    for i, ci in enumerate(b.ap_class_index):
        pc_f1 = 2 * b.p[i] * b.r[i] / (b.p[i] + b.r[i] + 1e-9)
        per_class.append({"class": names[int(ci)], "P": round(float(b.p[i]), 3),
                          "R": round(float(b.r[i]), 3), "F1": round(float(pc_f1), 3),
                          "mAP50": round(float(b.ap50[i]), 3)})
    # FPS on test images (GPU, single-frame, 512px)
    imgs = [cv2.imread(f) for f in glob.glob(os.path.join(BACKEND, "..", "tmp", "distracted_driving", "test", "images", "*.jpg"))[:100]]
    m.predict(imgs[0], imgsz=512, device=0, verbose=False)
    t = time.perf_counter()
    for im in imgs:
        m.predict(im, imgsz=512, device=0, verbose=False)
    fps = len(imgs) / (time.perf_counter() - t)

    summary = {
        "overall": {"precision": round(float(b.mp), 3), "recall": round(float(b.mr), 3),
                    "F1": round(float(f1), 3), "mAP50": round(float(b.map50), 3),
                    "mAP50_95": round(float(b.map), 3)},
        "per_class": per_class,
        "fps_gpu_512": round(fps, 1),
        "test_images": len(glob.glob(os.path.join(BACKEND, "..", "tmp", "distracted_driving", "test", "images", "*.jpg"))),
    }
    with open(os.path.join(OUT, "eval_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print("EVAL_DONE", json.dumps(summary))

"""Auto-label competition frames + merge with the public dataset, then write a
combined YOLO dataset for fine-tuning behavior with a `cigarette` class.

Classes: ['phone', 'cigarette', 'safe']
- public (behavior_ds): phone (real, abundant, bright domain), safe
- competition clips: weak clip-prior labels at the driver's head region
  (tekno-01 = cigarette/smoking, tekno-02 = phone, tekno-03 = safe), for night-
  domain adaptation. These are weak labels (low-res) — meant to be human-refined.
"""
import os, glob, shutil, random
import cv2
from ultralytics import YOLO

random.seed(42)
BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(BACKEND)
PUB = os.path.join(ROOT, "tmp", "behavior_ds")        # public: 0=phone,1=safe
FRAMES = os.path.join(ROOT, "tmp", "dataset_frames")  # competition frames
DST = os.path.join(ROOT, "tmp", "behavior_combined")
NAMES = ["phone", "cigarette", "safe"]
CLIP_BEHAVIOR = {"tekno-01": 1, "tekno-02": 0, "tekno-03": 2}  # cig, phone, safe

veh = YOLO(os.path.join(BACKEND, "yolov8x.pt"))


def driver_head_box(img):
    """Return the driver's head/hand region in frame coords, or None."""
    h, w = img.shape[:2]
    r = veh.predict(img, imgsz=512, conf=0.25, classes=[2, 3, 5, 7], device=0, verbose=False)[0]
    if not len(r.boxes):
        return None
    b = max(r.boxes, key=lambda b: (b.xyxy[0][2]-b.xyxy[0][0])*(b.xyxy[0][3]-b.xyxy[0][1]))
    x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
    ox, oy = max(0, x1), max(0, y1)
    crop = img[oy:y2, ox:x2]
    if crop.size == 0:
        return None
    sc = max(1.0, 480.0 / max(crop.shape[1], 1))
    big = cv2.resize(crop, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)
    pr = veh.predict(big, imgsz=640, conf=0.40, classes=[0], device=0, verbose=False)[0]
    if not len(pr.boxes):
        return None
    p = max(pr.boxes, key=lambda b: float(b.conf))
    px1, py1, px2, py2 = [v / sc for v in p.xyxy[0].tolist()]
    # head/hand region = upper 55% of the driver box (where cigarette/phone sit)
    fx1, fy1 = ox + px1, oy + py1
    pw, ph = px2 - px1, py2 - py1
    hx1, hy1, hx2, hy2 = fx1, fy1, fx1 + pw, fy1 + ph * 0.55
    return (hx1, hy1, hx2, hy2, w, h)


def main():
    for split in ["train", "valid", "test"]:
        os.makedirs(f"{DST}/{split}/images", exist_ok=True)
        os.makedirs(f"{DST}/{split}/labels", exist_ok=True)

    # 1) copy public dataset, remap safe 1->2 (phone stays 0)
    pub_n = 0
    for split in ["train", "valid", "test"]:
        for img in glob.glob(f"{PUB}/{split}/images/*.jpg"):
            base = os.path.splitext(os.path.basename(img))[0]
            lbl = f"{PUB}/{split}/labels/{base}.txt"
            lines = []
            if os.path.exists(lbl):
                for ln in open(lbl):
                    p = ln.split()
                    if not p:
                        continue
                    c = int(p[0]); c = 2 if c == 1 else 0  # safe->2, phone->0
                    lines.append(" ".join([str(c)] + p[1:]))
            shutil.copy(img, f"{DST}/{split}/images/pub_{base}.jpg")
            open(f"{DST}/{split}/labels/pub_{base}.txt", "w").write("\n".join(lines))
            pub_n += 1

    # 2) auto-label competition frames (weak clip-prior), stratified 70/15/15
    comp = {"train": 0, "valid": 0, "test": 0}
    by_clip = {}
    for f in sorted(glob.glob(f"{FRAMES}/*.jpg")):
        clip = os.path.basename(f).split("_")[0]
        if clip in CLIP_BEHAVIOR:
            by_clip.setdefault(clip, []).append(f)
    for clip, files in by_clip.items():
        random.shuffle(files); n = len(files); a, b = int(n*.7), int(n*.85)
        for split, chunk in [("train", files[:a]), ("valid", files[a:b]), ("test", files[b:])]:
            for f in chunk:
                img = cv2.imread(f); box = driver_head_box(img)
                base = os.path.splitext(os.path.basename(f))[0]
                lines = []
                if box:
                    hx1, hy1, hx2, hy2, w, h = box
                    cls = CLIP_BEHAVIOR[clip]
                    if cls != 2:  # phone/cigarette -> a box at the head region
                        cx, cy = ((hx1+hx2)/2)/w, ((hy1+hy2)/2)/h
                        bw, bh = (hx2-hx1)/w, (hy2-hy1)/h
                        lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                    # tekno-03 (safe) -> keep image as background negative (no box)
                shutil.copy(f, f"{DST}/{split}/images/{base}.jpg")
                open(f"{DST}/{split}/labels/{base}.txt", "w").write("\n".join(lines))
                comp[split] += 1

    open(f"{DST}/data.yaml", "w").write(
        f"path: {DST}\ntrain: train/images\nval: valid/images\ntest: test/images\nnc: 3\nnames: {NAMES}\n")
    print(f"COMBINED dataset -> {DST}")
    print(f"public images: {pub_n} | competition images: {comp}")


if __name__ == "__main__":
    main()

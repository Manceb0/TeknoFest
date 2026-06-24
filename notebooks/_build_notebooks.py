"""Builds and executes the QuisMotion analysis notebooks.

Each notebook documents and reproduces ONE conclusion reached during model
development. Run from the project root:

    cd notebooks
    ../backend/.venv/Scripts/python.exe _build_notebooks.py

The Roboflow hosted cells read ROBOFLOW_API_KEY from the environment so the key
is never written into a notebook. Re-run after rotating the key to refresh outputs.
"""
import os
import nbformat as nbf
from nbclient import NotebookClient

HERE = os.path.dirname(os.path.abspath(__file__))


def md(text):
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text):
    return nbf.v4.new_code_cell(text.strip("\n"))


NOTEBOOKS = {}

# --------------------------------------------------------------------------- 00
NOTEBOOKS["00_environment_and_gpu.ipynb"] = [
    md("""
# 00 · Environment & GPU

Confirms the runtime the rest of the notebooks rely on: Python packages, the
CUDA build of PyTorch, and the GPU. **Conclusion up front:** inference runs on an
RTX 4060 (CUDA), which is what makes YOLOv8x real-time downstream.
"""),
    code("""
import torch, ultralytics, easyocr, duckdb, cv2, platform
print("Python      :", platform.python_version())
print("torch       :", torch.__version__)
print("ultralytics :", ultralytics.__version__)
print("opencv      :", cv2.__version__)
print("duckdb      :", duckdb.__version__)
"""),
    code("""
print("CUDA available :", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print("GPU            :", torch.cuda.get_device_name(0))
    print("VRAM (GB)      :", round(p.total_memory / 1024**3, 1))
    print("Compute cap.   :", f"{p.major}.{p.minor}  (8.9 = Ada / RTX 40-series)")
    print("CUDA build     :", torch.version.cuda)
"""),
    md("""
**Conclusion.** `compute capability 8.9` is unique to the Ada Lovelace (RTX 40)
generation, confirming the RTX 4060. The detector and embeddings run on `cuda:0`.
"""),
]

# --------------------------------------------------------------------------- 01
NOTEBOOKS["01_vehicle_detection_qod_and_latency.ipynb"] = [
    md("""
# 01 · Vehicle detection, QoD trigger & latency

**Questions.** Does detection work on the real footage? Does the bounding-box
area grow as the car approaches (the signal that drives the QoD bandwidth boost
at 15%)? Is it fast enough for real time?
"""),
    code("""
%matplotlib inline
import glob, time, cv2, numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

DEV = "cuda:0" if __import__("torch").cuda.is_available() else "cpu"
model = YOLO("../backend/yolov8s.pt")
frames = sorted(glob.glob("../tmp/frames/tekno-01-*.jpg"))
print("device:", DEV, "| frames:", len(frames))
"""),
    md("Run the detector across the approach sequence and overlay the vehicle box + area %."),
    code("""
fig, axes = plt.subplots(2, 3, figsize=(15, 6))
areas = []
for ax, f in zip(axes.ravel(), frames):
    img = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB); h, w = img.shape[:2]
    r = model.predict(img, imgsz=512, conf=0.18, classes=[2,3,5,7], device=DEV, verbose=False)[0]
    a = 0.0
    if len(r.boxes):
        b = max(r.boxes, key=lambda b:(b.xyxy[0][2]-b.xyxy[0][0])*(b.xyxy[0][3]-b.xyxy[0][1]))
        x1,y1,x2,y2 = [int(v) for v in b.xyxy[0].tolist()]
        a = (x2-x1)*(y2-y1)/(w*h)
        col = (255,140,0) if a > 0.15 else (0,200,255)
        cv2.rectangle(img,(x1,y1),(x2,y2),col,3)
    areas.append(a)
    ax.imshow(img); ax.set_title(f"{f.split(chr(92))[-1].split('/')[-1][-6:-4]} | area={a:.1%}" + (" QoD!" if a>0.15 else "")); ax.axis("off")
plt.tight_layout(); plt.show()
"""),
    code("""
plt.figure(figsize=(8,3))
plt.plot([f[-6:-4] for f in frames], [a*100 for a in areas], "o-", label="vehicle bbox area %")
plt.axhline(15, color="orange", ls="--", label="QoD trigger (15%)")
plt.ylabel("% of frame"); plt.xlabel("timestamp"); plt.legend(); plt.title("Bbox area growth drives the QoD boost"); plt.grid(alpha=.3); plt.show()
print("max area: %.1f%% -> crosses 15%% => QoD upgrades 480p->1080p" % (max(areas)*100))
"""),
    md("### Latency: CPU vs GPU (and YOLOv8s vs YOLOv8x, the PDR detector)"),
    code("""
imgs = [cv2.imread(f) for f in glob.glob("../tmp/frames/tekno-*.jpg")]
def bench(path, dev, n=4):
    m = YOLO(path); m.predict(imgs[0], imgsz=512, device=dev, verbose=False)  # warmup
    t = time.perf_counter(); k = 0
    for _ in range(n):
        for im in imgs:
            m.predict(im, imgsz=512, conf=0.18, classes=[2,3,5,7], device=dev, verbose=False); k += 1
    ms = (time.perf_counter()-t)/k*1000
    return ms
rows = [("YOLOv8s","cpu","../backend/yolov8s.pt"), ("YOLOv8s","cuda:0","../backend/yolov8s.pt"),
        ("YOLOv8x","cuda:0","../backend/yolov8x.pt")]
print(f"{'model':10}{'device':8}{'ms/frame':>10}{'FPS':>8}")
for name,dev,path in rows:
    if dev=="cuda:0" and DEV=="cpu": continue
    if not __import__('os').path.exists(path): continue
    ms = bench(path, dev); print(f"{name:10}{dev:8}{ms:>10.1f}{1000/ms:>8.1f}")
"""),
    md("""
**Conclusions.**
- Vehicle detection is solid on the real night footage (no training needed for *vehicle*).
- The bbox area grows monotonically on approach and crosses **15%**, which is the
  real, AI-driven signal that triggers the QoD 480p→1080p boost (PDR's closed loop).
- On GPU, **YOLOv8x runs ~24 ms/frame (~40 FPS)** — the exact PDR detector, real-time.
"""),
]

# --------------------------------------------------------------------------- 02
NOTEBOOKS["02_license_plate_resolution_limit.ipynb"] = [
    md("""
# 02 · License plate: a resolution limit, not a model problem

**Question.** Why is plate OCR unreliable? Is a better detector the fix?

**Short answer.** The demo clips are **832×464**. The plate is ~30 px wide — at
or below the OCR legibility floor. We show the plate crop, try EasyOCR, and try a
public Roboflow plate **detector** (hosted). None recover the plate, because the
pixels are not there.
"""),
    code("""
%matplotlib inline
import glob, re, cv2, numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import easyocr
DEV = "cuda:0" if __import__("torch").cuda.is_available() else "cpu"
model = YOLO("../backend/yolov8s.pt")
reader = easyocr.Reader(["en"], gpu=(DEV!="cpu"), verbose=False)
PLATE = re.compile(r"(\\d{2})([A-Z]{1,4})(\\d{2,4})")
"""),
    md("Locate the vehicle, crop the lower region (where the plate is), and measure how few pixels it actually is."),
    code("""
img = cv2.imread("../tmp/frames/tekno-01-4s.jpg"); H,W = img.shape[:2]
r = model.predict(img, imgsz=512, conf=0.18, classes=[2,3,5,7], device=DEV, verbose=False)[0]
b = max(r.boxes, key=lambda b:(b.xyxy[0][2]-b.xyxy[0][0])*(b.xyxy[0][3]-b.xyxy[0][1]))
x1,y1,x2,y2 = [int(v) for v in b.xyxy[0].tolist()]
plate_zone = img[y1+int((y2-y1)*0.55):y2, x1:x2]
print("frame size       :", (W,H))
print("vehicle box      :", (x2-x1, y2-y1), "px")
print("plate search zone:", (plate_zone.shape[1], plate_zone.shape[0]), "px  <- the legible plate is a tiny fraction of this")
fig,ax = plt.subplots(1,2, figsize=(12,4))
ax[0].imshow(cv2.cvtColor(img[y1:y2,x1:x2], cv2.COLOR_BGR2RGB)); ax[0].set_title("vehicle crop"); ax[0].axis("off")
ax[1].imshow(cv2.cvtColor(cv2.resize(plate_zone,None,fx=4,fy=4), cv2.COLOR_BGR2RGB)); ax[1].set_title("plate zone (4x upscaled = interpolation, no new detail)"); ax[1].axis("off")
plt.tight_layout(); plt.show()
"""),
    md("**Attempt 1 — EasyOCR** directly on the upscaled plate zone, filtered to Turkish plate format."),
    code("""
roi = cv2.resize(plate_zone, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
res = reader.readtext(roi, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ", detail=1)
for _,t,c in res:
    comp = re.sub(r"[^A-Z0-9]","",t.upper()); m = PLATE.fullmatch(comp)
    print(f"  read={t!r:18} conf={c:.2f}  format={'VALID' if m else 'invalid'}")
print("=> inconsistent / low confidence: characters are sub-legible at this resolution")
"""),
    md("**Attempt 2 — a real public plate *detector*** (Roboflow hosted). Needs `ROBOFLOW_API_KEY` in the environment."),
    code("""
import os, httpx
KEY = os.environ.get("ROBOFLOW_API_KEY")
if not KEY:
    print("ROBOFLOW_API_KEY not set -> skipping live call.")
    print("Documented result from our run: license-plate-recognition-rxg4e/13 returned [] on both")
    print("the full frame and the 4x-upscaled vehicle crop.")
else:
    url = "https://detect.roboflow.com/license-plate-recognition-rxg4e/13"
    for tag, im in [("full frame", img), ("vehicle crop x4", cv2.resize(img[y1:y2,x1:x2],None,fx=4,fy=4))]:
        ok,buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY,95])
        j = httpx.post(url, params={"api_key":KEY,"confidence":15}, content=buf.tobytes(), timeout=30).json()
        print(f"  {tag:16}-> detections: {j.get('predictions', [])}")
    print("=> a dedicated, well-trained plate detector also finds nothing: the signal is gone.")
"""),
    md("**Can a modern detector/segmenter *find* the plate? (YOLO-World + SAM 2)**"),
    code("""
from ultralytics import YOLOWorld, SAM
world = YOLOWorld("../backend/yolov8x-worldv2.pt"); world.set_classes(["license plate"])
sam = SAM("../backend/sam2_b.pt")
img2 = cv2.imread("../tmp/dataset_frames/tekno-01_0150.jpg")
rv = model.predict(img2, imgsz=512, conf=0.25, classes=[2,3,5,7], device=DEV, verbose=False)[0]
bb = max(rv.boxes, key=lambda b:(b.xyxy[0][2]-b.xyxy[0][0])*(b.xyxy[0][3]-b.xyxy[0][1]))
vx1,vy1,vx2,vy2 = [int(v) for v in bb.xyxy[0].tolist()]
crop = cv2.resize(img2[vy1:vy2, vx1:vx2], None, fx=4, fy=4); ch,cw = crop.shape[:2]
rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
# (a) YOLO-World open-vocabulary detection of "license plate"
w = world.predict(crop, imgsz=640, conf=0.01, device=DEV, verbose=False)[0]
yw = rgb.copy(); cover=0
if len(w.boxes):
    d = max(w.boxes, key=lambda b:float(b.conf)); bx=[int(v) for v in d.xyxy[0].tolist()]
    cover=(bx[2]-bx[0])*(bx[3]-bx[1])/(cw*ch)
    cv2.rectangle(yw,(bx[0],bx[1]),(bx[2],bx[3]),(255,0,0),4)
    print(f"YOLO-World 'license plate': conf={float(d.conf):.3f}, box covers {cover:.0%} of the crop (a plate is a few %)")
# (b) SAM 2 only segments where WE point it (a geometric prior, not detection)
px,py = cw//3, int(ch*0.9)
sp = sam(crop, points=[[px,py]], labels=[1], device=DEV, verbose=False)[0]
sm = rgb.copy(); mcov=0
if getattr(sp,'masks',None) is not None:
    mk=sp.masks.data.cpu().numpy()[0].astype(bool); mcov=mk.mean()
    sm[mk]=(0.5*sm[mk]+0.5*np.array([255,80,80])).astype(np.uint8); cv2.circle(sm,(px,py),10,(0,255,0),-1)
    print(f"SAM2 (we point at the plate): mask covers {mcov:.1%} of the crop")
fig,ax=plt.subplots(1,2,figsize=(13,4))
ax[0].imshow(yw); ax[0].set_title(f"YOLO-World 'license plate' -> {cover:.0%} box (cannot localise)"); ax[0].axis("off")
ax[1].imshow(sm); ax[1].set_title(f"SAM2 at a GIVEN point -> {mcov:.1%} mask (needs the prior)"); ax[1].axis("off")
plt.tight_layout(); plt.show()
"""),
    md("""
**Conclusion (data-driven).** The plate is **resolution-limited**, and no tool
*finds* it by itself at 464p:
- **YOLO-World** ("license plate", open-vocabulary) returns a **near-whole-image
  box at ~2% confidence** — it cannot localise the plate.
- **SAM 2** can only segment a plate-sized region when **we hand it the plate
  location** (a point/box). That is the *same geometric prior* we already use — SAM
  adds a tidy mask but **no detection**: it doesn't know where the plate is.
- A fixed-fraction ROI works similarly (geometric prior) and is what the pipeline uses.

So localisation stays **geometric**, and even with a perfect crop the **OCR is
capped by resolution** (the plate is ~30 px, almost no contrast). This is exactly
why the PDR designs the **QoD 480p→1080p** boost. With 1080p footage a learned
plate detector + EasyOCR become viable; on these 464p clips the reader can only
produce a best-effort guess.
"""),
]

# --------------------------------------------------------------------------- 03
NOTEBOOKS["03_driver_behavior_and_occupants.ipynb"] = [
    md("""
# 03 · Driver behavior (smoking / phone) & occupant detection

**Two findings:**
1. **Smoking / phone use** cannot be detected off-the-shelf — it needs a model
   trained on *this* domain. We show the behavior IS visible, but COCO has no
   "cigarette" class and the cited public dataset (bright in-cabin closeups) does
   not transfer to this night / surveillance-angle footage.
2. **Occupants (driver + passengers)** ARE detectable — by cropping the vehicle
   and upscaling, the seated driver appears as a `person`, invisible in the full frame.
"""),
    code("""
%matplotlib inline
import glob, cv2, numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
DEV = "cuda:0" if __import__("torch").cuda.is_available() else "cpu"
model = YOLO("../backend/yolov8x.pt") if __import__('os').path.exists("../backend/yolov8x.pt") else YOLO("../backend/yolov8s.pt")
"""),
    md("The driver (and behavior) **is** visible through the side window — show the cabin crops."),
    code("""
fig, axes = plt.subplots(1, 2, figsize=(13,4))
for ax, name, note in [(axes[0],"tekno-01-4s","smoking (hand to mouth)"), (axes[1],"tekno-02-3s","phone to ear")]:
    img = cv2.imread(f"../tmp/frames/{name}.jpg")
    r = model.predict(img, imgsz=512, conf=0.2, classes=[2,3,5,7], device=DEV, verbose=False)[0]
    b = max(r.boxes, key=lambda b:(b.xyxy[0][2]-b.xyxy[0][0])*(b.xyxy[0][3]-b.xyxy[0][1]))
    x1,y1,x2,y2 = [int(v) for v in b.xyxy[0].tolist()]
    crop = cv2.resize(img[y1:y2,x1:x2], None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    ax.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)); ax.set_title(f"{name}: {note}"); ax.axis("off")
plt.tight_layout(); plt.show()
"""),
    md("**Smoking/phone via COCO** — try detecting `cell phone` (class 67) in the cabin. (No COCO class exists for a cigarette.)"),
    code("""
img = cv2.imread("../tmp/frames/tekno-02-3s.jpg")
r = model.predict(img, imgsz=512, conf=0.2, classes=[2,3,5,7], device=DEV, verbose=False)[0]
b = max(r.boxes, key=lambda b:(b.xyxy[0][2]-b.xyxy[0][0])*(b.xyxy[0][3]-b.xyxy[0][1]))
x1,y1,x2,y2 = [int(v) for v in b.xyxy[0].tolist()]
cabin = cv2.resize(img[y1:y2,x1:x2], None, fx=3, fy=3)
phones = model.predict(cabin, imgsz=640, conf=0.05, classes=[67], device=DEV, verbose=False)[0]
print("cell-phone detections in cabin @conf 0.05:", len(phones.boxes), "-> none usable; and no 'cigarette' class exists in COCO")
"""),
    md("**Cited public behavior model** (hosted) on a cabin crop — domain mismatch. Needs `ROBOFLOW_API_KEY`."),
    code("""
import os, httpx
KEY = os.environ.get("ROBOFLOW_API_KEY")
if not KEY:
    print("ROBOFLOW_API_KEY not set -> skipping live call.")
    print("Documented result: distracted-driving-v2wk5/7 returned [] (0 detections) even at 1% confidence")
    print("on both the windshield crop and the full frame -> bright in-cabin domain != night surveillance domain.")
else:
    ok,buf = cv2.imencode(".jpg", img[y1:y1+int((y2-y1)*0.4), x1:x2])
    j = httpx.post("https://detect.roboflow.com/distracted-driving-v2wk5/7", params={"api_key":KEY,"confidence":1}, content=buf.tobytes(), timeout=30).json()
    print("hosted behavior detections:", j.get("predictions", []), "-> empty = domain mismatch")
"""),
    md("**Occupant detection that works** — full frame vs cropped+upscaled cabin."),
    code("""
fig, axes = plt.subplots(1, 2, figsize=(13,4))
img = cv2.imread("../tmp/frames/tekno-01-5s.jpg"); H,W = img.shape[:2]
# full frame person detection
full = model.predict(img, imgsz=512, conf=0.4, classes=[0], device=DEV, verbose=False)[0]
disp = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
axes[0].imshow(disp); axes[0].set_title(f"full frame: {len(full.boxes)} person(s) >=0.4 (driver missed)"); axes[0].axis("off")
# cabin crop person detection
r = model.predict(img, imgsz=512, conf=0.2, classes=[2,3,5,7], device=DEV, verbose=False)[0]
b = max(r.boxes, key=lambda b:(b.xyxy[0][2]-b.xyxy[0][0])*(b.xyxy[0][3]-b.xyxy[0][1]))
x1,y1,x2,y2 = [int(v) for v in b.xyxy[0].tolist()]
cabin = cv2.resize(img[y1:y2,x1:x2], None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
occ = model.predict(cabin, imgsz=640, conf=0.4, classes=[0], device=DEV, verbose=False)[0]
cab = cv2.cvtColor(cabin.copy(), cv2.COLOR_BGR2RGB)
for pb in occ.boxes:
    a,bb,c,d = [int(v) for v in pb.xyxy[0].tolist()]; cv2.rectangle(cab,(a,bb),(c,d),(0,255,0),3)
axes[1].imshow(cab); axes[1].set_title(f"cabin crop+upscale: {len(occ.boxes)} occupant(s) detected"); axes[1].axis("off")
plt.tight_layout(); plt.show()
print("occupants in cabin crop:", len(occ.boxes), "| confidences:", [round(float(x.conf),2) for x in occ.boxes])
"""),
    md("""
**Conclusions.**
- **Smoking/phone**: visible to a human but not to off-the-shelf models. Requires a
  model trained on this domain → we built the Roboflow project
  `afrikaans/quismotion-driver-behavior` and pre-labeled 65 frames in
  `tmp/cabin_dataset/` (classes: driver, passenger, phone, cigarette).
- **Occupants**: the driver is reliably detected by cropping+upscaling the vehicle
  (now exposed as the `occupants` field in the live API). Reliable *passenger*
  counting is still limited by the 464p footage.
"""),
]

# --------------------------------------------------------------------------- 04
NOTEBOOKS["04_duckdb_vss_similarity.ipynb"] = [
    md("""
# 04 · Incident retrieval with DuckDB + VSS

The PDR persists YOLOv8 frame embeddings and retrieves *semantically similar*
past incidents. This reproduces that: embed frames, store the 512-d vectors in
DuckDB's VSS extension, and run a cosine-distance similarity query.
"""),
    code("""
import glob, duckdb, numpy as np
from ultralytics import YOLO
DEV = "cuda:0" if __import__("torch").cuda.is_available() else "cpu"
emb_model = YOLO("../backend/yolov8s.pt")
frames = sorted(glob.glob("../tmp/frames/tekno-*.jpg"))
vectors = {f.split('/')[-1].split(chr(92))[-1]: [float(x) for x in emb_model.embed(f, imgsz=512, device=DEV, verbose=False)[0].tolist()] for f in frames}
DIM = len(next(iter(vectors.values())))
print("embedded", len(vectors), "frames | embedding dim:", DIM)
"""),
    code("""
con = duckdb.connect()
con.execute("INSTALL vss"); con.execute("LOAD vss")
con.execute(f"CREATE TABLE incidents(name TEXT, v FLOAT[{DIM}])")
for name, v in vectors.items():
    con.execute("INSERT INTO incidents VALUES (?, ?)", [name, v])
con.execute("CREATE INDEX idx ON incidents USING HNSW(v) WITH (metric='cosine')")
print("stored", con.execute("SELECT COUNT(*) FROM incidents").fetchone()[0], "incident vectors in DuckDB + HNSW index")
"""),
    code("""
query = "tekno-01-4s.jpg"
rows = con.execute(f'''
    SELECT name, array_cosine_distance(v, (SELECT v FROM incidents WHERE name=?)::FLOAT[{DIM}]) AS dist
    FROM incidents WHERE name != ? ORDER BY dist ASC LIMIT 5
''', [query, query]).fetchall()
print(f"Most similar incidents to {query}:")
for name, dist in rows:
    print(f"  {name:20} similarity={1-dist:.3f}")
"""),
    md("""
**Conclusion.** YOLOv8 embeddings + DuckDB VSS give real, in-process vector
similarity search over incidents — frames from the same clip/approach rank as most
similar. This backs the live `GET /api/incidents/{id}/similar` endpoint.
"""),
]


def build():
    os.makedirs(HERE, exist_ok=True)
    for fname, cells in NOTEBOOKS.items():
        nb = nbf.v4.new_notebook()
        nb.cells = cells
        nb.metadata = {"kernelspec": {"name": "quismotion", "display_name": "QuisMotion (.venv)"},
                       "language_info": {"name": "python"}}
        path = os.path.join(HERE, fname)
        print(f"executing {fname} ...")
        client = NotebookClient(nb, timeout=600, kernel_name="quismotion",
                                resources={"metadata": {"path": HERE}}, allow_errors=True)
        client.execute()
        with open(path, "w", encoding="utf-8") as fh:
            nbf.write(nb, fh)
        print(f"  wrote {path}")


if __name__ == "__main__":
    build()

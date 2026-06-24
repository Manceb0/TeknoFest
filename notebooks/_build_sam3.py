"""Builds & executes notebook 10 - SAM 3 promptable concept segmentation
(complementary analysis). Runs automatically once backend/sam3.pt is present;
otherwise prints the exact steps to obtain the gated checkpoint."""
import os, nbformat as nbf
from nbclient import NotebookClient

HERE = os.path.dirname(os.path.abspath(__file__))
md = lambda t: nbf.v4.new_markdown_cell(t.strip("\n"))
code = lambda t: nbf.v4.new_code_cell(t.strip("\n"))

CELLS = [
    md("""
# 10 · SAM 3 — Promptable Concept Segmentation (complementary)

This **complements** (does not replace) notebooks 00–09. SAM 3 (Meta, Nov 2025) is
a single model that, from a **text prompt**, *detects + segments + tracks* every
instance of a concept — e.g. `cigarette`, `mobile phone`, `person`. That is exactly
what robustifies the smoking-vs-phone case: it returns the right concept **and**
pixel masks in one shot (vs. the 13-class model that mislabelled smoking as phone).

**Status in this repo.** SAM 3 is integrated in Ultralytics ≥ 8.3.237 (here
8.4.75 ✓). Its checkpoint `sam3.pt` is **gated by Meta** and is **not
auto-downloaded** — it must be obtained manually (see steps below). This notebook
runs SAM 3 automatically once `backend/sam3.pt` exists; otherwise it shows the
working equivalent from notebook 09 (YOLO-World + SAM2).
"""),
    code("""
%matplotlib inline
import os, cv2, numpy as np
import matplotlib.pyplot as plt
DEV = "cuda:0" if __import__("torch").cuda.is_available() else "cpu"
SAM3_PATH = "../backend/sam3.pt"
CONCEPTS = ["cigarette", "mobile phone", "person"]
print("SAM3 checkpoint present:", os.path.exists(SAM3_PATH))
"""),
    md("### Run SAM 3 concept segmentation (auto-activates when `sam3.pt` is present)"),
    code("""
import cv2, numpy as np, matplotlib.pyplot as plt
frames = ["tekno-01-4s", "tekno-02-4s"]   # smoking, phone
if os.path.exists(SAM3_PATH):
    from ultralytics import SAM, YOLO
    veh = YOLO("../backend/yolov8x.pt")
    sam3 = SAM(SAM3_PATH)
    fig, axes = plt.subplots(1, len(frames), figsize=(13, 5))
    for ax, name in zip(np.atleast_1d(axes), frames):
        img = cv2.imread(f"../tmp/frames/{name}.jpg")
        r = veh.predict(img, imgsz=512, conf=0.25, classes=[2,3,5,7], device=DEV, verbose=False)[0]
        b = max(r.boxes, key=lambda b:(b.xyxy[0][2]-b.xyxy[0][0])*(b.xyxy[0][3]-b.xyxy[0][1]))
        x1,y1,x2,y2 = [int(v) for v in b.xyxy[0].tolist()]
        crop = cv2.cvtColor(cv2.resize(img[y1:y2,x1:x2], None, fx=4, fy=4), cv2.COLOR_BGR2RGB)
        overlay = crop.copy()
        # SAM 3 promptable concept segmentation (text prompts)
        for ci, concept in enumerate(CONCEPTS):
            res = sam3(cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), prompt=concept, device=DEV, verbose=False)[0]
            if getattr(res, "masks", None) is not None:
                col = [(255,60,60),(60,160,255),(60,220,120)][ci]
                for m in res.masks.data.cpu().numpy():
                    overlay[m.astype(bool)] = (0.5*overlay[m.astype(bool)] + 0.5*np.array(col)).astype(np.uint8)
        ax.imshow(overlay); ax.set_title(f"{name} — SAM3 concepts: {', '.join(CONCEPTS)}", fontsize=9); ax.axis("off")
    plt.tight_layout(); plt.show()
    print("SAM 3 concept segmentation rendered.")
else:
    print("sam3.pt not found -> SAM 3 cell skipped.")
    print("The equivalent capability (concept detection + masks) is shown NOW in")
    print("notebook 09 (YOLO-World + SAM2). Add sam3.pt and re-run to use SAM 3 directly.")
"""),
    md("""
## How to obtain `sam3.pt` (gated — your action)

Meta gates the SAM 3 weights, so they cannot be fetched automatically. One-time:

1. Request access on the model card: https://huggingface.co/facebook/sam3
2. Authenticate: `huggingface-cli login` (paste your HF token).
3. Download and place it as `backend/sam3.pt`, e.g.:
   ```python
   from huggingface_hub import hf_hub_download
   hf_hub_download("facebook/sam3", "sam3.pt", local_dir="backend")
   ```
4. Re-run this notebook — the SAM 3 cell activates automatically.

## Why SAM 3 here (and the honest caveats)
- **Fixes semantics:** text prompts return `cigarette` vs `mobile phone` directly —
  no custom class needed — plus pixel masks (better than boxes).
- **Same resolution limit:** at 832×464 night footage, confidence on small objects
  stays low (as seen with YOLO-World in nb 09); SAM 3 helps most on higher-res frames.
- **Not real-time / edge:** SAM 3 is heavy; best used **off the live loop** — as an
  auto-labeller to build the domain dataset, then fine-tune the fast YOLOv8 for the
  live page. Notebooks 00–09 remain the primary analysis; this is complementary.

**References.**
- SAM 3 — Ultralytics docs: https://docs.ultralytics.com/models/sam-3
- SAM 3 — Meta AI: https://ai.meta.com/research/sam3/  ·  facebookresearch/sam3 (GitHub)
"""),
]


def build():
    nb = nbf.v4.new_notebook(); nb.cells = CELLS
    nb.metadata = {"kernelspec": {"name": "quismotion", "display_name": "QuisMotion (.venv)"},
                   "language_info": {"name": "python"}}
    NotebookClient(nb, timeout=600, kernel_name="quismotion",
                   resources={"metadata": {"path": HERE}}, allow_errors=True).execute()
    with open(os.path.join(HERE, "10_sam3_concept_segmentation.ipynb"), "w", encoding="utf-8") as fh:
        nbf.write(nb, fh)
    print("wrote 10_sam3_concept_segmentation.ipynb")


if __name__ == "__main__":
    build()

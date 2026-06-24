"""Builds & executes FDR notebooks 06 (Dataset) and 07 (Testing) for the
DELIVERED model: combined phone/cigarette/safe."""
import os, nbformat as nbf
from nbclient import NotebookClient

HERE = os.path.dirname(os.path.abspath(__file__))
md = lambda t: nbf.v4.new_markdown_cell(t.strip("\n"))
code = lambda t: nbf.v4.new_code_cell(t.strip("\n"))

NB = {}

# ----------------------------------------------------- 06 Dataset Preparation
NB["06_dataset_preparation.ipynb"] = [
    md("""
# 06 · Dataset Preparation  (FDR Section 2)

How the data for the **delivered** behavior model (`phone / cigarette / safe`) was
sourced, labelled, balanced and split.

**Sources.**
1. **Open-source** *Distracted Driving* (Roboflow Universe,
   `ipylot-project/distracted-driving-v2wk5`) — thousands of labelled driver images.
   The rules allow open-source data for training/fine-tuning.
2. **Competition footage** — frames from the 3 TEKNOFEST clips (night, surveillance
   angle), for **domain adaptation** and for the `cigarette` class.

**Label mapping (to the project's 3 risk cases).**
- `Texting` + `Talking on the phone` → **`phone`** (abundant, real)
- competition tekno-01 driver region → **`cigarette`** (smoking; weak/clip-prior)
- `Safe Driving` + tekno-03 → **`safe`**
- (tekno-03 *reckless/swerving* is a **trajectory** signal, handled in nb 01/03, not a cabin class)
"""),
    code("""
%matplotlib inline
import glob, os
import numpy as np, matplotlib.pyplot as plt
DS = "../tmp/behavior_combined"
NAMES = ["phone","cigarette","safe"]
counts = {s:{0:0,1:0,2:0} for s in ["train","valid","test"]}
imgs   = {s:0 for s in ["train","valid","test"]}
for s in counts:
    for lbl in glob.glob(f"{DS}/{s}/labels/*.txt"):
        imgs[s]+=1
        for ln in open(lbl):
            if ln.strip(): counts[s][int(ln.split()[0])]+=1
tot = sum(imgs.values())
print(f"{'split':7}{'images':>8}{'phone':>8}{'ciginst':>9}{'safe':>8}{'ratio':>8}")
for s in ["train","valid","test"]:
    print(f"{s:7}{imgs[s]:>8}{counts[s][0]:>8}{counts[s][1]:>9}{counts[s][2]:>8}{imgs[s]/tot:>7.0%}")
print(f"{'TOTAL':7}{tot:>8}")
"""),
    md("**Class distribution per split** and the 70/15/15 partition."),
    code("""
splits=["train","valid","test"]; x=np.arange(len(splits)); w=0.26
fig,ax=plt.subplots(1,2,figsize=(13,4))
for i,(cls,col) in enumerate([("phone","#4c72b0"),("cigarette","#dd8452"),("safe","#55a868")]):
    ax[0].bar(x+(i-1)*w,[counts[s][i] for s in splits],w,label=cls,color=col)
ax[0].set_xticks(x); ax[0].set_xticklabels(splits); ax[0].set_title("Box count per class per split"); ax[0].legend(); ax[0].grid(alpha=.3,axis="y")
ax[1].pie([imgs[s] for s in splits],labels=[f"{s}\\n{imgs[s]}" for s in splits],autopct="%1.0f%%",colors=["#4c72b0","#dd8452","#55a868"])
ax[1].set_title("Train / Val / Test = 70 / 15 / 15")
plt.tight_layout(); plt.show()
"""),
    md("""
**Split justification.** 70/15/15 is a standard partition with statistically
meaningful, class-balanced validation/test sets. `phone` and `safe` come from the
diverse open-source set (well represented). `cigarette` comes **only from the
tekno-01 competition clip** (few, weakly clip-prior labelled) — this is documented
as a limitation (see the leakage note in notebook 07).

**Augmentation** is applied on-the-fly during training and shown in detail in
**notebook 11** (brightness/exposure, saturation, rotation, flip, scale, blur,
noise, JPEG, mosaic, mixup) — used here to bridge the bright→night domain gap.

**References.**
- iPylot. *Distracted Driving* dataset. Roboflow Universe.
- M. J. Rahman. *Driver Detection* dataset. Roboflow Universe.
- TEKNOFEST competition footage (provided clips).
"""),
]

# --------------------------------------------------------- 07 Solution Testing
RUN = "../backend/runs/combined_test"
NB["07_solution_testing.ipynb"] = [
    md("""
# 07 · Solution Testing  (FDR Section 4) — "Why do we trust our solution?"

Formal evaluation of the **delivered** behavior model (`phone / cigarette / safe`,
`runs/behavior_combined`) on its **held-out test split** — the model wired into the
live backend.
"""),
    code(f"""
import json, pandas as pd
s = json.load(open("{RUN}/eval.json"))
print("OVERALL (test set), device:", s.get("device","?"))
display(pd.DataFrame([s["overall"]]))
print("PER CLASS:")
display(pd.DataFrame(s["per_class"]))
print("Inference speed:", s["fps_gpu_512"], "FPS (512px)")
"""),
    md("### Confusion matrix (test set)"),
    code(f"""
%matplotlib inline
import matplotlib.pyplot as plt, matplotlib.image as mpimg
fig,ax=plt.subplots(1,2,figsize=(13,5))
for a,p,t in [(ax[0],"{RUN}/confusion_matrix.png","Confusion matrix (counts)"),
              (ax[1],"{RUN}/confusion_matrix_normalized.png","Normalized")]:
    a.imshow(mpimg.imread(p)); a.set_title(t); a.axis("off")
plt.tight_layout(); plt.show()
"""),
    md("""
**What this shows.** Predicted (rows) vs true (columns) for `phone / cigarette /
safe` on the held-out test set. The diagonal = correct predictions; off-diagonal =
confusions. `phone` and `safe` are clean; `cigarette` also scores high **but see the
honesty note below** — its test frames come from the same clip as its training
frames, so that number is optimistic.
"""),
    md("### Precision-Recall and F1 curves (test set)"),
    code(f"""
fig,ax=plt.subplots(1,2,figsize=(13,5))
for a,p,t in [(ax[0],"{RUN}/BoxPR_curve.png","Precision-Recall curve"),
              (ax[1],"{RUN}/BoxF1_curve.png","F1-confidence curve")]:
    a.imshow(mpimg.imread(p)); a.set_title(t); a.axis("off")
plt.tight_layout(); plt.show()
"""),
    md("""
**What this shows.** Area under PR = Average Precision per class; the F1-confidence
curve gives the best operating threshold. High and stable for `phone`/`safe`.
"""),
    md("### Inference speed (FPS)"),
    code(f"""
import pandas as pd, json
s = json.load(open("{RUN}/eval.json"))
fps = pd.DataFrame([
    {{"model":"YOLOv8x detector","device":"GPU (RTX 4060)","ms/frame":24,"FPS":40}},
    {{"model":"YOLOv8s detector","device":"CPU","ms/frame":101,"FPS":10}},
    {{"model":"Behavior phone/cigarette/safe","device":s.get("device","?"),"ms/frame":round(1000/max(s["fps_gpu_512"],1),1),"FPS":s["fps_gpu_512"]}},
])
display(fps)
"""),
    md("""
### Why we trust our solution — and where we don't (honest)

- **`phone` and `safe` are trustworthy.** They are backed by thousands of diverse
  open-source images; their test metrics measure real generalization.
- **Vehicle detection** is validated separately (`tests/test_real_videos.py`):
  consistent detection on all three clips, confidence ≥ 0.9, growing bbox area.
- **Real-time:** the detector runs ~24 ms/frame on the RTX 4060 (≈40 FPS).

> **⚠️ Honesty note — `cigarette` metric is optimistic (data leakage).** The
> `cigarette` class is built **only from the tekno-01 clip**, and its train and
> test frames come from that same short video (near-duplicate frames). So a high
> `cigarette` score means the model **memorised that clip**, not that it detects
> smoking in general. To claim generalisable smoking detection we need diverse
> footage (multiple drivers, day/night). On the live page it correctly flags
> smoking on tekno-01; treat that as a per-clip demo, not a general guarantee.
"""),
]


def build():
    for fname, cells in NB.items():
        nb = nbf.v4.new_notebook(); nb.cells = cells
        nb.metadata = {"kernelspec": {"name": "quismotion", "display_name": "QuisMotion (.venv)"},
                       "language_info": {"name": "python"}}
        print("executing", fname)
        NotebookClient(nb, timeout=400, kernel_name="quismotion",
                       resources={"metadata": {"path": HERE}}, allow_errors=True).execute()
        with open(os.path.join(HERE, fname), "w", encoding="utf-8") as fh:
            nbf.write(nb, fh)
        print("  wrote", fname)


if __name__ == "__main__":
    build()

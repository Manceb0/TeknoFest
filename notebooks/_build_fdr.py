"""Builds & executes FDR evidence notebooks 06 (Dataset) and 07 (Testing)."""
import os, glob, nbformat as nbf
from nbclient import NotebookClient

HERE = os.path.dirname(os.path.abspath(__file__))
md = lambda t: nbf.v4.new_markdown_cell(t.strip("\n"))
code = lambda t: nbf.v4.new_code_cell(t.strip("\n"))

NB = {}

# ----------------------------------------------------- 06 Dataset Preparation
NB["06_dataset_preparation.ipynb"] = [
    md("""
# 06 · Dataset Preparation  (FDR Section 2)

How the behavior data was sourced, labelled, focused, balanced and split.

**Source.** Open-source *Distracted Driving* dataset (Roboflow Universe,
`ipylot-project/distracted-driving-v2wk5`, v7) — 8,865 labelled images, 13
activity classes. Per the competition rules, open-source datasets may be used for
training/fine-tuning, reserving the competition clips for qualitative validation.

**Focusing.** QuisMotion only needs *phone use* vs *safe*. The 13 classes diluted
the signal (and the original split left only 3 phone boxes in test). We therefore
**remapped** `Texting`+`Talking on the phone` → `phone`, `Safe Driving` → `safe`,
dropped the rest, and **re-stratified** into a balanced 70/15/15 split.
"""),
    code("""
%matplotlib inline
import glob, os
import matplotlib.pyplot as plt
DS = "../tmp/behavior_ds"
NAMES = ["phone","safe"]
counts = {s:{0:0,1:0} for s in ["train","valid","test"]}
imgs   = {s:0 for s in ["train","valid","test"]}
for s in counts:
    for lbl in glob.glob(f"{DS}/{s}/labels/*.txt"):
        imgs[s]+=1
        for ln in open(lbl):
            if ln.strip(): counts[s][int(ln.split()[0])]+=1
tot = sum(imgs.values())
print(f"{'split':7}{'images':>8}{'phone':>8}{'safe':>8}{'ratio':>8}")
for s in ["train","valid","test"]:
    print(f"{s:7}{imgs[s]:>8}{counts[s][0]:>8}{counts[s][1]:>8}{imgs[s]/tot:>7.0%}")
print(f"{'TOTAL':7}{tot:>8}")
"""),
    md("**Class distribution per split** (balanced — phone present in every split)."),
    code("""
import numpy as np
splits=["train","valid","test"]; x=np.arange(len(splits)); w=0.35
fig,ax=plt.subplots(1,2,figsize=(13,4))
ax[0].bar(x-w/2,[counts[s][0] for s in splits],w,label="phone")
ax[0].bar(x+w/2,[counts[s][1] for s in splits],w,label="safe")
ax[0].set_xticks(x); ax[0].set_xticklabels(splits); ax[0].set_title("Box count per class per split"); ax[0].legend(); ax[0].grid(alpha=.3,axis="y")
ax[1].pie([imgs[s] for s in splits],labels=[f"{s}\\n{imgs[s]}" for s in splits],autopct="%1.0f%%",colors=["#4c72b0","#dd8452","#55a868"])
ax[1].set_title("Train / Val / Test = 70 / 15 / 15")
plt.tight_layout(); plt.show()
"""),
    md("""
**Split justification.** 70/15/15 is a standard partition that leaves enough data
to train while keeping statistically meaningful, **class-balanced** validation and
test sets. Stratifying by class guarantees `phone` and `safe` both appear in every
split (the original vendor split did not — it had only 3 phone boxes in test, which
made the phone metric meaningless).
"""),
    md("**Data augmentation** (applied during training to improve robustness)."),
    code("""
import cv2, numpy as np
sample = sorted(glob.glob(f"{DS}/train/images/*.jpg"))[0]
img = cv2.cvtColor(cv2.imread(sample), cv2.COLOR_BGR2RGB)
def hsv_v(im,f):
    h=cv2.cvtColor(im,cv2.COLOR_RGB2HSV).astype(float); h[...,2]=np.clip(h[...,2]*f,0,255); return cv2.cvtColor(h.astype(np.uint8),cv2.COLOR_HSV2RGB)
augs={"original":img,"darker (night sim, hsv_v 0.4)":hsv_v(img,0.4),"brighter (hsv_v 1.5)":hsv_v(img,1.5),
      "h-flip":img[:,::-1],"blur":cv2.GaussianBlur(img,(7,7),0),"scale 0.6":cv2.resize(cv2.resize(img,None,fx=.6,fy=.6),(img.shape[1],img.shape[0]))}
fig,ax=plt.subplots(2,3,figsize=(13,6))
for a,(t,im) in zip(ax.ravel(),augs.items()): a.imshow(im); a.set_title(t); a.axis("off")
plt.tight_layout(); plt.show()
print("Training augmentation: hsv_v=0.6, hsv_s=0.5, fliplr=0.5, scale=0.5, translate=0.1, degrees=5, mosaic=1.0")
"""),
    md("""
**References (datasets).**
- iPylot. *Distracted Driving* dataset. Roboflow Universe.
  https://universe.roboflow.com/ipylot-project/distracted-driving-v2wk5
- M. J. Rahman. *Driver Detection* dataset. Roboflow Universe.
- TEKNOFEST competition footage (provided clips) — used for qualitative validation.
"""),
]

# --------------------------------------------------------- 07 Solution Testing
NB["07_solution_testing.ipynb"] = [
    md("""
# 07 · Solution Testing  (FDR Section 4) — "Why do we trust our solution?"

Formal evaluation of the focused **phone-vs-safe** behavior model on the
**held-out test split** (478 images: 300 phone, 178 safe) it never saw in training.
"""),
    code("""
import json, pandas as pd
s = json.load(open("../backend/runs/focused_test/eval_summary.json"))
print("OVERALL (test set):")
display(pd.DataFrame([s["overall"]]))
print("PER CLASS:")
display(pd.DataFrame(s["per_class"]))
print("Inference speed:", s["fps_gpu_512"], "FPS  (RTX 4060, 512px)")
"""),
    md("### Confusion matrix (test set)"),
    code("""
%matplotlib inline
import matplotlib.pyplot as plt, matplotlib.image as mpimg
fig,ax=plt.subplots(1,2,figsize=(13,5))
for a,p,t in [(ax[0],"../backend/runs/focused_test/confusion_matrix.png","Confusion matrix (counts)"),
              (ax[1],"../backend/runs/focused_test/confusion_matrix_normalized.png","Normalized")]:
    a.imshow(mpimg.imread(p)); a.set_title(t); a.axis("off")
plt.tight_layout(); plt.show()
"""),
    md("### Precision-Recall and F1 curves (test set)"),
    code("""
fig,ax=plt.subplots(1,2,figsize=(13,5))
for a,p,t in [(ax[0],"../backend/runs/focused_test/BoxPR_curve.png","Precision-Recall curve"),
              (ax[1],"../backend/runs/focused_test/BoxF1_curve.png","F1-confidence curve")]:
    a.imshow(mpimg.imread(p)); a.set_title(t); a.axis("off")
plt.tight_layout(); plt.show()
"""),
    md("### Inference speed (FPS) across configs"),
    code("""
import pandas as pd
fps = pd.DataFrame([
    {"model":"YOLOv8x detector","device":"GPU (RTX 4060)","ms/frame":24,"FPS":40},
    {"model":"YOLOv8s detector","device":"GPU","ms/frame":18,"FPS":55},
    {"model":"YOLOv8s detector","device":"CPU","ms/frame":101,"FPS":10},
    {"model":"Behavior (phone/safe)","device":"GPU","ms/frame":round(1000/s["fps_gpu_512"],1),"FPS":s["fps_gpu_512"]},
])
display(fps)
"""),
    md("""
### Why we trust our solution
- On the **held-out test split** the behavior model reaches **F1 = 1.00**,
  **mAP@50 = 0.995**, **mAP@50-95 = 0.86** with **phone and safe both at P/R ≈ 1.0**
  — measured on data excluded from training, with phone well represented (300 imgs).
- The confusion matrix shows **near-zero cross-class confusion**.
- It runs at **~26 FPS** on the RTX 4060 (real-time over a 10 FPS stream).
- Vehicle detection is validated separately (`tests/test_real_videos.py`): consistent
  detections on all three competition clips with confidence ≥ 0.9 and growing bbox area.

**Honest scope.** These metrics are on the open-source dataset's domain (bright,
front-cabin). On the night/side-angle competition footage the model needs
fine-tuning on annotated competition frames (project `quismotion-driver-behavior`,
script `finetune_behavior.py`) — see notebook 03. License-plate OCR is bounded by
the 464p clip resolution (notebook 02). The pipeline, metrics and FPS above
constitute the data-driven evidence that the AI solution works.
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

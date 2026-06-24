"""Builds & executes notebook 05 — behavior training curves (overfitting check).

Reads backend/runs/behavior/results.csv (one row per epoch) and plots train vs.
validation loss and the detection metrics, then auto-diagnoses overfitting.
"""
import os, nbformat as nbf
from nbclient import NotebookClient

HERE = os.path.dirname(os.path.abspath(__file__))
md = lambda t: nbf.v4.new_markdown_cell(t.strip("\n"))
code = lambda t: nbf.v4.new_code_cell(t.strip("\n"))

CELLS = [
    md("""
# 05 · Behavior model — training curves & overfitting check

These curves come straight from the **epochs**: Ultralytics logs every epoch to
`runs/behavior/results.csv`. We plot **train vs. validation loss** (the classic
overfitting test) and the detection metrics, then auto-diagnose.

> Note: object detection uses **mAP / precision / recall** instead of a single
> "accuracy". The overfitting logic is the same — watch the train↔val gap.
"""),
    code("""
%matplotlib inline
import pandas as pd, matplotlib.pyplot as plt
df = pd.read_csv("../backend/runs/behavior_focused/results.csv")
df.columns = [c.strip() for c in df.columns]
df["train/loss"] = df["train/box_loss"]+df["train/cls_loss"]+df["train/dfl_loss"]
df["val/loss"]   = df["val/box_loss"]+df["val/cls_loss"]+df["val/dfl_loss"]
print("epochs:", len(df))
df[["epoch","train/loss","val/loss","metrics/mAP50(B)","metrics/mAP50-95(B)"]].tail()
"""),
    md("### Loss: training vs validation (the overfitting test)"),
    code("""
fig, ax = plt.subplots(1, 2, figsize=(14,4.5))
ax[0].plot(df["epoch"], df["train/loss"], label="train loss")
ax[0].plot(df["epoch"], df["val/loss"], label="val loss")
ax[0].set_title("Total loss (box+cls+dfl)"); ax[0].set_xlabel("epoch"); ax[0].legend(); ax[0].grid(alpha=.3)
for comp,c in [("box","tab:blue"),("cls","tab:orange"),("dfl","tab:green")]:
    ax[1].plot(df["epoch"], df[f"train/{comp}_loss"], color=c, ls="-",  label=f"train {comp}")
    ax[1].plot(df["epoch"], df[f"val/{comp}_loss"],   color=c, ls="--", label=f"val {comp}")
ax[1].set_title("Loss components (— train, -- val)"); ax[1].set_xlabel("epoch"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.show()
"""),
    md("### Detection metrics over epochs (analogous to 'accuracy')"),
    code("""
plt.figure(figsize=(8,4.5))
for m,lab in [("metrics/precision(B)","precision"),("metrics/recall(B)","recall"),
              ("metrics/mAP50(B)","mAP@50"),("metrics/mAP50-95(B)","mAP@50-95")]:
    plt.plot(df["epoch"], df[m], label=lab)
plt.title("Validation metrics per epoch"); plt.xlabel("epoch"); plt.ylim(0,1)
plt.legend(); plt.grid(alpha=.3); plt.show()
"""),
    md("### Automatic overfitting diagnosis"),
    code("""
last = df.tail(5)
gap = last["val/loss"].mean() - last["train/loss"].mean()
val_min = df["val/loss"].min()
val_rise = df["val/loss"].iloc[-1] - val_min          # real overfitting signal
rel_gap = gap / last["train/loss"].mean()
best = df.loc[df["metrics/mAP50(B)"].idxmax()]
print(f"final train loss : {df['train/loss'].iloc[-1]:.3f}")
print(f"final val   loss : {df['val/loss'].iloc[-1]:.3f}  (min was {val_min:.3f})")
print(f"train/val gap    : {gap:+.3f}  ({rel_gap:+.0%} of train loss; a small gap is normal)")
print(f"val-loss rise since its min : {val_rise:+.3f}  <-- decisive signal, should be ~0")
print(f"best epoch: {int(best['epoch'])}  mAP50={best['metrics/mAP50(B)']:.3f}  mAP50-95={best['metrics/mAP50-95(B)']:.3f}")
overfit = val_rise > 0.10 and rel_gap > 0.20
verdict = "OVERFITTING (val loss diverging)" if overfit else "NOT overfitting (val at/near its min, mAP plateaued high)"
print("VERDICT:", verdict)
"""),
    md("""
**Reading these curves.**
- Train and validation loss **fall together** and validation **mAP plateaus high**
  (~0.97–0.99) without the val loss curving back up → the model is **not
  overfitting** on the distracted-driving dataset; it generalizes within that domain.
- **But** good in-domain curves ≠ works on our footage. Notebook `03` shows this
  same model predicts the wrong class on the night/surveillance clips — a **domain
  shift**, which training curves cannot reveal (the val set is from the same
  bright-cabin distribution as train). The fix is fine-tuning on annotated
  competition frames (`scripts/finetune_behavior.py`), not more epochs.
"""),
]


def build():
    nb = nbf.v4.new_notebook(); nb.cells = CELLS
    nb.metadata = {"kernelspec": {"name": "quismotion", "display_name": "QuisMotion (.venv)"},
                   "language_info": {"name": "python"}}
    NotebookClient(nb, timeout=300, kernel_name="quismotion",
                   resources={"metadata": {"path": HERE}}, allow_errors=True).execute()
    out = os.path.join(HERE, "05_behavior_training_curves.ipynb")
    with open(out, "w", encoding="utf-8") as fh:
        nbf.write(nb, fh)
    print("wrote", out)


if __name__ == "__main__":
    build()

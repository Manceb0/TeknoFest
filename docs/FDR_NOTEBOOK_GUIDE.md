# QuisMotion FDR Notebook Guide

This guide maps the current consolidated notebooks to the TEKNOFEST Final Design Report sections. The FDR should focus on the AI/video-labelling model: dataset, labels, architecture, training, testing, metrics, and limitations.

For the full Spanish delivery plan and the exact positioning we discussed, see `docs/TEKNOFEST_FDR_PLAN.md`.

For clean visual review, open `reports/notebooks_html_no_code/index.html`. GitHub
renders `.ipynb` files with code cells visible; the `notebooks_html_no_code`
exports hide code inputs and are better for reading/submission review.

## Executive status

The current evidence supports this claim:

> QuisMotion implements a real video-labelling pipeline using YOLOv8, OpenCV, OCR components, and optional Roboflow-assisted dataset workflows. Vehicle detection and proximity estimation work on the provided low-light parking-garage videos. The focused phone-vs-safe behavior detector performs strongly on a held-out public dataset. Cigarette/smoking and exact plate OCR are treated as domain-adaptation targets because the available TEKNOFEST-like footage is low-resolution, side-angle, and data-limited.

Do not claim robust cigarette/smoking detection yet. Do not claim reliable exact plate OCR at 464p.

## Current notebook map

| FDR section | Current notebook | What to extract |
|---|---|---|
| 1. Project Summary | `03_ai_architecture.ipynb`, `05_solution_testing.ipynb` | One-paragraph overview: raw video -> frame sampling -> detections -> labelled outputs. |
| 2. Dataset Preparation | `01_dataset_preparation.ipynb` | Dataset sources, class remapping, split strategy, balancing, augmentation and domain-gap notes. |
| 3.1 Problem Analysis | `02_problem_analysis.ipynb` | Low light, motion blur, reflections, occlusion, 464p plate limits, side-angle behavior difficulty and class imbalance. |
| 3.2 Solution Architecture | `03_ai_architecture.ipynb` | AI pipeline diagram, components, provider interfaces and output schema. |
| 3.3 Solution Details | `00_environment_setup.ipynb`, `03_ai_architecture.ipynb`, `04_model_training.ipynb` | YOLO model choices, image size, confidence thresholds, OCR preprocessing, tracking/risk signals, software stack. |
| 4. Solution Testing | `05_solution_testing.ipynb`, `backend/tests/test_real_videos.py`, `backend/runs/*` | Precision, recall, F1, mAP, FPS, real-video detection evidence and limitations. |
| 5. References | `README.md`, `docs/TEKNOFEST_FDR_PLAN.md` | YOLO/Ultralytics, OpenCV, Roboflow, Supervision, PyTorch and OCR references. |

The older notebooks are archived locally under `notebooks/_archive/` and should not be used as the main FDR evidence package.

## How to read overlay boxes

Use this legend when copying figures from `03_ai_architecture.ipynb`:

| Color | Meaning | Strength of claim |
|---|---|---|
| Cyan | Vehicle detection + track ID | Strong on the supplied clips. |
| Green | Occupant/person box in diagnostic views only | Auxiliary evidence of driver visibility, not a primary claim; omitted from the general overlay because low-light person boxes can be visually ambiguous. |
| Red | Behavior model evidence box, only when a specialized behavior detector returns a box | Report as phone/safe evidence only when backed by the evaluated model; smoking remains experimental. |
| Orange | Plate OCR ROI only in OCR-specific figures | Auxiliary ROI for OCR, not a general object detection claim. It is intentionally omitted from the general architecture overlay to avoid stale cached boxes. |

If a plate ROI is not aligned with the current vehicle, it should not be shown in the final figure; cached OCR votes are useful for text stability but can create stale visual boxes if drawn blindly.

## Key results to cite

### Vehicle detection on provided videos

From `backend/tests/test_real_videos.py`:

| Video | Detections | Max confidence | Bbox area growth |
|---|---:|---:|---:|
| `tekno-01.mp4` | 5 / 5 sampled timestamps | 0.944 | 0.125 -> 0.406 |
| `tekno-02.mp4` | 5 / 5 sampled timestamps | 0.951 | 0.120 -> 0.410 |
| `tekno-03.mp4` | 5 / 5 sampled timestamps | 0.926 | 0.019 -> 0.307 |

Interpretation: the vehicle detector is reliable on the provided clips, and bounding-box area growth is a measurable proxy for approach/proximity. Do not report physical speed in km/h without ground-truth calibration.

### Focused phone-vs-safe behavior model

From `backend/runs/focused_test/eval_summary.json`:

| Metric | Value |
|---|---:|
| Precision | 0.999 |
| Recall | 1.000 |
| F1 | 1.000 |
| mAP@50 | 0.995 |
| mAP@50-95 | 0.858 |
| FPS GPU 512px | 26.1 |

Required caveat:

> These metrics validate the focused phone-vs-safe model in the open-source dataset domain. Operational testing on TEKNOFEST-like low-light videos shows domain shift, especially for smoking/cigarette and plate OCR, therefore those labels are reported as experimental and require additional domain-specific annotation.

FPS caveat:

> The 26.1 FPS value is the saved benchmark for the focused behavior model at 512px. The integrated pipeline measurement in the current local notebook run falls back to CPU/ONNX Runtime and does not reach 10 FPS; real-time deployment requires CUDA/ONNX Runtime configuration, asynchronous sampling, or model optimization.

### Full/combined behavior attempts

Use these results to justify narrowing the reportable claim:

| Run | Best mAP@50 | Last precision | Last recall | Comment |
|---|---:|---:|---:|---|
| `behavior` public multi-class | 0.987 | 0.975 | 0.968 | Strong on public bright-cabin domain. |
| `behavior_focused` phone/safe | 0.995 | 0.997 | 0.998 | Best reportable behavior model. |
| `behavior_combined` phone/cigarette/safe | 0.851 best, 0.711 last | 0.642 | 0.827 | Weaker due to few/noisy cigarette labels and domain mismatch. |

The combined model is useful as research evidence, not as the main performance claim.

## Dataset preparation narrative

Recommended wording:

1. Public distracted-driving data was used to train the focused phone/safe behavior model.
2. Labels were remapped:
   - `Texting` and `Talking on the phone` -> `phone`;
   - `Safe Driving` -> `safe`;
   - unrelated classes were excluded from the focused experiment.
3. A stratified train/validation/test split was used so both `phone` and `safe` appear in every split.
4. The provided TEKNOFEST-like videos were used for operational validation and domain-gap analysis.
5. Cigarette/smoking remains data-limited because the visible object is tiny, low-resolution and frequently occluded.

## Preprocessing and augmentation to mention

| Technique | Why it matters |
|---|---|
| Resize to 512px | Balances accuracy and FPS. |
| Brightness/exposure variation | Simulates parking lots, headlights and low light. |
| Saturation variation | Handles color shifts from artificial lighting. |
| Small rotation | Simulates tilted cameras and angled vehicles. |
| Translation/scale | Handles off-center objects and distance changes. |
| Gaussian/motion blur | Simulates motion and low-quality frames. |
| Noise/compression | Simulates WhatsApp/CCTV artifacts. |
| Crop/occlusion | Simulates windshield, hands, steering wheel and columns. |
| Horizontal flip | Useful for behavior, but avoid for OCR evaluation because it flips text. |

## Supervision, Roboflow and NVIDIA

- Roboflow Supervision: useful now for overlays, visual QA, metrics, count tables and report images.
- Roboflow: useful as a modular dataset/hosted-inference workflow.
- NVIDIA TAO/DeepStream: viable future optimization for deployment, not the core current claim.

## Final claim hierarchy

Use this order in the FDR:

1. Proven: real vehicle detection on the supplied clips.
2. Proven: bbox area/proximity signal works as a measurable proxy.
3. Proven: phone/safe model works on a held-out public dataset.
4. Partially proven: operational videos show domain shift but still validate the pipeline.
5. Experimental: plate OCR on 464p videos.
6. Not yet robust: cigarette/smoking detection.

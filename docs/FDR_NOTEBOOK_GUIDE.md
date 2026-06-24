# QuisMotion FDR Notebook Guide

This guide maps the current notebooks and experiment outputs to the Final Design Report sections. The FDR scope, based on the organizer answers, is the AI/video-labelling model only: dataset, labels, model architecture, testing, metrics, and references. Do not spend report space on the mobile app, Number Verification, QoD API integration, or full backend deployment except as optional future integration context.

## Executive status

The current evidence supports this honest claim:

> QuisMotion implements a real video-labelling pipeline using YOLOv8, OpenCV, EasyOCR, and optional Roboflow-assisted domain adaptation. Vehicle detection and proximity estimation work reliably on the provided night parking-garage videos. Phone-vs-safe driver-behavior detection is strong in the open-source dataset domain. Smoking/cigarette detection remains the weakest label because the available competition frames are few, low-resolution, and side-angle, so it needs more annotated domain data before claiming robust performance.

Do not claim that cigarette/smoking is solved. Present it as the next fine-tuning target and explain why.

## Notebook map for the FDR

| FDR section | Use these notebooks | What to extract |
|---|---|---|
| 1. Project Summary | `01_vehicle_detection_qod_and_latency.ipynb`, `03_driver_behavior_and_occupants.ipynb`, `07_solution_testing.ipynb` | One-paragraph overview of the AI pipeline: video frames -> vehicle detection -> plate/behavior ROIs -> labelled outputs. |
| 2. Dataset Preparation | `06_dataset_preparation.ipynb`, `08_operational_analysis.ipynb`, scripts in `backend/scripts/` | Dataset sources, class remapping, split strategy, augmentation list, and domain-gap explanation. |
| 3.1 Problem Analysis | `02_license_plate_resolution_limit.ipynb`, `03_driver_behavior_and_occupants.ipynb`, `08_operational_analysis.ipynb` | Low light, motion blur, reflections, occlusion, 464p plate limit, side-angle behavior difficulty, class imbalance. |
| 3.2 Solution Architecture | `01_vehicle_detection_qod_and_latency.ipynb`, backend `LocalYOLOProvider` | Diagram and description of raw video -> frame sampling -> YOLOv8 -> tracking -> ROI OCR/behavior -> post-processing -> JSON labels. |
| 3.3 Solution Details | `05_behavior_training_curves.ipynb`, `06_dataset_preparation.ipynb`, `backend/app/services/local_yolo_provider.py` | YOLOv8 model choices, image size, confidence threshold, OCR preprocessing, tracking, risk signals, libraries. |
| 4. Solution Testing | `07_solution_testing.ipynb`, `backend/tests/test_real_videos.py`, `backend/runs/*` | Tables: precision, recall, F1, mAP, FPS, real-video detection confidence, limitations. |
| 5. References | `README.md`, PDR references, Roboflow dataset pages | YOLO/Ultralytics, EasyOCR, Roboflow Universe datasets, DeepSORT/BoT-SORT if mentioned, OpenCV. |

## Key results to cite

### Vehicle detection on provided competition-like videos

From `backend/tests/test_real_videos.py`:

| Video | Detections | Max confidence | Bbox area growth |
|---|---:|---:|---:|
| `tekno-01.mp4` | 5 / 5 sampled timestamps | 0.944 | 0.125 -> 0.406 |
| `tekno-02.mp4` | 5 / 5 sampled timestamps | 0.951 | 0.120 -> 0.410 |
| `tekno-03.mp4` | 5 / 5 sampled timestamps | 0.926 | 0.019 -> 0.307 |

Interpretation: the vehicle detector is reliable on the provided clips and the increasing bounding-box area is a measurable proxy for approach/proximity.

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

Per class:

| Class | Precision | Recall | F1 | mAP@50 |
|---|---:|---:|---:|---:|
| phone | 0.999 | 1.000 | 1.000 | 0.995 |
| safe | 1.000 | 1.000 | 1.000 | 0.995 |

Important wording: “These metrics validate the focused behavior detector in the open-source dataset domain. The operational notebook shows a domain shift when applying it to low-light, side-angle competition footage, therefore additional domain-specific annotation is required for smoking/cigarette and robust phone detection.”

### Full/combined behavior attempts

Use these results to justify narrowing the scope and improving the dataset:

| Run | Best mAP@50 | Last precision | Last recall | Comment |
|---|---:|---:|---:|---|
| `behavior` public multi-class | 0.987 | 0.975 | 0.968 | Strong on public bright-cabin domain. |
| `behavior_focused` phone/safe | 0.995 | 0.997 | 0.998 | Best reportable model for FDR. |
| `behavior_combined` phone/cigarette/safe | 0.851 best, 0.711 last | 0.642 | 0.827 | Weaker due to few/noisy cigarette labels and domain mismatch. |

The combined model is useful as a research attempt, not as the main claimed result.

## Dataset preparation narrative

Recommended wording:

1. Public dataset: Roboflow Universe “Distracted Driving” dataset, approximately 8,900 labelled images.
2. Labels were remapped for a focused FDR model:
   - `Texting` and `Talking on the phone` -> `phone`
   - `Safe Driving` -> `safe`
   - other classes removed from the focused experiment.
3. A stratified 70/15/15 train/validation/test split was used so that both `phone` and `safe` appear in every split.
4. The provided competition-like videos were used for operational validation and for extracting domain frames.
5. A small cabin/competition dataset was prepared with labels such as `driver`, `passenger`, `phone`, and `cigarette`, but the cigarette class is still data-limited.

## Preprocessing and augmentation options

### Already documented / reasonable for the report

These are safe to include because they match the existing scripts/results:

| Technique | Why it matters for this project |
|---|---|
| Resize to 512px | Standardizes inference/training input and keeps FPS high. |
| HSV brightness/value (`hsv_v`) | Simulates dark parking lots, headlight glare, and exposure variation. |
| HSV saturation (`hsv_s`) | Handles color shifts from garage lighting and compression. |
| Small rotation (`degrees=5-8`) | Simulates slight camera tilt and vehicle angle changes. |
| Translation (`translate=0.1-0.12`) | Improves robustness when the driver/vehicle is not centered. |
| Scale (`scale=0.5-0.6`) | Simulates vehicle distance and zoom differences. |
| Horizontal flip (`fliplr=0.5`) | Useful for left/right driving-side variation, but mention that sign/plate OCR evaluation should avoid flipped text. |
| Mosaic (`mosaic=1.0`) | Helps object detectors generalize to mixed scales/backgrounds. |
| MixUp (`mixup=0.1`) | Used in the combined fine-tuning attempt to reduce overfitting. |
| Gaussian blur | Simulates motion blur and low-quality frames. |
| Noise/compression artifacts | Should be added or documented as future augmentation for CCTV/WhatsApp compression robustness. |

### Recommended additions for the next notebook/report iteration

Add a small “Ablation and Robustness” notebook or section with examples:

| Augmentation family | Variants to show |
|---|---|
| Resize | 320, 416, 512, 640 inference comparison: FPS vs confidence. |
| Brightness/exposure | darker 0.4x, normal, brighter 1.5x. |
| Saturation | desaturated, normal, oversaturated. |
| Blur | Gaussian blur, motion blur horizontal/diagonal. |
| Noise | Gaussian noise, salt-and-pepper noise. |
| Compression | JPEG quality 95, 70, 45. |
| Rotation | -8, -4, 0, +4, +8 degrees. |
| Crop/occlusion | crop windshield/plate zones, partial occlusion rectangles. |
| Night/rain simulation | darkening + blur + noise as a combined stress test. |

For the FDR, you do not need to train every variant. It is enough to document the training augmentations used and show a stress-test table on the validation clips if time allows.

## Smoking/cigarette weakness

Your intuition is correct: cigarette is the weak link.

Why:

- The supplied videos are only 832x464, so the cigarette is tiny.
- Only around tens of competition frames contain visible smoking evidence.
- The public distracted-driving dataset supports phone/safe better than cigarette in this side-angle night domain.
- COCO YOLO has no `cigarette` class, so off-the-shelf detection cannot solve this directly.
- The driver is visible through windshield/side glass, with reflections and occlusion.

How to write it:

> The smoking/cigarette label is treated as a domain-adaptation target rather than a fully solved class. Initial labels from the competition-like clips are visually weak due to 464p resolution and side-angle occlusion. Therefore, the current FDR reports strong vehicle and phone/safe performance while identifying cigarette detection as requiring more manually labelled domain data and/or prompt-assisted auto-labelling followed by human correction.

What to do next:

1. Extract more frames from smoking moments, not uniformly from the whole video.
2. Label cigarette only when visually confirmable.
3. Add a `hand_to_mouth` auxiliary label if the cigarette object is too small.
4. Use YOLO-World/SAM as an annotation assistant, but manually verify labels.
5. Train `phone/cigarette/safe` only after the cigarette class has enough examples.

## What is missing before the FDR is strong

| Priority | Missing item | Why it matters |
|---|---|---|
| P0 | A clean exported PDF/HTML report from notebooks 06 and 07 | The evaluator needs readable evidence, not raw notebooks only. |
| P0 | Dataset table with exact image/label counts per split | Required by Section 2. |
| P0 | A concise architecture diagram for AI-only pipeline | Required by Section 3.2. |
| P0 | Honest limitation paragraph for cigarette/smoking and OCR | Prevents overclaiming. |
| P1 | Robustness/augmentation stress-test notebook | Strengthens Section 3.1 and 4. |
| P1 | Combined model evaluation table with per-class cigarette metrics | Shows why cigarette is not yet trusted. |
| P1 | OCR metric definition | Exact plate match and character-level accuracy, even if small sample. |
| P2 | Speed estimation metric definition | If ground truth speed is missing, report proxy: bbox growth consistency, not km/h accuracy. |

## Recommended notebook list to keep

For final submission clarity, keep these as the main notebooks:

1. `00_environment_and_gpu.ipynb` - reproducibility.
2. `01_vehicle_detection_qod_and_latency.ipynb` - vehicle/proximity/FPS.
3. `02_license_plate_resolution_limit.ipynb` - OCR limitation and need for higher resolution.
4. `03_driver_behavior_and_occupants.ipynb` - behavior feasibility and domain gap.
5. `05_behavior_training_curves.ipynb` - training curves and overfitting check.
6. `06_dataset_preparation.ipynb` - FDR Section 2.
7. `07_solution_testing.ipynb` - FDR Section 4.
8. `08_operational_analysis.ipynb` - field validation on real clips.

Keep `09` and `10` as optional research/future-work notebooks, not core evidence, unless the FDR has space for “robustification plan”.

## Suggested FDR claim hierarchy

Use this hierarchy in the report:

1. Proven: real vehicle detection on competition-like footage.
2. Proven: bbox area/proximity signal works and is measurable.
3. Proven: phone/safe model works on held-out public dataset.
4. Partially proven: phone events appear in operational clips, but not continuous per-frame detection.
5. Experimental: plate OCR on 464p clips.
6. Not yet robust: cigarette/smoking detection.

That order is more credible than pretending all classes are equally solved.

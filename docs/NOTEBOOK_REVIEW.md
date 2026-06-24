# Notebook Review - QuisMotion FDR

Review date: 2026-06-24

Scope: notebooks `00` to `10` under `notebooks/`, with outputs already saved in the `.ipynb` files and experiment artifacts under `backend/runs/`.

## Executive verdict

The notebooks are technically useful and mostly runnable, but they need editorial cleanup before being shown as a polished FDR evidence package.

Main issue: the technical story is currently spread across too many notebooks. Some notebooks are strong evidence, while others are exploratory future work. The FDR should not treat all of them equally.

Recommended core notebooks for FDR:

1. `06_dataset_preparation.ipynb`
2. `07_solution_testing.ipynb`
3. `01_vehicle_detection_qod_and_latency.ipynb`
4. `08_operational_analysis.ipynb`
5. `02_license_plate_resolution_limit.ipynb`
6. `05_behavior_training_curves.ipynb`

Optional / future-work notebooks:

- `03_driver_behavior_and_occupants.ipynb`
- `04_duckdb_vss_similarity.ipynb`
- `09_robustification_world_sam.ipynb`
- `10_sam3_concept_segmentation.ipynb`

## Notebook-by-notebook review

| Notebook | FDR role | Status | Main issue | Action |
|---|---|---|---|---|
| `00_environment_and_gpu` | Reproducibility appendix | Good | Too deployment-focused for main FDR | Keep short, maybe appendix only. |
| `01_vehicle_detection_qod_and_latency` | Strong evidence | Good | Mentions QoD, but FDR scope is AI-only | Reword QoD as proximity threshold / labelled event trigger. |
| `02_license_plate_resolution_limit` | Strong limitation evidence | Good | “None recover the plate” is slightly too absolute because OCR gives partial reads | Say “not reliable at 464p” instead of impossible. |
| `03_driver_behavior_and_occupants` | Useful but exploratory | Medium | Mixes behavior, occupants, Roboflow hosted tests, domain-gap claims | Split into clearer subsections or summarize only. |
| `04_duckdb_vss_similarity` | Not needed for FDR | Optional | Vector search is outside AI labelling model scope | Move to appendix/future system. |
| `05_behavior_training_curves` | Strong evidence | Good | Opening text says `runs/behavior/results.csv`, but output uses focused model | Fix wording to `runs/behavior_focused/results.csv`. |
| `06_dataset_preparation` | Essential | Good but incomplete | Needs stronger exact dataset count + annotation policy + augmentation table | Expand before final FDR. |
| `07_solution_testing` | Essential | Strong but risky | F1=1.00 may sound overclaimed | Add domain-scope caveat directly beside metrics. |
| `08_operational_analysis` | Very important honesty notebook | Good | Shows weak transfer, could look like failure if not framed | Present as field validation/domain-gap analysis. |
| `09_robustification_world_sam` | Future work / optional | Medium | Very low concept confidence, not final evidence | Keep as annotation-assistance strategy only. |
| `10_sam3_concept_segmentation` | Future work only | Weak for FDR | No checkpoint, no real output | Do not include in main FDR. |

## Detailed observations

### 00 - Environment and GPU

Outputs:

- Python 3.12.5
- Torch 2.12.1 + CUDA 12.6
- GPU: NVIDIA GeForce RTX 4060, 8 GB

Good for reproducibility, but the FDR does not need much hardware detail unless discussing FPS.

Recommended FDR usage: one sentence in Solution Details or Testing.

### 01 - Vehicle detection, proximity and latency

Strong notebook. It proves the vehicle detector works on the supplied footage and that bounding-box area grows as the vehicle approaches.

Outputs:

- Max bbox area crosses 15%.
- YOLOv8x GPU output in notebook: 47.2 ms/frame, 21.2 FPS.
- YOLOv8s GPU output: 66.9 ms/frame, 14.9 FPS.

Concern: the final markdown says “YOLOv8x runs ~24 ms/frame (~40 FPS)”, but the saved output shows 47.2 ms/frame / 21.2 FPS. This must be fixed. Use the saved measured output unless you rerun and get a new value.

Recommended correction:

> In the saved run, YOLOv8x reached 21.2 FPS on RTX 4060, which is above the target 10 FPS stream sampling rate.

### 02 - License plate resolution limit

Useful and honest. It shows:

- source video is 832x464;
- OCR gives unstable partial reads such as `DIL 0577`;
- hosted plate detector returned no detections;
- conclusion: OCR is resolution-limited.

Recommended wording change:

Avoid:

> None recover the plate, because the pixels are not there.

Use:

> At 464p, OCR is not reliable enough for reportable exact-match accuracy; higher-resolution frames or a plate-specific model are required.

This avoids overclaiming while preserving the point.

### 03 - Driver behavior and occupants

Technically useful, but it mixes several ideas:

- COCO cannot detect cigarette;
- public behavior model does not transfer;
- occupant/person detection works after crop+upscale;
- Roboflow hosted behavior test returned empty predictions.

This notebook is good for explaining problem analysis, not for claiming final behavior accuracy.

Most important output:

- Cell-phone detections in cabin at 0.05 confidence: 0.
- Hosted behavior detections: empty.
- Occupants in cabin crop: 2, confidences [0.84, 0.47].

Recommended FDR use:

Use it in Section 3.1 Problem Analysis, not Section 4 Testing.

### 04 - DuckDB VSS similarity

Works technically:

- embedded 18 frames;
- 512-dimensional embeddings;
- DuckDB HNSW similarity query works.

But it is not central to the FDR because the organizers clarified the FDR is about the AI/video labelling model, not the full incident-retrieval system.

Recommendation: remove from main FDR or mention in future work.

### 05 - Behavior training curves

Strong notebook. It supports the claim that the focused behavior model does not overfit in its training domain.

Outputs:

- epochs: 40
- final train loss: 1.897
- final val loss: 2.024
- train/val gap: +7%
- best epoch: 21
- mAP50: 0.995
- mAP50-95: 0.826
- verdict: not overfitting

Concern: markdown says logs come from `runs/behavior/results.csv`, but output actually references `backend/runs/behavior_focused/results.csv`.

Fix that wording.

### 06 - Dataset preparation

Essential notebook.

Outputs:

| Split | Images | Phone boxes | Safe boxes | Ratio |
|---|---:|---:|---:|---:|
| train | 2224 | 1398 | 826 | 70% |
| valid | 477 | 300 | 177 | 15% |
| test | 478 | 300 | 178 | 15% |
| total | 3179 | - | - | 100% |

Good. But it needs more FDR detail:

- exact original dataset name/version;
- exact filtering/remapping rules;
- whether images with removed classes were excluded or converted to background;
- annotation format: YOLO normalized bounding boxes;
- manual competition-frame extraction count: 85 frames;
- cabin dataset count: 130 labelled images in `tmp/cabin_dataset`;
- combined dataset count: 3264 images/labels;
- augmentation table, not only example images.

Recommended addition: add a final markdown cell named “FDR-ready summary”.

### 07 - Solution testing

Essential and strong, but risky if presented without caveat.

Outputs:

| Metric | Value |
|---|---:|
| Precision | 0.999 |
| Recall | 1.000 |
| F1 | 1.000 |
| mAP50 | 0.995 |
| mAP50-95 | 0.858 |
| FPS | 26.1 |

Per class:

| Class | P | R | F1 | mAP50 |
|---|---:|---:|---:|---:|
| phone | 0.999 | 1.000 | 1.000 | 0.995 |
| safe | 1.000 | 1.000 | 1.000 | 0.995 |

Concern:

The “F1=1.00” result is believable only if framed as focused phone/safe evaluation on a held-out open-source test set. It should not be presented as performance on the competition videos or smoking/cigarette.

Required caveat:

> These metrics evaluate the focused phone-vs-safe model in the open-source dataset domain. Operational testing on the TEKNOFEST clips is reported separately and shows domain shift under low-light side-angle footage.

### 08 - Operational analysis

Very important. This is the notebook that keeps the report honest.

Outputs:

- sampled frames:
  - tekno-01: 85
  - tekno-02: 92
  - tekno-03: 77
- behavior detection rate:
  - tekno-01: phone 6%, safe 8%, none 69%
  - tekno-02: phone 7%, safe 7%, none 63%
  - tekno-03: phone 6%, safe 36%, none 34%
- plate reads are inconsistent.

This can look bad if not framed carefully, but it is actually valuable: it proves you tested outside the training domain.

Recommended interpretation:

> The model detects behavior events intermittently in operational footage because the driver is visible only during part of the pass and the side-angle/night domain differs from the public training set. This motivates domain-specific fine-tuning with manually labelled competition frames.

### 09 - YOLO-World + SAM2 robustification

Interesting, but not main evidence.

Good:

- proposes open-vocabulary labels (`cigarette`, `mobile phone`, `person`);
- conceptually solves cigarette vs phone semantics;
- useful for assisted annotation.

Risk:

- confidence is low (~0.03-0.09 according to markdown);
- can look experimental;
- may distract from the core FDR if included too prominently.

Use as future-work or annotation-assistance method.

### 10 - SAM3

Do not use as FDR evidence.

Reason:

- `sam3.pt` not present;
- cell skipped;
- no real output.

Can be listed as future work only if needed.

## Highest-priority fixes before submission

1. Fix the mismatch in `01`: saved output says YOLOv8x = 47.2 ms / 21.2 FPS, not 24 ms / 40 FPS.
2. Fix `05` wording to say `behavior_focused/results.csv`.
3. Add caveat beside `07` metrics: focused phone/safe open-source domain only.
4. Expand `06` with exact dataset counts and annotation/remapping policy.
5. Move `09` and `10` out of the main FDR evidence path.
6. Create one consolidated `11_fdr_report_pack.ipynb` with only final tables, final charts, annotated examples, and text ready for the report.

## Recommended final FDR evidence stack

Use this order in the report:

1. Dataset preparation table from `06`.
2. Vehicle detection/proximity results from `01`.
3. Training curves from `05`.
4. Formal phone/safe metrics from `07`.
5. Operational domain-gap results from `08`.
6. OCR limitation from `02`.
7. Future annotation strategy from `09`, optional.

This tells a credible story: strong core model, measured limitations, and a clear data-driven improvement path.

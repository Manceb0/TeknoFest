# QuisMotion notebook reports

Use these notebooks as the evidence base for the FDR AI/video-labelling report.

## Recommended reading order

1. `00_environment_and_gpu.ipynb` - runtime, packages, GPU/CPU context.
2. `06_dataset_preparation.ipynb` - dataset sources, split, balancing, augmentation.
3. `01_vehicle_detection_qod_and_latency.ipynb` - vehicle detection, proximity signal, FPS.
4. `02_license_plate_resolution_limit.ipynb` - why OCR is limited by 464p footage.
5. `03_driver_behavior_and_occupants.ipynb` - behavior feasibility and domain mismatch.
6. `05_behavior_training_curves.ipynb` - training/validation curves.
7. `07_solution_testing.ipynb` - formal metrics and “why we trust it”.
8. `08_operational_analysis.ipynb` - what happens on the three supplied clips.

## Main numbers to cite

- Real-video vehicle detector: max confidence 0.926-0.951 across the three supplied videos.
- Focused phone/safe model: Precision 0.999, Recall 1.000, F1 1.000, mAP@50 0.995, mAP@50-95 0.858.
- Inference speed for focused behavior model: 26.1 FPS at 512px on GPU.

## Important limitation

Do not claim robust cigarette/smoking detection yet. The cigarette label is weak because the available frames are low-resolution, side-angle, and few. Treat cigarette detection as a domain-adaptation target that requires more labelled competition-like frames.

For the full section-by-section writing guide, see `docs/FDR_NOTEBOOK_GUIDE.md`.

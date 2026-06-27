# QuisMotion notebook reports

These notebooks are the clean evidence package for the TEKNOFEST FDR AI/video-labelling report.

## Recommended reading order

1. `00_environment_setup.ipynb` - runtime, packages, GPU/CPU context.
2. `01_dataset_preparation.ipynb` - dataset sources, split, balancing and augmentation.
3. `02_problem_analysis.ipynb` - low light, motion blur, occlusion, 464p plate limits and domain shift.
4. `03_ai_architecture.ipynb` - raw video to labelled output pipeline.
5. `04_model_training.ipynb` - YOLO training setup, curves and technical decisions.
6. `05_solution_testing.ipynb` - precision, recall, F1, mAP, FPS and "why we trust it".

## Main numbers to cite

- Real-video vehicle detector: max confidence 0.926-0.951 across the three supplied videos.
- Focused phone/safe model: Precision 0.999, Recall 1.000, F1 1.000, mAP@50 0.995, mAP@50-95 0.858.
- Inference speed for focused behavior model: 26.1 FPS at 512px on GPU.

## Important limitation

Do not claim robust cigarette/smoking detection yet. The cigarette label is weak because the available frames are low-resolution, side-angle and few.

Do not claim reliable exact plate OCR from the 464p clips. Treat OCR as experimental unless higher-resolution frames or a plate-specific dataset are added.

## Archived notebooks

Older exploratory notebooks are kept locally in `notebooks/_archive/` for reference, but the FDR should use the consolidated `00` to `05` notebooks above.

For the full section-by-section writing guide, see `docs/FDR_NOTEBOOK_GUIDE.md` and `docs/TEKNOFEST_FDR_PLAN.md`.

# Notebook Review - QuisMotion TEKNOFEST FDR

Review date: 2026-06-27

Scope: consolidated notebooks `00` to `05` under `notebooks/`, experiment artifacts under `backend/runs/`, and real-video validation in `backend/tests/test_real_videos.py`.

## Executive verdict

The notebook package is now aligned with the TEKNOFEST FDR structure. It is viable as an evidence base if the report keeps the claims scoped correctly:

- Strong claim: vehicle detection and proximity proxy on the provided videos.
- Strong claim: phone/safe behavior model on the held-out public dataset domain.
- Careful claim: operational validation shows domain shift on low-light TEKNOFEST-like videos.
- Experimental claim only: cigarette/smoking detection and exact plate OCR.
- Performance caveat: the focused behavior model has a saved 26.1 FPS benchmark, but the current integrated local pipeline run falls back to CPU/ONNX Runtime and does not yet meet 10 FPS.

The notebooks should not present cigarette detection or plate OCR as solved.

## Current notebook review

| Notebook | FDR role | Status | Main action |
|---|---|---|---|
| `00_environment_setup.ipynb` | Reproducibility / Solution Details | Good | Keep short; use as appendix or technical setup evidence. |
| `01_dataset_preparation.ipynb` | Section 2 Dataset Preparation | Essential | Use for dataset sources, split, balancing and augmentation. |
| `02_problem_analysis.ipynb` | Section 3.1 Problem Analysis | Essential | Use to explain low light, blur, occlusion, 464p plate limits and domain gap. |
| `03_ai_architecture.ipynb` | Sections 3.2 / 3.3 Architecture and Details | Essential | Use as the main architecture explanation. |
| `04_model_training.ipynb` | Section 3.3 Solution Details | Good | Use for training curves and model-selection narrative. |
| `05_solution_testing.ipynb` | Section 4 Solution Testing | Essential | Use for metrics, FPS and trust argument, with caveats. |

## Recommended FDR evidence order

1. Start with the architecture diagram from `03_ai_architecture.ipynb`.
2. Explain the dataset in `01_dataset_preparation.ipynb`.
3. Use `02_problem_analysis.ipynb` to justify why the task is hard.
4. Use `04_model_training.ipynb` to show the training process.
5. Use `05_solution_testing.ipynb` to report metrics and evidence.
6. Mention `00_environment_setup.ipynb` only for reproducibility.

## Claims that are safe

| Claim | Evidence |
|---|---|
| The system processes real videos through a modular AI pipeline. | Backend/frontend implementation and `03_ai_architecture.ipynb`. |
| Vehicle detection works on the supplied clips. | `backend/tests/test_real_videos.py`. |
| Bounding-box area growth is a usable proximity proxy. | Real-video detection summaries. |
| Phone/safe behavior detection performs strongly in the public dataset domain. | `backend/runs/focused_test/eval_summary.json` and `05_solution_testing.ipynb`. |
| The system is modular enough to test new videos. | Demo video folder, backend providers and frontend video selection. |

## Claims that require caveats

| Claim | Required caveat |
|---|---|
| Behavior detection works | Say phone/safe works in the public dataset domain; TEKNOFEST-like videos show domain shift. |
| OCR works | Say OCR is experimental and resolution-limited at 464p. |
| Cigarette/smoking detection works | Do not claim robustness; call it a data-limited domain-adaptation target. |
| Speed detection works | Report proximity proxy unless ground-truth speed calibration is added. |

## Supervision, Roboflow and NVIDIA recommendation

Roboflow Supervision is useful and realistic for the current FDR as a support layer:

- visual overlays;
- annotation quality checks;
- detection tables;
- dataset analysis;
- report-ready figures.

Roboflow itself is useful as a modular dataset/hosted inference path.

NVIDIA TAO or DeepStream should be described only as future deployment optimization, not as the core implemented model.

## Final reviewer note

The project is not weak because it admits limitations. It becomes more credible because the report separates validated performance from experimental work. TEKNOFEST evaluators are more likely to trust a system that shows real tests, metrics and honest failure modes than one that claims perfect detection for low-resolution cigarette and plate OCR.

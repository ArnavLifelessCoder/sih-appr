# NB11: final population cascade with CV and temporal modeling

Import `notebooks/kaggle/11_cv_ts_cascade.ipynb` into Kaggle. The notebook is self-contained and does not clone GitHub.

Use these settings:

1. Accelerator: GPU T4
2. Internet: On
3. Input: complete saved NB2 output
4. Input: latest saved NB6 v2 output with at least 295 NPZ chips
5. Input: complete saved NB8 output with the three final models and population LOCO predictions

Do not attach NB1, NB4, NB6 v1, NB9, NB10, or India data. NB2 already contains the six foreign detection caches and common-window feature files. Extra old notebook outputs can create duplicate filenames and fail preflight.

Run with Save Version and Save and Run All. The notebook will:

- verify every required input before training
- extract an optional frozen TorchGeo Sentinel-2 embedding
- build 36 monthly FIRMS bins for all 1.63 million foreign sources
- train a 27,873-parameter TCN using two deterministic PU bags
- evaluate NB8, the TCN, and a fixed 80:20 rank blend at population prevalence
- run strict nested country selection for the image reranker
- preserve NB8 automatically if a challenger fails the anti-overfit guard
- create `nb11_multimodal_results.zip`

The main population result is in `11_stage1_country_metrics.csv`. Read macro country PR-AUC and worst-country PR-AUC first. `11_stage1_gate_metrics.csv` reports precision, source recall, and site recall at fixed top fractions.

The image result is in `11_country_metrics.csv`. Use the `nested_champion` rows for the honest model-selection procedure. Stage B uses an enriched 294-source pilot, so its F1 and threshold do not estimate deployment precision.

India has already been inspected and has no matching imagery. This run never loads India. Until India chips are acquired, India inference remains Stage A only and any later India comparison must be reported as exploratory, not as a new untouched holdout.

# Result evidence registry

This directory contains compact evidence copied from completed Kaggle outputs. Filenames
are kept unchanged so every claim can be traced back to its native run artifact.

| Directory | Run | Retained evidence |
|---|---|---|
| `01_audit/` | local audit | schema, file inventory, and quality summary |
| `02_source_clustering/` | source construction | country source summary |
| `04_baselines/` | initial baselines | rule and Isolation Forest metrics |
| `stage05_lightgbm/` | original tabular model | ablations, feature importance, and early LOCO diagnostics |
| `stage05b_robust_tabular/` | common-window PU evaluation | corrected LOCO, manifest, and model summary |
| `stage05c_balanced_tabular/` | weighting study | country, site, threshold, and feature evidence |
| `stage05d_nested_tabular/` | nested tabular selection | outer-country metrics, inner decisions, and spatial map |
| `nb04_cv_pilot/` | imagery smoke test | sample, manifest, config, WorldCover keys, and preview |
| `nb06_image_context/` | 600-source imagery preparation | coverage, QA, compact image features, state, and preview |
| `nb08_domain_revamp/` | regularized transfer variants | nested metrics, feature importance, and manifest |
| `nb09_final_fusion/` | first multimodal comparison | branch summary, country metrics, QA cohort, and compact predictions |
| `nb10_india_historical/` | superseded NB8 India evaluation | source metrics, site metrics, and manifest |
| `nb11_multimodal/` | temporal and cascade study | branch, gate, country, and TCN diagnostics |
| `nb12_cv_tabular/` | guarded fusion discovery | branch, country, review-budget, schema, and compact predictions |
| `nb12b_confirmatory/` | final fresh-seed confirmation | acceptance rules, stability, thresholds, review queue, manifest, and predictions |

Large raw detections, per-country feature populations, image chips, sidecar metadata,
population-scale predictions, logs, and duplicate copies of source code are excluded.
Together those external run folders exceed one gigabyte and can be regenerated. The
retained result evidence is about 7.5 MB.

Important status note: the India files record a historical evaluation with a superseded
NB8 model. The final NB12b guarded ensemble has not been evaluated on India, but the
project can no longer describe India as a pristine untouched holdout.

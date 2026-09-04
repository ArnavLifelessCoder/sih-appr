# NB12: Sentinel CV plus FIRMS tabular

NB12 is the final foreign-country model-selection run. India is not loaded in this run.
A separate historical India evaluation with the superseded NB8 model already exists,
so this run does not preserve a pristine project-level holdout.
It keeps the proven NB9 compact image and thermal model as a protected fallback,
then tests whether a stronger computer-vision specialist adds reliable signal.

## What the model uses

The structured branch combines the common-window FIRMS features with the compact
Sentinel-2 and WorldCover features that already improved macro country PR-AUC
from 0.8138 to 0.9059 in NB9.

The new visual branch adds:

- A frozen ResNet-18 pretrained on Sentinel-2 RGB imagery.
- Full 2 km and central 1 km views with rotation and flip averaging.
- Fold-local L2 normalization and 12-component PCA.
- All 77 multispectral and WorldCover features.
- Spatial morphology for spectral center-to-context contrast, gradients, built-up
  components, edge density, and compactness.
- PU-bagged regularized logistic probes because unmatched sources are unlabeled,
  not reliable negatives.

No TorchGeo, timm, Kornia, or end-to-end CNN training is required. The checkpoint
is pinned by URL and full SHA-256, and the loader uses a self-contained ResNet-18.

## Leakage and overfitting controls

India is never loaded. Every scaler, PCA, PU probe, LightGBM, fusion weight,
threshold, and branch decision is fitted inside the current outer training
countries.

The reported `nested_champion` score is the outer-country result of the complete
selection policy. It is not the post-hoc maximum of several outer scores.

The visual residual uses logit calibration fitted on training-country scores.
It does not calculate ranks from the label-enriched held-out queue. This makes
the score reproducible without using the composition of the test batch.

The challenger is rejected unless it satisfies all of these conditions against
the compact baseline:

- Macro country PR-AUC gain of at least 0.010.
- Improvement in at least four of six countries.
- Worst country PR-AUC change no worse than -0.020.
- Indonesia PR-AUC change no worse than -0.020.
- Macro F1 change no worse than -0.010.
- Paired bootstrap lower bound no worse than -0.010.

If the nested policy fails, the output deliberately keeps the compact NB9 model.
This is a valid CV plus tabular fallback because NB9 already uses Sentinel-2 and
WorldCover features.

## Evidence before the final Kaggle run

The frozen encoder and all real QA chips were exercised locally. In a diagnostic
that does not replace the final nested run:

| Branch | Macro country PR-AUC |
| --- | ---: |
| Protected compact CV plus tabular | 0.9059 |
| Frozen visual specialist alone | 0.8029 |
| Calibrated visual residual, alpha 0.50 | 0.9319 |

At alpha 0.50, five countries improved. Indonesia changed by about -0.0011.
This is evidence that the branch is worth testing, not a final performance claim.
NB12 selects alpha and the branch only inside the nested country protocol.

## Kaggle setup

Use the notebook at `notebooks/kaggle/12_cv_tabular.ipynb`.

Set:

- Accelerator: GPU T4.
- Internet: On.

Attach only:

1. The saved NB2 output containing exactly one common-window feature parquet for
   each of Algeria, Angola, Indonesia, Iraq, Libya, and Nigeria.
2. The latest NB6 v2 saved output containing `pilot_sources.csv`,
   `image_features.parquet`, `image_quality.csv`, `run_state.json`,
   `feature_manifest.json`, and the successful NPZ chips.

Do not attach NB4, NB6 v1, NB9, NB10, or NB11. Duplicate files are rejected by
the preflight checks.

Use Save Version, choose Save and Run All, then download
`nb12_cv_tabular_results.zip` from Output. A free T4 should be sufficient. The
expected workload is a frozen 294-chip embedding pass plus low-capacity tabular
and logistic models, not CNN fine-tuning.

## What the ZIP contains

- Country metrics for every diagnostic and eligible branch.
- The fully nested branch-selection estimate.
- Held-country predictions and fixed-review-budget metrics.
- Inner branch and alpha decisions.
- Compact fallback models and any selected deployment model.
- The frozen visual PU model and score-calibration references.
- Cached embeddings and morphology features.
- Exact schemas, hashes, manifests, and both Python implementations.

## Highest-value follow-up

The current unlabeled image sample is mostly random background. A repository
audit found 159 high-confidence foreign GEM facilities with persistent FIRMS
sources and no existing successful chip, including cement kilns, thermal power,
steel, and mixed industrial sites. Acquiring those chips is the strongest next
CV improvement because it teaches the model industrial morphology rather than
only catalogued gas-flare appearance. Those weak facility labels must stay out
of the predictive feature matrix. Any later India score must use the frozen
foreign design without retuning.

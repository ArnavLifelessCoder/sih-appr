# SIH26162: persistent industrial thermal-source detection

This repository groups repeated NASA FIRMS detections into candidate sources and ranks
sources that behave like persistent industrial heat. Known EOG gas-flare sites provide
positive supervision. Unmatched sources are unlabelled, not confirmed negatives, because
they can include kilns, cement plants, steelworks, missed flares, crop burning, or wildfire.

## Current result

The frozen foreign-country model is the NB12b guarded CV plus tabular ensemble. On one
stable, EOG-enriched panel of 294 sources from Algeria, Angola, Indonesia, Iraq, Libya,
and Nigeria, its country-macro scores were:

| Model | Macro F1 | Macro PR-AUC | Macro ROC-AUC | Worst-country PR-AUC |
|---|---:|---:|---:|---:|
| Compact tabular baseline | 0.767 | 0.905 | 0.920 | 0.825 |
| Guarded CV plus tabular | **0.817** | **0.929** | **0.948** | 0.840 |

All 12 preregistered acceptance checks passed. The PR-AUC gain was positive for all
three fresh seeds. The grouped country-stratified bootstrap interval for the mean gain
was -0.0006 to 0.0515, so the experiment supports the non-inferiority guard but does not
establish a conventionally significant positive gain. These panel metrics compare models;
they are not population precision estimates.

India requires careful wording. A historical India run was completed with the superseded
NB8 model and produced F1 0.151 and PR-AUC 0.142 on 706,686 sources. The final NB12b
guarded model has not been evaluated on India. Therefore India is not a pristine
project-level holdout, although it remained excluded from NB12 and NB12b training,
selection, calibration, and thresholding.

See [RESULTS.md](RESULTS.md) for the complete progression and limitations.

## Data audit

The source archive contains 19,342,072 FIRMS detections from seven countries over
2019-2024. The audit found no nulls, malformed dates, or exact duplicate rows. The
training labels contain 1,342 active EOG flare sites across Algeria, Nigeria, Iraq,
and Libya.

| Country | FIRMS detections | Coverage | NOAA-20 availability |
|---|---:|---|---|
| India | 7,594,269 | 2019-2024 | yes |
| Angola | 4,869,716 | 2022-2024 | no |
| Nigeria | 3,914,564 | 2021-2024 | yes |
| Iraq | 1,289,983 | 2021-2024 | yes |
| Algeria | 744,252 | 2021-2024 | yes |
| Indonesia | 501,771 | 2022-2024 | no |
| Libya | 427,517 | 2021-2024 | yes |

NOAA-20-derived features are excluded because missing coverage in Angola and Indonesia
would reveal country identity. The model also excludes latitude, longitude, country,
NASA `type`, and EOG distance.

## Method

1. Audit and normalize FIRMS, EOG, and validation inputs.
2. Aggregate detections into 1 km source cells.
3. Group validation by 10 km spatial block to reduce source-fragment leakage.
4. Build persistence, intensity, spectral, timing, seasonality, sensor, and spread
   features from MODIS and VIIRS S-NPP.
5. Acquire Sentinel-2 and WorldCover context for a bounded foreign-country panel.
6. Evaluate tabular, temporal, image-only, and fused branches with country-held-out
   outer evaluation and training-only threshold selection.
7. Confirm the selected fusion policy with fresh seeds and fixed acceptance rules.

The temporal TCN branch did not justify replacing the tabular baseline. Image-only models
also underperformed. CV became useful only as a guarded residual combined with compact
tabular evidence.

## Repository map

| Path | Contents |
|---|---|
| `kaggle/` | Reusable pipeline implementations for clustering, features, evaluation, CV, temporal models, fusion, and India inference |
| `notebooks/kaggle/` | Importable Kaggle notebooks for the main numbered runs |
| `docs/runs/` | Exact run instructions and input contracts |
| `docs/data/` | Imagery and external-data request specification |
| `results/` | Curated CSV, JSON, compact parquet, and QA evidence from completed runs |
| `artifacts/nb12b/` | Frozen final NB12b models, calibration, embeddings, and morphology inputs |
| `output/pdf/` | Current plain-text project report |
| `src/` | Local audit utilities |
| `tests/` | Leakage, schema, evaluation, and artifact-contract tests |
| `docs/archive/` | Superseded narrative documents retained only for history |

Large raw detections, feature tables, image chips, notebook logs, and population prediction
files are intentionally excluded from Git. The compact evidence retained in `results/`
is sufficient to audit every reported number.

## Running on Kaggle

Import the required notebook from `notebooks/kaggle/`, attach the saved inputs named in
its matching `docs/runs/` file, and use Save and Run All. NB12b is the final confirmation
run. It reuses cached image embeddings and morphology features, so its short CPU runtime
does not include end-to-end image encoding.

## Important data details

- `acq_time` must be read as a zero-padded HHMM string.
- NOAA-20 files use the `viirs-jpss1_*` naming pattern.
- MODIS confidence is numeric, while VIIRS confidence is categorical.
- VIIRS MIR brightness saturates at 367 K and saturation must be modeled explicitly.
- The final scores measure agreement with incomplete gas-flare labels, not verified
  accuracy for every industrial source category.

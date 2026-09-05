# Model and experiment architecture

Status: frozen NB12b models; NB13 India transfer completed on 2026-09-05.
See [RESULTS.md](RESULTS.md) for measurements and [docs/README.md](docs/README.md)
for the run and evidence index.

## Task and supervision

The system ranks persistent thermal sources constructed from NASA FIRMS detections.
Its supervised target is agreement with active EOG gas-flare locations. Unmatched
sources are unlabelled; they can include industrial facilities absent from EOG.
The current deliverable is an experimental ranking pipeline with saved models and
evaluation evidence. There is no production API, web app, or validated multiclass
industrial-source classifier in this repository.

## Data flow and module ownership

| Step | Input | Implementation | Output |
|---|---|---|---|
| Audit | FIRMS files and EOG inventory | `src/`, `kg_common.py` | schema and quality evidence |
| Candidate construction | FIRMS detections | `kg_02_source_clustering.py` | 1 km sources and 10 km block IDs |
| Thermal features | clustered MODIS and VIIRS S-NPP | `kg_03_features.py` | common-window source feature tables |
| Tabular development | six foreign feature populations | `kg_05*.py` | grouped and country-held-out experiments |
| Image acquisition | source centers | `kg_imagery_io.py`, `kg_07_context.py` | validated image chips, metadata, 77 summaries |
| Fusion baseline | thermal features and imagery summaries | `kg_08_fusion.py` | thermal, image, early-fusion and late-fusion evidence |
| Temporal experiments | detection histories and source context | `kg_10_temporal_tcn.py`, `kg_11_multimodal.py` | temporal descriptors and TCN comparisons |
| CV discovery | image chips and common-window features | `kg_12_cv_tabular.py` | CNN/morphology probes and guarded fusion |
| Confirmation | same foreign panel, fresh seeds | `kg_12b_confirmatory.py` | frozen ensemble and calibration references |
| India transfer | NB10 feature cache and fixed India panel | `kg_13_india_guarded.py` | image batches, frozen scores and ranking audit |

Names in the table refer to files under `kaggle/` unless a directory is specified.
Stage numbers are historical identifiers: `kg_08_fusion.py` was run as NB9,
`kg_05e_domain_revamp.py` as NB8. Use the notebook registry for execution names.

## Temporal and spatial contracts

Common-window features use 2022, 2023 and 2024 detections from MODIS and VIIRS S-NPP.
NOAA-20 features remain excluded because coverage is missing for Angola and Indonesia.
Source cells use a 1 km grid; 10 km blocks reduce leakage between nearby fragments
when spatial grouping is used. Country-held-out evaluation separates Algeria,
Angola, Indonesia, Iraq, Libya, and Nigeria during development.

The thermal branch uses 17 direct features and 22 intensity percentile features.
Percentiles are computed within each country's entire common-window candidate table
before selecting the imagery panel. This is label-free use of the target country's
feature distribution, not calibration learned on India labels. Inference therefore
needs the same reference population or a separately designed percentile contract.

Coordinates are used for source construction, EOG matching, image retrieval, and review
maps. IDs and labels are retained for joins and evaluation. Coordinates, country, NASA
`type`, EOG distance, and labels are excluded from model feature arrays.

## Image representation

Each chip covers a 2 km square at a 10 m grid (200 x 200 pixels). Sentinel-2 collection
`sentinel-2-c1-l2a` is queried over 2022-2024. The acquisition procedure tries up to
eight scenes, ordered by scene cloud coverage. It requires at least 80% clear pixels
in the full chip and central 500 m square, accepting SCL classes 4, 5 and 6.

The six bands are blue, green, red, NIR, SWIR16 and SWIR22. SWIR is resampled from
20 m. WorldCover v200 supplies 2021 categorical land cover. A chip uses one accepted
Sentinel scene rather than a temporal image composite. Missing WorldCover is excluded
from class-fraction denominators; absence of built-up pixels gives a missing distance.

## Frozen NB12b ensemble

| Component | Features | Model and role |
|---|---:|---|
| Compact early-fusion baseline | 39 thermal + 30 image summaries = 69 | nine regularized LightGBM models |
| Frozen image encoder | RGB full chip and center 1 km view | Sentinel-2 MoCo ResNet-18, 512 features per view |
| Visual probe | 1,024 CNN values plus 77 image and 41 morphology features | saved scaling, PCA to 12 CNN components, PU-bagged logistic regression |
| Guarded fusion | compact and visual probabilities | three separately calibrated seed pipelines, averaged |

The compact baseline is not thermal-only. Its 30 image summaries include spectral
indices, reflectance, WorldCover fractions and distance to built-up pixels.
The visual probe normalizes each CNN view, applies training-fitted scaling and PCA,
and appends 118 auxiliary features with training-fitted imputation and scaling.
The image encoder is frozen and uses four deterministic augmentations per view.

Confirmation seeds are 272, 273 and 274. Each seed has three full-foreign compact
models and one saved visual PU model containing five logistic-regression bags.
The nine compact models are not nine outer-country fold checkpoints.

For seed s, let c be the mean probability from that seed's three compact models,
and v its visual probability. The saved foreign OOF references define separate
logit standardizations, followed by sigmoid, giving calibrated c and v. Fusion is:

```text
seed_score = c + alpha * 4*c*(1-c) * (v-c)
final_guarded_score = mean(seed_score for seeds 272, 273, 274)
```

Every final alpha is 0.5. Each seed is calibrated and fused before averaging.
The baseline comparison uses the raw mean of the nine compact probabilities.
The authoritative feature order is in
[`12b_selected_schema.json`](results/nb12b_confirmatory/12b_selected_schema.json).
The LightGBM files use positional `Column_0` names, so the schema is essential.

The residual policy can fall back to calibrated compact scores if the visual
residual is unavailable. That does not supply missing compact image features.
NB13 consequently scores only sources with accepted imagery; it does not claim
validated whole-country scoring with image features replaced by missing values.

## Validation and decision boundaries

NB12 discovers the visual residual on a 294-source foreign panel. NB12b uses fresh
seeds on that same panel, with transformations and tuning restricted to outer-fold
training countries. Its 12 acceptance checks confirm stability under predefined
margins. This reuses the panel and is not independent replication.

NB13 freezes 300 sources before requesting images, with up to 96 EOG-positive sites
and one selected source per site and block. The actual panel contains 72 positives;
299 sources were scored. It loads the NB12b models without fitting. The India rule
retains guarded ranking if PR-AUC falls by no more than 0.01 and at most one positive
site is lost in the top-20% review queue. It passed, selecting guarded for India.
That selection itself uses India metrics, so a future selected-policy assessment
requires independent data. NB10 had also already exposed India at project level.

No threshold is deployment-calibrated. Foreign F1 was evaluated with fold-local
thresholds; NB13 therefore reports ranking and review-budget metrics without F1.
Panel enrichment changes prevalence. None of these panel results estimates population
precision, and the bootstrap intervals for CV improvement include zero.

## Execution and persistence

NB13 needs the saved NB10 output containing `features_India_2022_2024.parquet`.
NB2 contains foreign features only. Continuation runs also attach the immediately
previous NB13 saved output with its `.npz` chips. A continuation may have a different
Kaggle notebook title; keep its panel configuration and working root unchanged.

Native run state, input and panel hashes, per-chip metadata, and atomic writes support
resumption. The three completed batches attempted 100 new sources each. The final
ZIP contains compact audit outputs but excludes image chips. Retaining only that ZIP
is insufficient to resume image acquisition or reproduce encoder extraction.

Models, schema and calibration are checked against the NB12b manifest. The encoder
checkpoint is checked against a fixed SHA-256. The notebook pins scikit-learn 1.6.1,
LightGBM 4.6.0, and joblib 1.5.3 before scoring. Cached feature and model hashes allow
audit without rerunning the external imagery acquisition.

## Storage and current boundaries

`kaggle/` owns reusable code; `notebooks/kaggle/` owns notebook entry points;
`docs/runs/` owns operational instructions; `results/` owns compact native evidence;
`artifacts/nb12b/` owns the frozen models and foreign feature caches. India evaluation
caches are kept beside its predictions under `results/nb13_india_transfer/` so its
native output manifest can be verified without rewriting paths.

Large feature populations, raw detections, image chips and logs remain in saved Kaggle
outputs or local downloads outside Git. Original download folders are preserved.
See [docs/FILE_LAYOUT.md](docs/FILE_LAYOUT.md) for their mappings and recovery limits.

Before deployment, the remaining work includes a population screening design, an
independent review set, facility-category labels, threshold calibration on separate
validation data, and candidate-coverage assessment. No comparison establishes that
this approach outperforms Solution 2 on equivalent data.
